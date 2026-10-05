"""Light-curve loading (Kepler/TESS via lightkurve, or synthetic) and star selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .inject import Star

KEPLER_LONG_CADENCE = 29.4244 / 1440  # days
TESS_2MIN = 2.0 / 1440


@dataclass
class LightCurve:
    time: np.ndarray  # days (BKJD for Kepler, BTJD for TESS)
    flux: np.ndarray  # normalised to median 1
    exptime: float  # days

    def __post_init__(self):
        good = np.isfinite(self.time) & np.isfinite(self.flux)
        order = np.argsort(self.time[good])
        self.time = np.asarray(self.time[good][order], dtype=float)
        self.flux = np.asarray(self.flux[good][order], dtype=float)


def load_mission(star_id: str, mission: str, quarters: list[int] | None, cache_dir: Path) -> LightCurve:
    """PDCSAP flux, quality-masked, normalised per quarter/sector, stitched. Cached as Parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "all" if not quarters else "-".join(map(str, quarters))
    path = cache_dir / f"{mission}_{star_id}_{tag}.parquet"
    exptime = KEPLER_LONG_CADENCE if mission == "kepler" else TESS_2MIN
    if path.exists():
        df = pd.read_parquet(path)
        return LightCurve(df.time.to_numpy(), df.flux.to_numpy(), exptime)

    import lightkurve as lk

    if mission == "kepler":
        res = lk.search_lightcurve(f"KIC {star_id}", mission="Kepler", author="Kepler", exptime="long")
        if quarters:
            res = res[np.isin(res.table["mission"], [f"Kepler Quarter {q:02d}" for q in quarters])]
    elif mission == "tess":
        res = lk.search_lightcurve(f"TIC {star_id}", mission="TESS", author="SPOC", exptime=120)
        if quarters:  # sectors, for TESS
            res = res[np.isin(res.table["mission"], [f"TESS Sector {s:02d}" for s in quarters])]
    else:
        raise ValueError(f"unknown mission {mission!r}")
    if len(res) == 0:
        raise LookupError(f"no {mission} light curves for {star_id} (quarters={quarters})")

    coll = res.download_all(flux_column="pdcsap_flux", quality_bitmask="default")
    parts = [lc.remove_nans().normalize() for lc in coll]
    time = np.concatenate([p.time.value for p in parts])
    flux = np.concatenate([np.asarray(p.flux.value, dtype=float) for p in parts])
    pd.DataFrame({"time": time, "flux": flux}).to_parquet(path)
    return LightCurve(time, flux, exptime)


def synthetic(
    rng: np.random.Generator,
    baseline: float,
    noise_ppm: float,
    var_amp_ppm: float = 0.0,
    var_period: float = 10.0,
    gap_fraction: float = 0.05,
    cadence: float = KEPLER_LONG_CADENCE,
) -> LightCurve:
    """White noise plus a drifting two-harmonic 'rotation' signal and random data gaps."""
    time = np.arange(0.0, baseline, cadence)
    n_gaps = max(1, int(baseline / 90))  # roughly one gap per quarter, like Kepler downlinks
    keep = np.ones(time.size, bool)
    for _ in range(n_gaps):
        start = rng.uniform(0, baseline)
        keep &= ~((time > start) & (time < start + gap_fraction * baseline / n_gaps))
    time = time[keep]
    phase0 = rng.uniform(0, 2 * np.pi, 2)
    p = var_period * (1 + 0.02 * np.sin(2 * np.pi * time / (5 * var_period)))
    var = var_amp_ppm * 1e-6 * (
        np.sin(2 * np.pi * time / p + phase0[0]) + 0.4 * np.sin(4 * np.pi * time / p + phase0[1])
    )
    flux = 1 + var + rng.normal(0, noise_ppm * 1e-6, time.size)
    return LightCurve(time, flux, cadence)


def read_stars(path: Path) -> list[Star]:
    """Frozen star list. Required column: star_id. Optional: radius, mass, teff, u1, u2."""
    df = pd.read_csv(path, dtype={"star_id": str})
    fields = {"radius", "mass", "teff", "u1", "u2"}
    stars = []
    for row in df.to_dict("records"):
        kw = {k: float(v) for k, v in row.items() if k in fields and pd.notna(v)}
        if kw.get("radius", 1.0) <= 0 or kw.get("mass", 1.0) <= 0:
            raise ValueError(f"{path}: star {row['star_id']} has non-positive radius or mass")
        stars.append(Star(star_id=str(row["star_id"]), **kw))
    return stars


def select_kepler_stars(n: int, seed: int, out: Path, kepmag=(11.0, 13.0), teff=(4500.0, 6500.0)) -> pd.DataFrame:
    """Quiet FGK dwarfs from the DR25 stellar table with no KOIs; frozen to `out`.

    'Quiet' = 6-hr CDPP below the median of the magnitude/temperature-cut sample.
    The archive is queried once here; experiments only ever read the frozen CSV.
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA

    stellar = NEA.query_criteria(
        table="q1_q17_dr25_stellar",
        select="kepid,teff,logg,radius,mass,kepmag,rrmscdpp06p0,dataspan",
        where=(
            f"kepmag between {kepmag[0]} and {kepmag[1]} and teff between {teff[0]} and {teff[1]} "
            "and logg > 4.1 and dataspan > 700"
        ),
    ).to_pandas()
    kois = NEA.query_criteria(table="q1_q17_dr25_koi", select="kepid").to_pandas()
    stellar = stellar[~stellar.kepid.isin(kois.kepid)].dropna(subset=["rrmscdpp06p0", "radius", "mass"])
    # Some DR25 rows carry placeholder parameters (mass 0); the transit geometry needs real ones.
    stellar = stellar[(stellar.mass > 0) & (stellar.radius > 0)]
    quiet = stellar[stellar.rrmscdpp06p0 < stellar.rrmscdpp06p0.median()]
    pick = quiet.sample(n=min(n, len(quiet)), random_state=seed).sort_values("kepid")
    df = pd.DataFrame({
        "star_id": pick.kepid.astype(int).astype(str),
        "radius": pick.radius,
        "mass": pick.mass,
        "teff": pick.teff,
        "kepmag": pick.kepmag,
        "cdpp6_ppm": pick.rrmscdpp06p0,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
