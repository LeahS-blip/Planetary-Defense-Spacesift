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
    # Synthetic only: the same light curve without its stellar variability (white noise
    # around 1), so SNR can be defined against photon noise alone.
    noise_only: np.ndarray | None = None

    def __post_init__(self):
        good = np.isfinite(self.time) & np.isfinite(self.flux)
        order = np.argsort(self.time[good])
        self.time = np.asarray(self.time[good][order], dtype=float)
        self.flux = np.asarray(self.flux[good][order], dtype=float)
        if self.noise_only is not None:
            self.noise_only = np.asarray(self.noise_only[good][order], dtype=float)


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


def _smooth_noise(rng: np.random.Generator, time: np.ndarray, timescale: float) -> np.ndarray:
    """Unit-variance random curve that varies on `timescale` days (cubic spline through knots)."""
    from scipy.interpolate import CubicSpline

    knots = np.arange(time.min() - timescale, time.max() + 2 * timescale, timescale)
    return CubicSpline(knots, rng.normal(0, 1, knots.size))(time)


def stellar_variability(rng: np.random.Generator, time: np.ndarray, kind: str, period_d: float,
                        amp_ppm: float) -> np.ndarray:
    """Fractional flux variation. amp_ppm = semi-amplitude of the fundamental.

    pulsation: coherent, strictly periodic; fundamental plus a first harmonic at 0.3x.
    spots: quasi-periodic rotation; both harmonics' amplitudes (+-40%) and phases
    (~1 rad) drift on a timescale of 3 rotations, as spots grow and decay.
    """
    if kind == "none" or amp_ppm == 0:
        return np.zeros_like(time)
    amp = amp_ppm * 1e-6
    w = 2 * np.pi * time / period_d
    if kind == "pulsation":
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        return amp * (np.sin(w + p1) + 0.3 * np.sin(2 * w + p2))
    if kind == "spots":
        tau = 3 * period_d
        a1 = np.clip(1 + 0.4 * _smooth_noise(rng, time, tau), 0, None)
        a2 = np.clip(1 + 0.4 * _smooth_noise(rng, time, tau), 0, None)
        ph1, ph2 = _smooth_noise(rng, time, tau), _smooth_noise(rng, time, tau)
        return amp * (a1 * np.sin(w + ph1) + 0.5 * a2 * np.sin(2 * w + ph2))
    raise ValueError(f"unknown variability kind {kind!r}")


def synthetic_variable(rng: np.random.Generator, baseline: float, noise_ppm: float, kind: str, period_d: float,
                       amp_ppm: float, gap_fraction: float = 0.05,
                       cadence: float = KEPLER_LONG_CADENCE) -> LightCurve:
    """White noise + one kind of stellar variability, with Kepler-like gaps; keeps the noise-only curve."""
    time = np.arange(0.0, baseline, cadence)
    n_gaps = max(1, int(baseline / 90))
    keep = np.ones(time.size, bool)
    for _ in range(n_gaps):
        start = rng.uniform(0, baseline)
        keep &= ~((time > start) & (time < start + gap_fraction * baseline / n_gaps))
    time = time[keep]
    noise = rng.normal(0, noise_ppm * 1e-6, time.size)
    var = stellar_variability(rng, time, kind, period_d, amp_ppm)
    return LightCurve(time, 1 + var + noise, cadence, noise_only=1 + noise)


CELL_COLUMNS = ("cell", "var_kind", "var_period_d", "var_amp_ppm")


def read_stars(path: Path) -> list[Star]:
    """Frozen star list. Required column: star_id. Optional: radius, mass, teff, u1, u2.

    Optional cell columns (cell, var_kind, var_period_d, var_amp_ppm) label each star's
    variability bin, so real stars can be analysed like a synthetic grid (SS-0002B).
    """
    df = pd.read_csv(path, dtype={"star_id": str})
    fields = {"radius", "mass", "teff", "u1", "u2"}
    stars = []
    for row in df.to_dict("records"):
        kw = {k: float(v) for k, v in row.items() if k in fields and pd.notna(v)}
        if kw.get("radius", 1.0) <= 0 or kw.get("mass", 1.0) <= 0:
            raise ValueError(f"{path}: star {row['star_id']} has non-positive radius or mass")
        meta = {k: row[k] for k in CELL_COLUMNS if k in row and pd.notna(row[k])}
        stars.append(Star(star_id=str(row["star_id"]), meta=meta, **kw))
    return stars


# Part A's spot cells, as bins: rotation period (d) and semi-amplitude (ppm), edges halfway
# between cell centres in log space.
SPOT_PERIOD_BINS = {1.0: (0.5, 2.0), 3.0: (2.0, 5.0), 10.0: (5.0, 15.0), 30.0: (15.0, 45.0)}
SPOT_AMP_BINS = {30.0: (17.0, 55.0), 100.0: (55.0, 170.0), 300.0: (170.0, 550.0), 1000.0: (550.0, 1700.0)}


def _dr25_quiet_pool(kepmag: tuple[float, float]) -> pd.DataFrame:
    """DR25 stellar parameters for stars with >700 d of data and no KOI."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as NEA

    stellar = NEA.query_criteria(
        table="q1_q17_dr25_stellar",
        select="kepid,teff,logg,radius,mass,kepmag,dataspan",
        where=f"kepmag between {kepmag[0]} and {kepmag[1]} and dataspan > 700",
    ).to_pandas()
    kois = NEA.query_criteria(table="q1_q17_dr25_koi", select="kepid").to_pandas()
    stellar = stellar[~stellar.kepid.isin(kois.kepid)].dropna(subset=["radius", "mass"])
    return stellar[(stellar.mass > 0) & (stellar.radius > 0)]


def select_variable_stars(out: Path, per_bin: int = 30, n_pulsators: int = 60, seed: int = 2,
                          kepmag: tuple[float, float] = (11.0, 14.0)) -> pd.DataFrame:
    """SS-0002B sample: spotted stars binned like Part A's spot cells, plus delta Scuti pulsators.

    Spots: McQuillan, Mazeh & Aigrain (2014) rotation periods; their Rper is the 5-95%
    flux range, about twice a sinusoid's semi-amplitude, so semi-amplitude = Rper / 2.
    Pulsators: stars flagged delta Scuti by Murphy et al. (2019). Their periods and
    amplitudes are measured during the run. Every star: DR25 parameters, no KOI.
    """
    from astroquery.vizier import Vizier

    pool = _dr25_quiet_pool(kepmag)
    viz = Vizier(row_limit=-1, columns=["KIC", "Prot", "Rper"])
    rot = viz.get_catalogs("J/ApJS/211/24/table1")[0].to_pandas()
    rot = rot.merge(pool, left_on="KIC", right_on="kepid")
    rot = rot[(rot.logg > 4.0) & rot.teff.between(4000, 6500)]
    rot["amp"] = rot.Rper / 2
    picks = []
    for pc, (plo, phi) in SPOT_PERIOD_BINS.items():
        for ac, (alo, ahi) in SPOT_AMP_BINS.items():
            b = rot[rot.Prot.between(plo, phi, inclusive="left") & rot.amp.between(alo, ahi, inclusive="left")]
            b = b.sample(n=min(per_bin, len(b)), random_state=seed)
            picks.append(b.assign(var_kind="spots", var_period_d=pc, var_amp_ppm=ac,
                                  cell=f"spots-P{pc:g}d-A{ac:g}", cat_period_d=b.Prot, cat_amp_ppm=b.amp))
    dsct = Vizier(row_limit=-1, columns=["KIC", "dSct"]).get_catalogs("J/MNRAS/485/2380/table1")[0].to_pandas()
    dsct = dsct[dsct.dSct.astype(str).str.strip().str.upper().isin(["Y", "1", "TRUE"])]
    dsct = dsct.merge(pool, left_on="KIC", right_on="kepid")
    dsct = dsct.sample(n=min(n_pulsators, len(dsct)), random_state=seed)
    picks.append(dsct.assign(var_kind="dsct", var_period_d=0.0, var_amp_ppm=0.0, cell="dsct",
                             cat_period_d=np.nan, cat_amp_ppm=np.nan))
    sel = pd.concat(picks, ignore_index=True)
    df = pd.DataFrame({
        "star_id": sel.kepid.astype(int).astype(str), "radius": sel.radius, "mass": sel.mass,
        "teff": sel.teff, "kepmag": sel.kepmag, "cell": sel.cell, "var_kind": sel.var_kind,
        "var_period_d": sel.var_period_d, "var_amp_ppm": sel.var_amp_ppm,
        "cat_period_d": sel.cat_period_d, "cat_amp_ppm": sel.cat_amp_ppm,
    }).drop_duplicates("star_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


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
