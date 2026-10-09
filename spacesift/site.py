"""Export data for the public Transit Lab (spacesift.vercel.app/lab/).

experiments.json: every experiment's record (trimmed), analyses, notes, and plot URLs.
stars/<kic>.json: pre-saved Kepler light curves for the in-browser demo, with each
star's radius, mass and white-noise level (cleaned point-to-point scatter).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import json_safe

# Real stars offered in the browser demo: (KIC, label, why it is interesting).
DEMO_STARS = [
    ("1868918", "Quiet star", "A quiet Sun-like star from SS-0001: planets are easy to find here."),
    ("6205852", "Pulsator, 4.6 h", "SS-0001 found a 4.6-hour oscillation that hides injected planets from biweight cleaning."),
    ("4280576", "Pulsator, 3.7 h", "A 3.7-hour oscillation whose amplitude changes from quarter to quarter."),
]
DEMO_FROM_CELLS = [  # one star from each of these SS-0002B bins
    ("spots-P3d-A300", "Spotted star, 3-day rotation", "Fast rotation with large spots: the hardest case for every method in SS-0002A."),
    ("spots-P10d-A1000", "Spotted star, 10-day rotation", "Slow rotation: biweight cleaning should cope."),
    ("dsct", "Delta Scuti pulsator", "A real multi-mode pulsator (Murphy et al. 2019): messier than SS-0002A's simulated ones."),
]


def _experiment_entry(d: Path, url_prefix: str) -> dict:
    r = json.loads(d.joinpath("record.json").read_text())
    keep = ("id", "question", "n_trials", "n_stars", "n_stars_failed", "git_commit", "git_dirty", "seed",
            "started", "finished", "preprocessing_version", "n_shards")
    entry = {k: r.get(k) for k in keep}
    s = r.get("summary") or {}
    entry["summary"] = {spec: {k: v for k, v in val.items() if k in ("status_counts", "gamma_cdf_fit", "depth",
                                                                   "threshold_for_far", "n_null_searches")}
                        for spec, val in s.items() if isinstance(val, dict)}
    entry["notes"] = d.joinpath("NOTES.md").read_text(encoding="utf-8") if d.joinpath("NOTES.md").exists() else ""
    entry["analyses"] = {}
    for a in sorted(d.glob("analysis/*/analysis.json")):
        an = json.loads(a.read_text())
        entry["analyses"][a.parent.name] = {k: an.get(k) for k in ("threshold_source", "thresholds", "sde_threshold",
                                                                    "excluded_stars", "control_completeness", "snr_band")}
    entry["images"] = [f"{url_prefix}/{d.name}/{p.relative_to(d).as_posix()}"
                       for p in sorted(d.rglob("*.png")) if "shards" not in p.parts]
    return entry


def export_experiments(experiments: Path, out: Path, url_prefix: str) -> list[dict]:
    entries = [_experiment_entry(rec.parent, url_prefix) for rec in sorted(experiments.glob("*/record.json"))
               if not rec.parent.name.startswith("_")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(entries), indent=1))
    return entries


def export_demo_stars(project_root: Path, out_dir: Path, quarters=(2, 3, 4, 5)) -> list[dict]:
    from .data import load_mission
    from .detrend import flatten
    from .inject import p2p_noise

    lists = [pd.read_csv(p, dtype={"star_id": str}) for p in sorted((project_root / "configs" / "stars").glob("*.csv"))]
    stars = pd.concat(lists, ignore_index=True).drop_duplicates("star_id").set_index("star_id")
    picks = list(DEMO_STARS)
    if "cell" in stars:
        for cell, label, why in DEMO_FROM_CELLS:
            hit = stars[stars.cell == cell]
            if len(hit):
                picks.append((hit.index[0], label, why))
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for kic, label, why in picks:
        lc = load_mission(kic, "kepler", list(quarters), project_root / "cache")
        sigma = p2p_noise(flatten(lc.time, lc.flux, "prewhiten+biweight-0.5"))
        row = stars.loc[kic] if kic in stars.index else None
        meta = {"kic": kic, "label": label, "why": why, "quarters": list(quarters),
                "radius": float(row.radius) if row is not None else 1.0,
                "mass": float(row.mass) if row is not None else 1.0,
                "sigma_ppm": round(sigma * 1e6, 2), "n": int(lc.time.size)}
        t0 = float(lc.time[0])
        payload = {**meta, "t0_bkjd": t0, "time": np.round(lc.time - t0, 5).tolist(),
                   "flux_ppm": np.round((lc.flux - 1) * 1e6, 1).tolist()}
        (out_dir / f"{kic}.json").write_text(json.dumps(payload, separators=(",", ":")))
        index.append(meta)
        print(f"  KIC {kic}: {label} ({lc.time.size} points, noise {sigma * 1e6:.0f} ppm)")
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index
