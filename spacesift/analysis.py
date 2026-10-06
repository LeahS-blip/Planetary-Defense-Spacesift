"""Re-analyse a finished injection experiment without recomputing searches.

Every trial stores its injected parameters and its search result, so the
recovery status can be re-derived under a different SDE threshold, and stars
can be excluded, after the fact. Results go to <exp>/analysis/<name>/; the
experiment's own record and trials are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .evaluate import STATUSES, match
from .inject import Injection
from .runner import summarize
from .search import Candidate


def rematch(trials: pd.DataFrame, sde_threshold: float, period_tol: float, min_transits: int) -> pd.Series:
    inj_cols = [c for c in trials.columns if c.startswith("inj_")]
    found_cols = [c for c in trials.columns if c.startswith("found_")]

    def one(row):
        inj = Injection(**{c[4:]: row[c] for c in inj_cols})
        has_cand = pd.notna(row["found_sde"])
        cand = Candidate(**{c[6:]: row[c] for c in found_cols}) if has_cand else None
        return match(inj, cand, sde_threshold=sde_threshold, period_tol=period_tol, min_transits=min_transits)

    return trials.apply(one, axis=1)


def analyze(exp: Path, name: str, sde_threshold: float | None = None, null_exp: Path | None = None,
            far: float = 0.01, exclude: list[str] | None = None) -> Path:
    """Re-derive statuses. The threshold comes from sde_threshold, or else from the null
    experiment's threshold for the given false-alarm rate. Stars are excluded if listed,
    or if their pre-injection search already beats the threshold (an existing signal)."""
    cfg, _ = load_config(exp / "config.yaml")
    trials = pd.read_parquet(exp / "trials.parquet")
    stars = pd.read_parquet(exp / "stars.parquet")
    source = "explicit"
    if sde_threshold is None:
        if null_exp is None:
            raise SystemExit("give --sde-threshold or --null")
        null = json.loads((null_exp / "record.json").read_text())["summary"]
        spec = cfg.detrend[0]
        sde_threshold = null[spec]["threshold_for_far"][str(far)]
        source = f"{null_exp.name} at false-alarm rate {far}"

    flagged = stars[stars.pre_sde > sde_threshold]
    excluded = sorted(set(exclude or []) | set(flagged.star_id))
    kept = trials[~trials.star_id.isin(excluded)].copy()
    r = cfg.recovery
    kept["status"] = rematch(kept, sde_threshold, r.period_tol, r.min_transits)

    out = exp / "analysis" / name
    out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out / "trials.parquet")
    summary = {
        "experiment": cfg.id,
        "sde_threshold": float(sde_threshold),
        "threshold_source": source,
        "excluded_stars": excluded,
        "excluded_reason": {s: ("pre-injection SDE above threshold" if s in set(flagged.star_id) else "listed")
                            for s in excluded},
        "n_stars": int(kept.star_id.nunique()),
        "n_trials": int(len(kept)),
        "summary": summarize(kept),
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2))
    from .plots import plot_experiment
    plot_experiment(kept, out / "plots")
    print(f"threshold {sde_threshold:.2f} ({source}); excluded {len(excluded)} stars; wrote {out}")
    return out
