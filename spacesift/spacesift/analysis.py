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


def rematch(trials: pd.DataFrame, sde_threshold: float, period_tol: float, min_transits: int,
            star_period: pd.Series | None = None) -> pd.Series:
    """Re-derive each trial's status. star_period (aligned with trials) enables 'star_signal'."""
    inj_cols = [c for c in trials.columns if c.startswith("inj_")]
    found_cols = [c for c in trials.columns if c.startswith("found_")]

    def one(row):
        inj = Injection(**{c[4:]: row[c] for c in inj_cols})
        has_cand = pd.notna(row["found_sde"])
        cand = Candidate(**{c[6:]: row[c] for c in found_cols}) if has_cand else None
        sp = star_period.loc[row.name] if star_period is not None else None
        return match(inj, cand, sde_threshold=sde_threshold, period_tol=period_tol, min_transits=min_transits,
                     star_period=sp if sp is not None and pd.notna(sp) else None)

    return trials.apply(one, axis=1)


def analyze_grid(exp: Path, far: float = 0.01, snr_band: tuple[float, float] = (10.0, 16.0),
                 name: str = "grid", null_exp: Path | None = None, fixed_threshold: float | None = None) -> Path:
    """SS-0002-style grid analysis: one SDE threshold per method, statuses re-derived with
    'star_signal', and per-cell completeness, capture rate and false-alarm rate.

    Thresholds (one per method, applied to every cell, as a pipeline cannot know a
    star's variability in advance) come from, in order of preference:
      null_exp        a false-alarm experiment on noise-only stars, at rate `far`
                      (the calibration a pipeline would do on quiet stars);
      fixed_threshold one SDE for every method (previews only);
      the grid's own inverted searches pooled over all cells. Avoid: strongly
      variable stars score high even inverted, so in a grid where half the cells
      vary strongly this sets a threshold far above the noise (SS-0002A: SDE ~24).
    The per-cell false-alarm rate then measures false alarms caused by variability.
    """
    from .evaluate import wilson

    cfg, _ = load_config(exp / "config.yaml")
    trials = pd.read_parquet(exp / "trials.parquet")
    base = pd.read_parquet(exp / "baselines.parquet")
    inverted = base[base.baseline == "inverted"]
    if null_exp is not None:
        null = json.loads((null_exp / "record.json").read_text())["summary"]
        missing = set(cfg.detrend) - set(null)
        if missing:
            raise SystemExit(f"{null_exp.name} has no false-alarm threshold for {sorted(missing)}")
        thresholds = {spec: float(null[spec]["threshold_for_far"][str(far)]) for spec in cfg.detrend}
        source = f"{null_exp.name} at false-alarm rate {far}"
    elif fixed_threshold is not None:
        thresholds = {spec: float(fixed_threshold) for spec in cfg.detrend}
        source = "fixed (preview only)"
    else:
        thresholds = {spec: float(np.quantile(g.found_sde, 1 - far)) for spec, g in inverted.groupby("detrend")}
        source = "pooled grid inverted searches (biased high by variable stars)"
    plain = base[base.baseline == "plain"].set_index(["star_id", "detrend"]).found_period

    r = cfg.recovery
    parts = []
    for spec, d in trials.groupby("detrend"):
        sp = pd.Series([plain.get((s, spec)) for s in d.star_id], index=d.index)
        d = d.copy()
        d["status"] = rematch(d, thresholds[spec], r.period_tol, r.min_transits, star_period=sp)
        parts.append(d)
    trials = pd.concat(parts).sort_index()

    obs = trials[trials.status != "not_observable"]
    band = obs[(obs.inj_expected_snr >= snr_band[0]) & (obs.inj_expected_snr < snr_band[1])]
    keys = ["detrend", "cell", "var_kind", "var_period_d", "var_amp_ppm"]
    g_band = band.groupby(keys)
    cells = pd.DataFrame({
        "n_band": g_band.size(),
        "k_band": g_band.apply(lambda x: (x.status == "recovered").sum(), include_groups=False),
    })
    g_all = obs.groupby(keys)
    cells["capture_rate"] = g_all.apply(lambda x: (x.status == "star_signal").mean(), include_groups=False)
    cells["completeness_all_snr"] = g_all.apply(lambda x: (x.status == "recovered").mean(), include_groups=False)
    cells["depth_kept"] = g_all.detrended_depth_ratio.median()
    cells = cells.reset_index()
    cells["completeness"] = cells.k_band / cells.n_band
    cells["lo68"], cells["hi68"] = wilson(cells.k_band, cells.n_band)
    fa = (inverted.assign(false_alarm_rate=inverted.found_sde > inverted.detrend.map(thresholds))
          .groupby(["detrend", "cell"]).false_alarm_rate.mean().reset_index())
    cells = cells.merge(fa, on=["detrend", "cell"], how="left")

    out = exp / "analysis" / name
    out.mkdir(parents=True, exist_ok=True)
    trials.to_parquet(out / "trials.parquet")
    cells.to_csv(out / "cells.csv", index=False)
    summary = {
        "experiment": cfg.id,
        "false_alarm_rate": far,
        "snr_band": list(snr_band),
        "thresholds": thresholds,
        "threshold_source": source,
        "status_counts": {spec: d.status.value_counts().to_dict() for spec, d in trials.groupby("detrend")},
        "control_completeness": cells[cells.var_kind == "none"].set_index("detrend").completeness.round(3).to_dict(),
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=2))
    from .plots import plot_grid
    plot_grid(cells, trials, out)
    print(f"thresholds: { {k: round(v, 2) for k, v in thresholds.items()} }; wrote {out}")
    return out


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
