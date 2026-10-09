"""SS-0002 Part B: does the synthetic map (Part A) predict completeness on real stars?

predict(): before the run, write each spotted-star bin's predicted completeness per
method, read from the matching Part A cell, so the comparison is a genuine prediction.
compare(): after the run, set observed completeness per bin against the prediction.
delta Scuti stars have no catalogue period, so their prediction is read from the
frozen Part A map at each star's measured period and amplitude.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from . import json_safe
import pandas as pd

MIN_STARS = 10  # bins with fewer stars are reported but not judged


def predict(cells_a: Path, stars_csv: Path, out: Path) -> pd.DataFrame:
    a = pd.read_csv(cells_a)
    stars = pd.read_csv(stars_csv, dtype={"star_id": str})
    spots = stars[stars.var_kind == "spots"].groupby(["cell", "var_period_d", "var_amp_ppm"]).size()
    rows = []
    for (cell, p, amp), n in spots.items():
        match = a[(a.var_kind == "spots") & np.isclose(a.var_period_d, p) & np.isclose(a.var_amp_ppm, amp)]
        for r in match.itertuples():
            rows.append({"cell": cell, "var_kind": "spots", "var_period_d": p, "var_amp_ppm": amp,
                         "n_stars": int(n), "detrend": r.detrend, "pred": r.completeness,
                         "pred_lo68": r.lo68, "pred_hi68": r.hi68})
    pred = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    return pred


def _nearest_pulsation_cell(a: pd.DataFrame, period_d: float, amp_ppm: float) -> pd.DataFrame:
    """Part A pulsation cell nearest in log period and log amplitude (clipped to the grid)."""
    puls = a[a.var_kind == "pulsation"]
    periods = np.sort(puls.var_period_d.unique())
    amps = np.sort(puls.var_amp_ppm.unique())
    p = periods[np.argmin(np.abs(np.log(periods) - np.log(np.clip(period_d, periods[0], periods[-1]))))]
    m = amps[np.argmin(np.abs(np.log(amps) - np.log(np.clip(amp_ppm, amps[0], amps[-1]))))]
    return puls[np.isclose(puls.var_period_d, p) & np.isclose(puls.var_amp_ppm, m)]


def compare(exp_b: Path, analysis: str, cells_a: Path, predictions: Path) -> Path:
    from .evaluate import wilson

    out = exp_b / "analysis" / analysis
    obs = pd.read_csv(out / "cells.csv")
    pred = pd.read_csv(predictions)
    a = pd.read_csv(cells_a)

    spots = obs[obs.var_kind == "spots"].merge(pred[["cell", "detrend", "n_stars", "pred", "pred_lo68", "pred_hi68"]],
                                              on=["cell", "detrend"], how="left")

    # delta Scuti: per star, observed recoveries in the SNR band vs the Part A cell at its measured variability.
    trials = pd.read_parquet(out / "trials.parquet")
    stars = pd.read_parquet(exp_b / "stars.parquet").set_index("star_id")
    meta = json.loads((out / "analysis.json").read_text())
    lo, hi = meta["snr_band"]
    band = trials[(trials.var_kind == "dsct") & (trials.status != "not_observable")
                  & trials.inj_expected_snr.between(lo, hi, inclusive="left")]
    dsct_rows = []
    for (star_id, spec), g in band.groupby(["star_id", "detrend"]):
        s = stars.loc[star_id]
        cellA = _nearest_pulsation_cell(a, s.meas_var_period_d, s.meas_var_amp_ppm)
        p = cellA[cellA.detrend == spec]
        dsct_rows.append({"star_id": star_id, "detrend": spec, "n": len(g), "k": int((g.status == "recovered").sum()),
                          "pred": float(p.completeness.iloc[0]) if len(p) else np.nan,
                          "meas_period_h": s.meas_var_period_d * 24, "meas_amp_ppm": s.meas_var_amp_ppm})
    dsct = pd.DataFrame(dsct_rows)
    dsct_summary = pd.DataFrame()
    if len(dsct):
        g = dsct.groupby("detrend")
        dsct_summary = pd.DataFrame({"n_stars": g.star_id.nunique(), "n": g.n.sum(), "k": g.k.sum(),
                                     "pred": g.pred.mean()}).reset_index()
        dsct_summary["completeness"] = dsct_summary.k / dsct_summary.n
        dsct_summary["lo68"], dsct_summary["hi68"] = wilson(dsct_summary.k, dsct_summary.n)
        dsct_summary.insert(0, "cell", "dsct")

    # Agreement: within 2 sigma, combining both 68% intervals in quadrature.
    def judged(df):
        s_obs = (df.hi68 - df.lo68) / 2
        s_pred = ((df.pred_hi68 - df.pred_lo68) / 2) if "pred_hi68" in df else 0.0
        df = df.copy()
        df["diff"] = df.completeness - df.pred
        df["agrees_2sigma"] = df["diff"].abs() <= 2 * np.sqrt(s_obs**2 + s_pred**2)
        return df

    spots = judged(spots)
    spots["judged"] = spots.n_stars >= MIN_STARS
    if len(dsct_summary):
        dsct_summary = judged(dsct_summary)
        dsct_summary["judged"] = dsct_summary.n_stars >= MIN_STARS
    table = pd.concat([spots, dsct_summary], ignore_index=True)
    table.to_csv(out / "comparison.csv", index=False)
    dsct.to_csv(out / "dsct_stars.csv", index=False)
    j = table[table.judged]
    summary = {"bins_judged": int(len(j)), "agree_2sigma": int(j.agrees_2sigma.sum()),
               "median_abs_diff": float(j["diff"].abs().median()) if len(j) else None}
    (out / "comparison.json").write_text(json.dumps(json_safe(summary), indent=2))
    from .plots import plot_comparison
    plot_comparison(table, out / "plots" / "observed_vs_predicted.png")
    print(f"{summary['agree_2sigma']} of {summary['bins_judged']} judged bin-methods agree within 2 sigma; "
          f"wrote {out / 'comparison.csv'}")
    return out
