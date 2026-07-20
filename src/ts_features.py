#!/usr/bin/env python3
"""Time-series features from successive NEOCP snapshots.

The single-snapshot feature table treats each object as a still photo. But the
capture archives the SAME object across many snapshots, so we can measure how it
*evolves* while on the page — a signal the A&A 2025 benchmark never used because
they had single discovery tracklets, not a polling time-series.

Intuition: a real NEO getting followed up shows its arc lengthen, obs pile up,
and positional staleness reset as observers re-detect it; digest2 score often
firms up. A bogus/uninteresting object stalls — arc frozen, no new obs, staleness
climbing until removal. These trajectories are computed per Temp_Desig here.

Output columns (merged into features by features.py):
  ts_n_snapshots      how many snapshots we caught it in
  ts_span_days        wall-clock days between first and last snapshot
  ts_score_delta      change in digest2 feed score (first -> last)
  ts_score_slope      score change per day
  ts_arc_growth       arc-length increase (days) — follow-up accumulating
  ts_nobs_growth      new observations added while watched
  ts_nobs_rate        new obs per day
  ts_unseen_max       worst staleness seen (loss-risk proxy)
  ts_unseen_trend     staleness change first -> last (rising = being lost)
  ts_v_faded          apparent-mag change (positive = fading = harder to recover)
"""
import json, glob
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _stamp_to_dt(s):
    return datetime.strptime(s[:13], "%Y%m%dT%H%M")

def build(capture_dir=None):
    capture_dir = Path(capture_dir or (ROOT / "data" / "capture"))
    hist = defaultdict(list)
    for f in sorted((capture_dir / "snapshots").glob("neocp_*.json")):
        stamp = f.name.split("neocp_")[1].replace(".json", "")
        try:
            dt = _stamp_to_dt(stamp)
        except ValueError:
            continue
        for r in json.loads(f.read_text()):
            hist[r["Temp_Desig"]].append((dt, r))
    out = {}
    for desig, seq in hist.items():
        seq.sort(key=lambda x: x[0])
        dts = [s[0] for s in seq]
        recs = [s[1] for s in seq]
        span = (dts[-1] - dts[0]).total_seconds() / 86400.0
        def g(rec, k):
            try:
                return float(rec.get(k))
            except (TypeError, ValueError):
                return None
        def first_last(k):
            vals = [(d, g(r, k)) for d, r in zip(dts, recs) if g(r, k) is not None]
            return (vals[0], vals[-1]) if vals else (None, None)
        row = {"ts_n_snapshots": len(seq), "ts_span_days": round(span, 3)}
        (f0, l0) = first_last("Score")
        if f0 and l0:
            row["ts_score_delta"] = l0[1] - f0[1]
            row["ts_score_slope"] = (l0[1] - f0[1]) / span if span > 0.01 else 0.0
        (fa, la) = first_last("Arc")
        if fa and la:
            row["ts_arc_growth"] = la[1] - fa[1]
        (fn, ln) = first_last("NObs")
        if fn and ln:
            row["ts_nobs_growth"] = ln[1] - fn[1]
            row["ts_nobs_rate"] = (ln[1] - fn[1]) / span if span > 0.01 else 0.0
        uns = [g(r, "Not_Seen_dys") for r in recs if g(r, "Not_Seen_dys") is not None]
        if uns:
            row["ts_unseen_max"] = max(uns)
            row["ts_unseen_trend"] = uns[-1] - uns[0]
        (fv, lv) = first_last("V")
        if fv and lv:
            row["ts_v_faded"] = lv[1] - fv[1]
        out[desig] = row
    return out

TS_COLS = ["ts_n_snapshots", "ts_span_days", "ts_score_delta", "ts_score_slope",
           "ts_arc_growth", "ts_nobs_growth", "ts_nobs_rate", "ts_unseen_max",
           "ts_unseen_trend", "ts_v_faded"]

if __name__ == "__main__":
    ts = build()
    n_multi = sum(1 for v in ts.values() if v["ts_n_snapshots"] > 1)
    print(f"time-series features: {len(ts)} objects, {n_multi} with >1 snapshot")
    import statistics
    growths = [v.get("ts_nobs_growth", 0) for v in ts.values() if v["ts_n_snapshots"] > 1]
    if growths:
        print(f"  median obs added while watched: {statistics.median(growths):.0f}")
