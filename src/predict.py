#!/usr/bin/env python3
"""Score the current NEOCP list -> outputs/daily_vetting.csv

Ranked by predicted P(real NEO); the feed's Digest2 score shown alongside for
comparison. This is the tangible artifact for the follow-up community.
Falls back to feed-score ranking if no trained model exists yet.
"""
import csv, pickle, sys
from pathlib import Path
import numpy as np
from features import build_feature_table
from followup import classify

def main(capture_dir, digest2_dir, model_pkl, out_csv):
    rows = build_feature_table(capture_dir, digest2_dir)
    model = feats = None
    if Path(model_pkl).exists():
        with open(model_pkl, "rb") as f:
            bundle = pickle.load(f)
        model, feats, mode = bundle["model"], bundle["features"], bundle.get("mode", "?")
        print(f"model: {mode} (trained on n={bundle.get('trained_on_n')})")
    Xall = None
    if model is not None and bundle.get("obs_freq") is not None:
        import numpy as _np
        from train import augment
        base = bundle["base_features"]
        Xb = _np.array([[float(r[c]) if r.get(c) not in (None, "", "None") else _np.nan
                         for c in base] for r in rows], dtype=float)
        Xall, _ = augment(Xb, base, rows, bundle["obs_freq"])
    out = []
    for i, r in enumerate(rows):
        rec = {"trksub": r["trksub"], "digest2_feed_score": r.get("feed_score"),
               "V": r.get("V"), "H": r.get("H"), "nobs": r.get("nobs_feed"),
               "arc_days": r.get("arc_feed"), "not_seen_dys": r.get("not_seen_dys")}
        rec["followup_status"] = classify(rec["nobs"], rec["arc_days"], rec["not_seen_dys"])
        if model is not None and Xall is not None:
            rec["p_real_neo"] = float(model.predict_proba(Xall[i:i+1])[0, 1])
        elif model is not None:
            x = np.array([[float(r[c]) if r.get(c) not in (None, "", "None") else np.nan
                           for c in feats]])
            try:
                rec["p_real_neo"] = float(model.predict_proba(x)[0, 1])
            except Exception:
                rec["p_real_neo"] = float(model.predict_proba(np.nan_to_num(x, nan=-999))[0, 1])
        else:
            rec["p_real_neo"] = (r.get("feed_score") or 0) / 100.0
        out.append(rec)
    out.sort(key=lambda r: -(r["p_real_neo"] or 0))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    tally = {}
    for r in out:
        tally[r["followup_status"]] = tally.get(r["followup_status"], 0) + 1
    print(f"vetting list: {len(out)} objects -> {out_csv}")
    print("  follow-up: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    for r in out[:10]:
        print(f'  {r["trksub"]:10s} p={r["p_real_neo"]:.3f}  digest2={r["digest2_feed_score"]}')

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    d2dir = sys.argv[1] if len(sys.argv) > 1 else None
    main(root / "data" / "capture", d2dir, root / "models" / "vetting_model.pkl",
         root / "outputs" / "daily_vetting.csv")
