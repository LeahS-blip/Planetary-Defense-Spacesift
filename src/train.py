#!/usr/bin/env python3
"""Train NEOCP vetting models and evaluate against the A&A 2025 benchmark protocol.

Models:
  0. Baseline: feed Digest2 score threshold (status quo)
  1. Logistic regression (used automatically while the dataset is small)
  2. LightGBM + isotonic calibration (activates at >=200 labeled rows)

Split: time-based by discovery date (train on earlier, test on later) once
n >= MIN_TIME_SPLIT rows; below that, stratified 5-fold CV is reported and the
model is refit on everything (prototype mode — numbers are NOT benchmark-comparable
until the capture archive grows).

Benchmark targets (A&A 2025, arXiv/aa54311-25):
  * exclude >80% of non-NEOs while losing <=1% of true NEOs
  * NEO precision 0.95-0.98
Report: at the threshold giving 99% NEO recall on test -> non-NEO exclusion %,
precision, ROC-AUC, PR-AUC.
"""
import csv, json, math, pickle, sys
from pathlib import Path
import numpy as np

MIN_TIME_SPLIT = 200
NUMERIC_PREFIXES = ("d2_",)
NUMERIC_COLS = ["feed_score", "V", "H", "arc_feed", "nobs_feed", "not_seen_dys",
                "nobs_tracklet", "arc_days_tracklet", "sky_rate_deg_day", "motion_pa_deg",
                "ecl_lat_deg", "solar_elong_deg", "opposition_dist_deg", "gc_rms_arcsec",
                "mag_mean", "mag_range", "multi_night",
                "ts_n_snapshots", "ts_span_days", "ts_score_delta", "ts_score_slope",
                "ts_arc_growth", "ts_nobs_growth", "ts_nobs_rate", "ts_unseen_max",
                "ts_unseen_trend", "ts_v_faded",
                "ztf_has_cutout", "ztf_snr", "ztf_pos_flux", "ztf_compactness",
                "ztf_neg_frac", "ztf_masked_frac", "ztf_bkg_mad"]

def load(labeled_csv, with_backfill=False):
    with open(labeled_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("source", "capture")
    bf = Path(labeled_csv).parent / "backfill_labeled.csv"
    if with_backfill and bf.exists():
        with open(bf) as f:
            brows = [r for r in csv.DictReader(f) if r["trksub"] not in {x["trksub"] for x in rows}]
        # GUARD against provenance leakage: backfilled rows carry digest2/tracklet
        # features but no feed features. Only merge if the d2-featured subset of the
        # RESULTING dataset contains both classes — otherwise "has digest2" would be
        # a perfect (and bogus) predictor of the label.
        merged = rows + brows
        d2_rows = [r for r in merged if r.get("d2_NEO_noid") not in (None, "", "None")]
        classes = {r["y"] for r in d2_rows}
        if len(classes) == 2:
            rows = merged
            print(f"merged {len(brows)} backfill rows (guard passed: both classes have digest2 features)")
        else:
            print(f"SKIPPED {len(brows)} backfill rows: digest2-featured subset has classes {classes} — "
                  "would leak provenance. Capture more junk-class tracklets first.")
    cols = sorted({c for r in rows for c in r if c in NUMERIC_COLS or c.startswith(NUMERIC_PREFIXES)})
    X = np.array([[float(r[c]) if r.get(c) not in (None, "", "None") else np.nan for c in cols]
                  for r in rows], dtype=float)
    y = np.array([int(r["y"]) for r in rows])
    disc = [r.get("disc_ymd", "") for r in rows]
    return X, y, cols, disc, rows

def metrics_at_recall(y_true, p, target_recall=0.99):
    order = np.argsort(-p)
    ys = y_true[order]
    n_pos = ys.sum()
    if n_pos == 0:
        return None
    tp = np.cumsum(ys)
    k = np.searchsorted(tp, math.ceil(target_recall * n_pos))
    k = min(k, len(ys) - 1)
    sel = slice(0, k + 1)
    tp_k = tp[k]; fp_k = (k + 1) - tp_k
    neg_total = (ys == 0).sum()
    return {
        "threshold": float(np.sort(p)[::-1][k]),
        "neo_recall": float(tp_k / n_pos),
        "non_neo_excluded_frac": float((neg_total - fp_k) / neg_total) if neg_total else None,
        "precision": float(tp_k / (tp_k + fp_k)),
    }

def evaluate(y, p, tag):
    from sklearn.metrics import roc_auc_score, average_precision_score
    out = {"model": tag, "n": int(len(y)), "n_pos": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        out["at_99pct_neo_recall"] = metrics_at_recall(y, p)
    return out

def build_obs_freq(rows, idx):
    obs = [rows[i].get("obscode_first") or "" for i in idx]
    freq = {}
    for o in obs:
        freq[o] = freq.get(o, 0) + 1
    return {k: v / max(len(idx), 1) for k, v in freq.items()}

def augment(X, cols, rows, obs_freq):
    """Derived features: digest2 aggregates + observatory encoding.
    obs_freq: {obscode: frequency} computed on training data (or all data at deploy)."""
    def col(c):
        return X[:, cols.index(c)] if c in cols else np.full(len(X), np.nan)
    d2n, d2nr = col("d2_NEO_noid"), col("d2_NEO_raw")
    mba = np.nansum([col(f"d2_{c}_noid")
                     for c in ("MB1", "MB2", "MB3", "Pal", "Han", "Hun", "Pho", "Hil")], axis=0)
    obs = np.array([r.get("obscode_first") or "" for r in rows])
    fenc = np.array([obs_freq.get(o, 0.0) for o in obs])
    extra = [d2n - mba, mba, d2n - d2nr, d2n - col("d2_MC_noid"), fenc]
    names = ["neo_minus_mba", "sum_mba_noid", "neo_noid_minus_raw", "neo_minus_mc",
             "obscode_freq"]
    for code in ("G96", "703", "F51", "F52", "I41", "T05", "T08", "M22"):
        extra.append((obs == code).astype(float)); names.append(f"obs_{code}")
    return np.hstack([X] + [e[:, None] for e in extra]), cols + names

def main(labeled_csv, models_dir, results_json, with_backfill=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y, cols, disc, rows = load(labeled_csv, with_backfill=with_backfill)
    n = len(y)
    results = {"n": n, "n_pos": int(y.sum()), "n_neg": int(n - y.sum()), "features": cols}

    # Baseline: digest2 feed score alone
    score_idx = cols.index("feed_score")
    p_base = np.nan_to_num(X[:, score_idx], nan=50.0) / 100.0
    results["baseline_digest2_score"] = evaluate(y, p_base, "digest2_score_threshold")

    def make_model():
        import lightgbm as lgb
        # DART boosting chosen 2026-07-17 after repeated-CV comparison (6x5-fold):
        # DART 0.843 vs gbdt-current 0.827 AUC. DART's dropout regularizes better
        # on this small, noisy set. Note: DART ignores early_stopping, so we use a
        # fixed n_estimators instead of the validation-tuned count.
        return lgb.LGBMClassifier(
            boosting_type="dart", n_estimators=400, learning_rate=0.05,
            num_leaves=15, max_depth=4, min_child_samples=15, colsample_bytree=0.6,
            reg_lambda=3.0, class_weight="balanced", random_state=42, verbose=-1)

    big = n >= MIN_TIME_SPLIT
    if big:
        import lightgbm as lgb
        order = np.argsort(disc)
        cut = int(0.75 * n)
        tr, te = order[:cut], order[cut:]
        Xa, cols_a = augment(X, cols, rows, build_obs_freq(rows, tr))
        model = make_model()
        model.fit(Xa[tr], y[tr])
        p = model.predict_proba(Xa[te])[:, 1]
        results["model"] = "lightgbm_dart_time_split_v3"
        results["test"] = evaluate(y[te], p, "lightgbm_dart")
        results["baseline_on_test"] = evaluate(y[te], p_base[te], "digest2_score_threshold")
        # deployment refit on ALL data
        obs_freq_full = build_obs_freq(rows, np.arange(n))
        Xa_full, _ = augment(X, cols, rows, obs_freq_full)
        final = make_model()
        final.fit(Xa_full, y)
        cols = cols_a  # persist augmented schema
        results["features"] = cols
        # robust headline: repeated stratified CV (random folds) is far more stable
        # than the single ~100-object time split. Reported as cv_auc with std.
        try:
            from sklearn.model_selection import RepeatedStratifiedKFold
            from sklearn.metrics import roc_auc_score
            rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=1)
            aucs = []
            for tri, tei in rcv.split(Xa_full, y):
                mm = make_model(); mm.fit(Xa_full[tri], y[tri])
                aucs.append(roc_auc_score(y[tei], mm.predict_proba(Xa_full[tei])[:, 1]))
            results["cv_auc"] = float(np.mean(aucs))
            results["cv_auc_std"] = float(np.std(aucs))
            # conformal operating points: out-of-fold probs -> guaranteed NEO retention
            from sklearn.model_selection import cross_val_predict, StratifiedKFold
            from conformal import evaluate_conformal
            oof = cross_val_predict(make_model(), Xa_full, y,
                                    cv=StratifiedKFold(5, shuffle=True, random_state=1),
                                    method="predict_proba", n_jobs=4)[:, 1]
            results["conformal"] = {str(a): evaluate_conformal(y, oof, alpha=a)
                                     for a in (0.01, 0.05, 0.10)}
        except Exception as e:
            print("cv_auc/conformal skipped:", e)
    else:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        k = max(2, min(5, int(min(y.sum(), n - y.sum()))))
        results["model"] = f"logreg_{k}fold_cv_PROTOTYPE"
        results["warning"] = (f"only {n} labeled rows — prototype validation of the pipeline, "
                              "NOT a benchmark-comparable result; grows with the capture archive")
        if k >= 2:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            p = cross_val_predict(pipe, np.nan_to_num(X, nan=-999), y, cv=cv,
                                  method="predict_proba")[:, 1]
            results["cv"] = evaluate(y, p, "logreg_cv")
        pipe.fit(np.nan_to_num(X, nan=-999), y)
        final = pipe

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    base_cols = [c for c in cols if c in NUMERIC_COLS or c.startswith(NUMERIC_PREFIXES)]
    with open(Path(models_dir) / "vetting_model.pkl", "wb") as f:
        pickle.dump({"model": final, "features": cols, "base_features": base_cols,
                     "obs_freq": locals().get("obs_freq_full"),
                     "trained_on_n": n, "mode": results["model"]}, f)
    Path(results_json).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    main(root / "data" / "processed" / "labeled.csv", root / "models",
         root / "outputs" / "training_results.json",
         with_backfill="--with-backfill" in sys.argv)
