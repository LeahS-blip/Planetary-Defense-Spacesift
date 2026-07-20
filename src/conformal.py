#!/usr/bin/env python3
"""Split-conformal operating point with a NEO-retention guarantee.

A raw probability ("P=0.31") doesn't tell an observatory how much risk they take
by skipping an object. Conformal prediction converts the model into a *guarantee*:
choose a probability threshold from a held-out calibration set such that, with
finite-sample validity, at most alpha of true NEOs fall below it.

Method (Mondrian / class-conditional split conformal on the positive class):
  - split labeled data into train + calibration
  - fit the model on train, score the calibration NEOs
  - the alpha-quantile of calibration-NEO scores is the threshold t: any object
    with p >= t is "keep for follow-up". By exchangeability, the expected fraction
    of future real NEOs wrongly dropped is <= alpha (+ finite-sample slack).
  - report the junk-exclusion achieved at that guaranteed-recall threshold.

This is the honest version of "exclude X% of junk while losing <=1% of NEOs":
the <=1% is a calibrated guarantee, not a hopeful point estimate.
"""
import numpy as np


def conformal_threshold(cal_scores_pos, alpha=0.01):
    """alpha-quantile of positive-class calibration scores (finite-sample adjusted).

    Returns threshold t such that P(score < t | true NEO) <= alpha in expectation.
    Uses the conservative rank: floor((n+1)*alpha) order statistic.
    """
    s = np.sort(np.asarray(cal_scores_pos))
    n = len(s)
    if n == 0:
        return 0.0
    k = max(int(np.floor((n + 1) * alpha)) - 1, 0)  # 0-based index of the low quantile
    return float(s[k])


def evaluate_conformal(y, p, alpha=0.01, n_splits=40, seed=1):
    """Repeated split-conformal estimate of junk-exclusion at guaranteed NEO recall.

    For each random calibration split: derive the threshold from calibration NEOs,
    then measure on the held-out test fold what fraction of non-NEOs it excludes
    and what NEO recall it actually achieves. Averaged over splits.
    """
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(seed)
    excl, rec, thr = [], [], []
    idx = np.arange(len(y))
    for _ in range(n_splits):
        cal, test = train_test_split(idx, test_size=0.5, stratify=y, random_state=rng.randint(1e9))
        pos_cal = p[cal][y[cal] == 1]
        if len(pos_cal) < 5:
            continue
        t = conformal_threshold(pos_cal, alpha)
        yt, pt = y[test], p[test]
        keep = pt >= t
        neg = yt == 0
        pos = yt == 1
        if neg.sum():
            excl.append(float((neg & ~keep).sum() / neg.sum()))
        if pos.sum():
            rec.append(float((pos & keep).sum() / pos.sum()))
        thr.append(t)
    if not excl:
        return None
    return {
        "alpha": alpha,
        "target_neo_recall": 1 - alpha,
        "achieved_neo_recall_mean": float(np.mean(rec)),
        "junk_excluded_mean": float(np.mean(excl)),
        "junk_excluded_std": float(np.std(excl)),
        "threshold_mean": float(np.mean(thr)),
        "n_splits_used": len(excl),
    }


if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from train import load, augment, build_obs_freq
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    X, y, cols, disc, rows = load(root / "data" / "processed" / "labeled.csv", with_backfill=True)
    Xa, _ = augment(X, cols, rows, build_obs_freq(rows, np.arange(len(y))))
    mk = lambda: lgb.LGBMClassifier(boosting_type="dart", n_estimators=400, learning_rate=0.05,
        num_leaves=15, max_depth=4, min_child_samples=15, colsample_bytree=0.6,
        reg_lambda=3.0, class_weight="balanced", random_state=42, verbose=-1)
    # out-of-fold probabilities so the conformal eval isn't optimistic
    p = cross_val_predict(mk(), Xa, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                          method="predict_proba", n_jobs=4)[:, 1]
    for a in (0.01, 0.05, 0.10):
        r = evaluate_conformal(y, p, alpha=a)
        print(f"alpha={a}: guarantee keep {100*(1-a):.0f}% of NEOs -> "
              f"achieved recall {100*r['achieved_neo_recall_mean']:.1f}%, "
              f"excludes {100*r['junk_excluded_mean']:.0f}±{100*r['junk_excluded_std']:.0f}% of junk")
