#!/usr/bin/env python3
"""Held-out evaluation of the NEOCP vetting model on a time split.

Everything here is decided on the TRAINING period only and then applied once to the
later TEST period, so the test numbers are what a deployed model would have done:

  * operating points: the score thresholds that keep 95% / 99% of training NEOs
    (split-conformal threshold on out-of-fold training scores), for the model and
    for the digest2 baseline alike;
  * confusion matrices at those thresholds on the test period (true/false
    positives/negatives), with NEO recall, junk excluded and precision;
  * calibration: Brier score and a reliability table (does "0.8" mean 80%?);
  * bootstrap 95% intervals for test ROC-AUC, and for the model-minus-baseline gap.

What the model predicts: P(a NEOCP candidate turns out to be a real near-Earth
object), i.e. not junk and not an ordinary main-belt asteroid. It is not an
impact probability.
"""
import numpy as np


def _confusion(y, p, t):
    keep = p >= t
    tp = int(((y == 1) & keep).sum()); fn = int(((y == 1) & ~keep).sum())
    fp = int(((y == 0) & keep).sum()); tn = int(((y == 0) & ~keep).sum())
    return {
        "threshold": float(t), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "neo_recall": tp / (tp + fn) if tp + fn else None,
        "junk_excluded": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def _reliability(y, p, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum():
            out.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                        "mean_predicted": float(p[m].mean()), "fraction_neo": float(y[m].mean())})
    return out


def _bootstrap_auc(y, p_model, p_base, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    am, ab, d = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a1, a2 = roc_auc_score(y[i], p_model[i]), roc_auc_score(y[i], p_base[i])
        am.append(a1); ab.append(a2); d.append(a1 - a2)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"model": ci(am), "baseline": ci(ab), "difference": ci(d),
            "p_model_not_better": float(np.mean(np.array(d) <= 0))}


def holdout_report(make_model, X, y, disc, tr, te, p_base, recalls=(0.95, 0.99), seed=1):
    """make_model: () -> fresh sklearn-style classifier. X: features (already augmented with
    training-only statistics). disc: discovery dates (sortable strings). tr/te: index arrays.
    p_base: baseline scores in [0, 1] for every row."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from conformal import conformal_threshold

    ytr, yte = y[tr], y[te]
    # Out-of-fold scores on the training period, used only to choose thresholds.
    oof = cross_val_predict(make_model(), X[tr], ytr, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                            method="predict_proba")[:, 1]
    model = make_model().fit(X[tr], ytr)
    p_te = model.predict_proba(X[te])[:, 1]
    b_tr, b_te = p_base[tr], p_base[te]

    ops = {}
    for r in recalls:
        a = round(1 - r, 4)
        t_m = conformal_threshold(oof[ytr == 1], a)
        t_b = conformal_threshold(b_tr[ytr == 1], a)
        ops[f"{int(r * 100)}pct_recall"] = {"model": _confusion(yte, p_te, t_m), "baseline": _confusion(yte, b_te, t_b)}
    ops["p_0.5"] = {"model": _confusion(yte, p_te, 0.5), "baseline": _confusion(yte, b_te, 0.5)}

    two = 0 < yte.sum() < len(yte)
    return {
        "what_is_predicted": "probability that a NEOCP candidate is a real near-Earth object (not an impact probability)",
        "split": {"train_n": int(len(tr)), "test_n": int(len(te)), "test_neo": int(yte.sum()),
                  "train_dates": [min(disc[i] for i in tr), max(disc[i] for i in tr)],
                  "test_dates": [min(disc[i] for i in te), max(disc[i] for i in te)]},
        "roc_auc": {"model": float(roc_auc_score(yte, p_te)) if two else None,
                    "baseline": float(roc_auc_score(yte, b_te)) if two else None},
        "pr_auc": {"model": float(average_precision_score(yte, p_te)) if two else None,
                   "baseline": float(average_precision_score(yte, b_te)) if two else None},
        "auc_bootstrap_95ci": _bootstrap_auc(yte, p_te, b_te) if two else None,
        "operating_points": ops,
        "calibration": {"brier_model": float(brier_score_loss(yte, p_te)),
                        "brier_baseline": float(brier_score_loss(yte, np.clip(b_te, 0, 1))),
                        "brier_always_base_rate": float(brier_score_loss(yte, np.full(len(yte), ytr.mean()))),
                        "reliability_model": _reliability(yte, p_te)},
    }
