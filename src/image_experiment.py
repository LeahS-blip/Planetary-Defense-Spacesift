#!/usr/bin/env python3
"""Image-track experiment on cached CSS matrices, WITHOUT torch.

train_cnn.py needs PyTorch (unavailable in the Cowork sandbox: the CPU wheel host
is blocked and the PyPI wheel is 755 MB). This reproduces the experiment
train_cnn.py was built for -- "do the discovery-image pixels add signal over the
digest2 score?" -- using a gradient-boosted stand-in on the cached feature
matrices (data/processed/css_matrices.npz: 64x64 pixels + 8 motion features).

Protocol matches outputs/css_image_results.json: stratified 75/25 split. Models
on the SAME test split:
  1. digest2-only               (tabular status quo)
  2. motion-features-only GBM
  3. image-pixels-only GBM       (the CNN stand-in; raw downsampled pixels)
  4. fused: image score + motion score + digest2 + magnitude -> logistic
Reports ROC-AUC (+ AP for the image model), plus junk-excluded @95% real recall.
"""
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

def gbm(seed=42):
    return lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.06, num_leaves=15, max_depth=4,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.5,
        reg_lambda=2.0, class_weight="balanced", random_state=seed, verbose=-1)

def excl_at_recall(y, p, target=0.95):
    order = np.argsort(-p); ys = y[order]
    npos = ys.sum(); nneg = (ys == 0).sum()
    tp = np.cumsum(ys)
    k = np.searchsorted(tp, int(np.ceil(target * npos)))
    k = min(k, len(ys) - 1)
    fp = (k + 1) - tp[k]
    return float((nneg - fp) / nneg) if nneg else None

def main():
    root = Path(__file__).resolve().parent.parent
    d = np.load(root / "data" / "processed" / "css_matrices.npz")
    Ximg, Xmot = d["X_img"], d["X_mot"]
    y = d["y"].astype(int)
    d2 = np.nan_to_num(d["d2"].astype(float), nan=50.0)
    mag = np.nan_to_num(d["mag"].astype(float), nan=20.0)
    n = len(y)

    SEEDS = [1, 2, 3]
    acc = {"digest2_only": [], "motion_only": [], "image_pixels_only": [],
           "image_pixels_only_ap": [], "fused": [], "junk_excl95_fused": []}
    for seed in SEEDS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        tr, te = next(sss.split(Ximg, y))

        lr = LogisticRegression().fit(d2[tr].reshape(-1, 1), y[tr])
        acc["digest2_only"].append(roc_auc_score(y[te], lr.predict_proba(d2[te].reshape(-1, 1))[:, 1]))

        m = gbm(seed).fit(Xmot[tr], y[tr])
        p_mot_tr, p_mot_te = m.predict_proba(Xmot[tr])[:, 1], m.predict_proba(Xmot[te])[:, 1]
        acc["motion_only"].append(roc_auc_score(y[te], p_mot_te))

        im = gbm(seed).fit(Ximg[tr], y[tr])            # raw 64x64 downsampled pixels
        p_img_tr, p_img_te = im.predict_proba(Ximg[tr])[:, 1], im.predict_proba(Ximg[te])[:, 1]
        acc["image_pixels_only"].append(roc_auc_score(y[te], p_img_te))
        acc["image_pixels_only_ap"].append(average_precision_score(y[te], p_img_te))

        Ftr = np.column_stack([p_img_tr, p_mot_tr, d2[tr], mag[tr]])
        Fte = np.column_stack([p_img_te, p_mot_te, d2[te], mag[te]])
        fl = LogisticRegression(max_iter=1000).fit(Ftr, y[tr])
        p_fused = fl.predict_proba(Fte)[:, 1]
        acc["fused"].append(roc_auc_score(y[te], p_fused))
        acc["junk_excl95_fused"].append(excl_at_recall(y[te], p_fused, 0.95))

    print(f"{n} subjects: {int(y.sum())} real / {int(n - y.sum())} bogus; "
          f"{len(SEEDS)} stratified 75/25 splits, test n={len(te)} each")

    results = {"n": int(n), "n_test": int(len(te)),
               "protocol": f"mean over {len(SEEDS)} stratified 75/25 splits (AUC mean +/- std)",
               "note": "GBM stand-in for the CNN (torch unavailable in sandbox); "
                       "run src/train_cnn.py locally for the true CNN result."}
    for k, v in acc.items():
        results[f"{k}_auc" if not k.endswith(("_ap", "_fused")) else k] = \
            [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)]
    print(json.dumps(results, indent=2))
    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs" / "image_experiment_results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
