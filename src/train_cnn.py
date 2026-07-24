#!/usr/bin/env python3
"""Small CNN on CSS 4-frame stacks: real asteroid vs artifact.

Design choices (deliberately simple first):
  * 4 frames stacked as input channels — temporal motion becomes a cross-channel
    pattern, which convs pick up easily at this scale.
  * ~300k params; trains on CPU for a few thousand subjects in minutes/epoch at 128px.
  * Compare three configurations on the SAME test split:
      1. digest2-score-only logistic (tabular status quo)
      2. CNN (image-only)
      3. fused: CNN logit + digest2 + magnitude -> logistic
    That comparison ("do pixels add anything over Digest2?") is the experiment.

Usage: python3 src/train_cnn.py [--crop 128] [--epochs 12] [--limit N]
Needs: torch, numpy, PIL. Run after fetch_css_bulk.py has populated
data/css_zooniverse (needs a few hundred labeled subjects minimum).
"""
import argparse, copy, json
from pathlib import Path
import numpy as np

def get_model(crop):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

def main():
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.linear_model import LogisticRegression
    from css_dataset import load_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs with no val-AUC gain (0 = off)")
    ap.add_argument("--balance", action="store_true",
                    help="downsample the majority class to equal the minority (recommended "
                         "on the imbalanced bulk feed)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    subs = load_all(root / "data" / "css_zooniverse", crop=args.crop, labeled_only=True)
    if args.limit:
        subs = subs[:args.limit]
    n = len(subs)
    if n < 60:
        print(f"only {n} labeled subjects — run fetch_css_bulk.py first (need a few hundred). "
              "Doing a forward-pass smoke test only.")
        if n:
            model = get_model(args.crop)
            x = torch.tensor(np.stack([s["x"] for s in subs]))
            with torch.no_grad():
                out = torch.sigmoid(model(x)).squeeze(-1)
            print("smoke test OK — model outputs:", [f"{v:.3f}" for v in out.tolist()])
        return

    X = np.stack([s["x"] for s in subs])
    y = np.array([s["y"] for s in subs], dtype=np.float32)
    d2 = np.array([float(s["digest2"] or 50) for s in subs], dtype=np.float32)
    mag = np.array([float(s["magnitude"] or 20) for s in subs], dtype=np.float32)

    # Optional class rebalancing: the CSS bulk feed is ~88% 'normal' (real) and adds
    # almost no new 'rjct' (bogus). That imbalance collapses discrimination (the net
    # just predicts "real"). --balance downsamples the majority class to match the
    # minority, so the model must actually separate the hard bogus cases.
    if args.balance:
        rb = np.random.RandomState(0)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        k = min(len(pos), len(neg))
        keep = np.sort(np.concatenate([rb.choice(pos, k, replace=False),
                                       rb.choice(neg, k, replace=False)]))
        X, y, d2, mag = X[keep], y[keep], d2[keep], mag[keep]
        print(f"--balance: downsampled to {len(y)} subjects "
              f"({int(y.sum())} real / {int(len(y) - y.sum())} bogus)")
    n = len(y)

    # Stratified train/val/test = 65/15/20. Best epoch is chosen on VAL (never on
    # test), so the reported test AUC stays an honest held-out estimate.
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(42)
    trv, te = train_test_split(np.arange(n), test_size=0.20, random_state=42, stratify=y)
    tr, val = train_test_split(trv, test_size=0.1875, random_state=42, stratify=y[trv])
    print(f"{n} labeled subjects: {int(y.sum())} real / {int(n - y.sum())} bogus; "
          f"train {len(tr)} / val {len(val)} / test {len(te)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.crop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor([(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)]).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.tensor(X); yt = torch.tensor(y)

    def auc_on(sel):
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xt[sel].to(dev)).squeeze(-1)).cpu().numpy()
        return roc_auc_score(y[sel], p)

    best = {"val_auc": -1.0, "epoch": -1, "state": None, "test_auc": None}
    patience, since_improved = max(args.patience, 0), 0
    for ep in range(args.epochs):
        model.train(); perm = rng.permutation(tr); tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            if rng.rand() < 0.5:  # augmentation: flips (motion direction is arbitrary)
                xb = torch.flip(xb, dims=[3])
            if rng.rand() < 0.5:
                xb = torch.flip(xb, dims=[2])
            opt.zero_grad()
            out = model(xb).squeeze(-1)
            loss = lossf(out, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(b)
        val_auc, test_auc = auc_on(val), auc_on(te)
        flag = ""
        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "epoch": ep + 1, "test_auc": test_auc,
                    "state": copy.deepcopy(model.state_dict())}
            since_improved = 0; flag = "  <- best"
        else:
            since_improved += 1
        print(f"epoch {ep+1}: loss {tot/len(tr):.4f}  val AUC {val_auc:.3f}  "
              f"test AUC {test_auc:.3f}{flag}")
        if patience and since_improved >= patience:
            print(f"early stop: no val improvement in {patience} epochs"); break

    # restore the best-validation checkpoint before reporting / saving
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    print(f"best epoch {best['epoch']}: val AUC {best['val_auc']:.3f}, "
          f"test AUC {best['test_auc']:.3f} (this checkpoint is saved)")

    with torch.no_grad():
        logit_tr = model(Xt[tr].to(dev)).squeeze(-1).cpu().numpy()
        logit_te = model(Xt[te].to(dev)).squeeze(-1).cpu().numpy()
    results = {"n": n, "n_test": len(te), "n_val": len(val),
               "checkpoint": "best val-AUC epoch (early-stopping); test never used for selection",
               "best_epoch": best["epoch"], "best_val_auc": float(best["val_auc"])}
    # 1. digest2 only
    lr = LogisticRegression().fit(d2[tr].reshape(-1, 1), y[tr])
    p = lr.predict_proba(d2[te].reshape(-1, 1))[:, 1]
    results["digest2_only_auc"] = float(roc_auc_score(y[te], p))
    # 2. CNN only
    p_cnn = 1 / (1 + np.exp(-logit_te))
    results["cnn_only_auc"] = float(roc_auc_score(y[te], p_cnn))
    results["cnn_only_ap"] = float(average_precision_score(y[te], p_cnn))
    # 3. fused: CNN + digest2 + magnitude (digest2 is ~chance on this subset)
    Ftr = np.column_stack([logit_tr, d2[tr], mag[tr]])
    Fte = np.column_stack([logit_te, d2[te], mag[te]])
    lr2 = LogisticRegression().fit(Ftr, y[tr])
    results["fused_auc"] = float(roc_auc_score(y[te], lr2.predict_proba(Fte)[:, 1]))

    # 4. fused + motion features: the CNN sees pixel morphology; the 8 cached motion
    #    features (data/processed/css_matrices.npz X_mot) see track kinematics -- a
    #    different, complementary signal that scores ~0.72 alone. GUARD: only trust the
    #    cached matrix if its label vector matches the loaded subjects row-for-row
    #    (both come from sorted labeled subject_* dirs, but verify -- silent misalignment
    #    would fabricate a bogus score).
    mot_npz = root / "data" / "processed" / "css_matrices.npz"
    if mot_npz.exists():
        dd = np.load(mot_npz)
        if dd["y"].shape[0] == n and np.array_equal(dd["y"].astype(int), y.astype(int)):
            Xm = dd["X_mot"]
            # Gradient-boosted fusion: motion signal is nonlinear (trajectory
            # thresholds), so a linear head throws most of it away. HistGBM also
            # ingests the NaN rows (subjects with too few detections) natively.
            from sklearn.ensemble import HistGradientBoostingClassifier
            Ftr2 = np.column_stack([logit_tr, Xm[tr], d2[tr], mag[tr]])
            Fte2 = np.column_stack([logit_te, Xm[te], d2[te], mag[te]])
            gb = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                class_weight="balanced", random_state=42).fit(Ftr2, y[tr])
            results["fused_with_motion_auc"] = float(
                roc_auc_score(y[te], gb.predict_proba(Fte2)[:, 1]))
        else:
            results["fused_with_motion_auc"] = None
            print("motion fusion SKIPPED: cached matrix does not align with loaded "
                  "subjects (rebuild css_matrices.npz).")
    print(json.dumps(results, indent=2))
    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs" / "cnn_results.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), root / "models" / "css_cnn.pt")
    print("saved models/css_cnn.pt")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
