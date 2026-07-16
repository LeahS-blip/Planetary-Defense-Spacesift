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
import argparse, json
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
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    print(f"{n} labeled subjects: {int(y.sum())} real / {int(n - y.sum())} bogus; "
          f"train {len(tr)} / test {len(te)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.crop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor([(len(tr) - y[tr].sum()) / max(y[tr].sum(), 1)]).to(dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.tensor(X); yt = torch.tensor(y)
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
            loss.backward(); opt.step(); tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            p_te = torch.sigmoid(model(Xt[te].to(dev)).squeeze(-1)).cpu().numpy()
        print(f"epoch {ep+1}: loss {tot/len(tr):.4f}  test AUC {roc_auc_score(y[te], p_te):.3f}")

    with torch.no_grad():
        logit_tr = model(Xt[tr].to(dev)).squeeze(-1).cpu().numpy()
        logit_te = model(Xt[te].to(dev)).squeeze(-1).cpu().numpy()
    results = {"n": n, "n_test": len(te)}
    # 1. digest2 only
    lr = LogisticRegression().fit(d2[tr].reshape(-1, 1), y[tr])
    p = lr.predict_proba(d2[te].reshape(-1, 1))[:, 1]
    results["digest2_only_auc"] = float(roc_auc_score(y[te], p))
    # 2. CNN only
    p_cnn = 1 / (1 + np.exp(-logit_te))
    results["cnn_only_auc"] = float(roc_auc_score(y[te], p_cnn))
    results["cnn_only_ap"] = float(average_precision_score(y[te], p_cnn))
    # 3. fused
    Ftr = np.column_stack([logit_tr, d2[tr], mag[tr]])
    Fte = np.column_stack([logit_te, d2[te], mag[te]])
    lr2 = LogisticRegression().fit(Ftr, y[tr])
    results["fused_auc"] = float(roc_auc_score(y[te], lr2.predict_proba(Fte)[:, 1]))
    print(json.dumps(results, indent=2))
    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs" / "cnn_results.json").write_text(json.dumps(results, indent=2))
    torch.save(model.state_dict(), root / "models" / "css_cnn.pt")
    print("saved models/css_cnn.pt")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
