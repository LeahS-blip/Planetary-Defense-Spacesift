#!/usr/bin/env python3
"""Motion-consistency baseline for the CSS 4-frame sequences — no deep learning.

Physics: a real asteroid moves at constant rate along a straight line across the
4 frames (~6 min apart); hot pixels / stars don't move; cosmic rays appear once.
Since frames are centered on the *candidate track*, the moving object should
appear near frame-specific positions along a line through the center.

Per frame we find the brightest compact blob near the center (after background
subtraction), then score:
  linearity_rms   residual (px) of the 4 blob positions from a fitted line
  step_cv         coefficient of variation of the 3 inter-frame displacements
  flux_cv         variation of blob peak flux across frames
  n_detected      frames with a significant blob (SNR > 3)
  max_step_px     largest displacement (0 for stationary artifacts)

Combined into P(real) with a tiny logistic model when labels are available
(train_motion_baseline), or a transparent heuristic score otherwise.
This is the number the CNN has to beat.
"""
import numpy as np
from pathlib import Path

def _blob(frame, search=100):
    """Brightest compact source near center: (row, col, snr, flux)."""
    h, w = frame.shape
    cy, cx = h // 2, w // 2
    win = frame[cy - search:cy + search, cx - search:cx + search].copy()
    med = np.median(win)
    mad = np.median(np.abs(win - med)) + 1e-6
    # light smoothing: 3x3 box (separable, pure numpy)
    k = np.ones(3) / 3.0
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, win)
    sm = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, sm)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    snr = (sm[iy, ix] - med) / (1.4826 * mad)
    return (iy - search, ix - search, float(snr), float(sm[iy, ix] - med))

def features(x):
    """x: (4, H, W) normalized stack -> dict of motion-consistency features."""
    pos, snrs, flux = [], [], []
    for f in x:
        r, c, snr, fl = _blob(f)
        pos.append((r, c)); snrs.append(snr); flux.append(fl)
    pos = np.array(pos, float); snrs = np.array(snrs); flux = np.array(flux)
    det = snrs > 3.0
    n_det = int(det.sum())
    t = np.arange(4.0)
    # fit linear motion row(t), col(t) on detected frames (need >=3)
    if n_det >= 3:
        A = np.vstack([t[det], np.ones(det.sum())]).T
        rr, _, _, _ = np.linalg.lstsq(A, pos[det, 0], rcond=None)
        cc, _, _, _ = np.linalg.lstsq(A, pos[det, 1], rcond=None)
        pred = np.stack([A @ rr, A @ cc], 1)
        lin_rms = float(np.sqrt(((pos[det] - pred) ** 2).sum(1).mean()))
        rate = float(np.hypot(rr[0], cc[0]))  # px/frame
    else:
        lin_rms, rate = np.nan, np.nan
    steps = np.sqrt(((pos[1:] - pos[:-1]) ** 2).sum(1))
    step_cv = float(steps.std() / (steps.mean() + 1e-6))
    flux_pos = flux[det] if n_det else flux
    flux_cv = float(flux_pos.std() / (abs(flux_pos.mean()) + 1e-6))
    return {"n_detected": n_det, "linearity_rms": lin_rms, "rate_px_frame": rate,
            "step_cv": step_cv, "flux_cv": flux_cv,
            "max_step_px": float(steps.max()), "min_snr": float(snrs.min()),
            "mean_snr": float(snrs.mean())}

def heuristic_score(f):
    """Transparent 0-1 score without training: moves linearly + consistently."""
    if f["n_detected"] < 3 or not np.isfinite(f["linearity_rms"]):
        return 0.15  # barely detected -> probably not a confirmable real object
    s = 1.0
    s *= np.exp(-max(f["linearity_rms"] - 2.0, 0) / 6.0)   # straight line
    s *= np.exp(-max(f["step_cv"] - 0.25, 0) / 0.5)        # constant rate
    s *= np.exp(-max(f["flux_cv"] - 0.6, 0) / 1.0)         # steady brightness
    if f["rate_px_frame"] < 1.0:
        s *= 0.3                                            # stationary => star/hot pixel
    return float(np.clip(s, 0.01, 0.99))

def run(css_dir, out_csv=None):
    import csv
    from css_dataset import load_all
    subs = load_all(css_dir, labeled_only=False)
    rows = []
    for s in subs:
        f = features(s["x"])
        f["p_real_heuristic"] = heuristic_score(f)
        f["subject_id"], f["y"], f["digest2"] = s["subject_id"], s["y"], s["digest2"]
        rows.append(f)
    lab = [r for r in rows if r["y"] is not None]
    if len(lab) >= 20 and len({r["y"] for r in lab}) == 2:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score([r["y"] for r in lab], [r["p_real_heuristic"] for r in lab])
        print(f"heuristic motion baseline AUC on {len(lab)} labeled subjects: {auc:.3f}")
    if out_csv and rows:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"{len(rows)} subjects -> {out_csv}")
    return rows

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    run(root / "data" / "css_zooniverse", root / "outputs" / "motion_baseline.csv")
