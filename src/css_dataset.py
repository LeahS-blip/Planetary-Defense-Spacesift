#!/usr/bin/env python3
"""Loader for the CSS/Zooniverse 4-frame detection dataset.

Each subject: 4 PNG frames (817x817, ~6 min apart), candidate at the green
circle in the center. We mask the green overlay (it is identical across classes
but a lazy CNN could still latch onto compression artifacts around it), convert
to grayscale float32, and center-crop.

IMPORTANT LEAKAGE NOTE: verify on the full pull that the green circle geometry
does not differ between real/bogus subjects (it shouldn't — it marks the
candidate position in every frame). The masking here makes that moot.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

def load_frame(path, crop=256):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=np.float32)
    # green-overlay mask: pixels where G strongly dominates R and B
    g_dom = (a[..., 1] > 1.4 * a[..., 0] + 20) & (a[..., 1] > 1.4 * a[..., 2] + 20)
    gray = a.mean(axis=2)
    med = np.median(gray)
    gray[g_dom] = med
    h, w = gray.shape
    c = crop // 2
    return gray[h // 2 - c:h // 2 + c, w // 2 - c:w // 2 + c]

def load_subject(sdir, crop=256):
    sdir = Path(sdir)
    meta_fp = sdir / "subject.json"
    if not meta_fp.exists():
        return None  # incomplete subject (e.g. interrupted fetch / OneDrive placeholder)
    meta = json.loads(meta_fp.read_text())
    frames = []
    for i in range(1, 5):
        fp = sdir / f"frame_{i}.png"
        if fp.exists():
            frames.append(load_frame(fp, crop))
    if len(frames) < 4:
        return None
    x = np.stack(frames)                      # (4, crop, crop)
    mu, sd = x.mean(), x.std() + 1e-6
    x = (x - mu) / sd
    lab = meta.get("label", {})
    y = lab.get("is_real_asteroid")
    if y is None:
        # fall back to CSS's own vetting verdict: 'normal' = accepted real detection,
        # 'rjct' = rejected as artifact. This IS the operational real/bogus label.
        kw = (meta.get("metadata") or {}).get("keyword")
        if kw == "normal":
            y = True
        elif kw == "rjct":
            y = False
    return {"x": x.astype(np.float32),
            "y": None if y is None else int(bool(y)),
            "subject_id": meta.get("subject_id"),
            "digest2": (meta.get("metadata") or {}).get("digest2_score"),
            "magnitude": (meta.get("metadata") or {}).get("magnitude"),
            "label_source": lab.get("label_source")}

def load_all(css_dir, crop=256, labeled_only=True):
    import sys, time
    dirs = sorted(Path(css_dir).glob("subject_*"))
    total = len(dirs)
    out = []
    t0 = time.time()
    for i, sdir in enumerate(dirs, 1):
        s = load_subject(sdir, crop)
        if i % 250 == 0 or i == total:
            print(f"  loading images: {i}/{total} dirs  "
                  f"({len(out)} labeled kept, {time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
        if s is None:
            continue
        if labeled_only and s["y"] is None:
            continue
        out.append(s)
    return out

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    subs = load_all(root / "data" / "css_zooniverse", labeled_only=False)
    n_lab = sum(1 for s in subs if s["y"] is not None)
    print(f"{len(subs)} subjects loadable, {n_lab} labeled")
    for s in subs[:5]:
        print(f'  {s["subject_id"]}: y={s["y"]} digest2={s["digest2"]} shape={s["x"].shape}')
