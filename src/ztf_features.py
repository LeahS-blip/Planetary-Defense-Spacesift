#!/usr/bin/env python3
"""Features from a ZTF difference-image cutout, keyed by NEOCP trksub.

This is the piece that makes real image+tabular FUSION possible: unlike the CSS
Zooniverse images (a different object population), these cutouts are of the
*same NEOCP objects* the tabular model scores, so ztf_* columns join per-object.

A ZTF difference image (science minus reference) shows a real moving asteroid as
a compact POSITIVE residual against a flat ~zero background; artifacts show as
dipoles, streaks, or masked junk. Features quantify that:
  ztf_has_cutout     1 if a usable cutout exists
  ztf_snr            peak SNR of the brightest source near center (vs robust noise)
  ztf_pos_flux       summed positive flux in the central aperture (DN)
  ztf_compactness    central-pixel flux / 5x5-box flux (point source -> ~high)
  ztf_neg_frac       fraction of central region that is negative (subtraction junk)
  ztf_masked_frac    fraction of NaN/degenerate pixels (bad cutout region)
  ztf_bkg_mad        robust background scatter (image quality proxy)

Needs astropy + numpy. Reads HDU[1] (ZTF cutouts put data there, not HDU[0]).
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ZTF_COLS = ["ztf_has_cutout", "ztf_snr", "ztf_pos_flux", "ztf_compactness",
            "ztf_neg_frac", "ztf_masked_frac", "ztf_bkg_mad"]

def features_from_fits(path):
    from astropy.io import fits
    try:
        with fits.open(path) as h:
            data = next((hd.data for hd in h if hd.data is not None), None)
    except Exception:
        return {"ztf_has_cutout": 0}
    if data is None or data.ndim != 2:
        return {"ztf_has_cutout": 0}
    a = np.asarray(data, dtype=float)
    masked = ~np.isfinite(a)
    # degenerate cutout (constant / off-detector): std near zero
    finite = a[np.isfinite(a)]
    if finite.size < 100 or np.ptp(finite) < 1e-3:
        return {"ztf_has_cutout": 0, "ztf_masked_frac": float(masked.mean())}
    h, w = a.shape
    cy, cx = h // 2, w // 2
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med)) * 1.4826 + 1e-6
    # central aperture (radius ~5 px, ZTF ~1"/px so ~5")
    yy, xx = np.ogrid[:h, :w]
    rc = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ap = (rc <= 5) & np.isfinite(a)
    core = (rc <= 1.5) & np.isfinite(a)
    box = (rc <= 2.5) & np.isfinite(a)
    peak = np.nanmax(a[ap]) if ap.any() else med
    pos_flux = float(np.nansum((a[ap] - med).clip(min=0))) if ap.any() else 0.0
    box_flux = float(np.nansum((a[box] - med).clip(min=0))) + 1e-6
    core_flux = float(np.nansum((a[core] - med).clip(min=0)))
    return {
        "ztf_has_cutout": 1,
        "ztf_snr": float((peak - med) / mad),
        "ztf_pos_flux": pos_flux,
        "ztf_compactness": float(core_flux / box_flux),
        "ztf_neg_frac": float(np.nanmean(a[ap] < med)) if ap.any() else 0.0,
        "ztf_masked_frac": float(masked.mean()),
        "ztf_bkg_mad": float(mad),
    }

def build(cutout_dir=None):
    """Return {trksub: ztf_features} for every <trksub>.fits in the cutout dir."""
    cutout_dir = Path(cutout_dir or (ROOT / "data" / "capture" / "ztf_cutouts"))
    out = {}
    if not cutout_dir.exists():
        return out
    for f in cutout_dir.glob("*.fits"):
        out[f.stem] = features_from_fits(f)
    return out

if __name__ == "__main__":
    # self-test on the validated sample cutouts committed in the repo root
    for s in ("ztf_2019aq3_diff.fits", "ztf_diff_cutout.fits"):
        p = ROOT / s
        if p.exists():
            print(f"{s}: {features_from_fits(p)}")
