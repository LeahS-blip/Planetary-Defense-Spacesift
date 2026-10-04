"""Detrending wrappers. The method is an experiment variable, so it is named by a short spec string."""

from __future__ import annotations

import numpy as np


def flatten(time: np.ndarray, flux: np.ndarray, spec: str) -> np.ndarray:
    """Return detrended flux (trend divided out). Spec examples: 'biweight-1.0', 'none'."""
    if spec == "none":
        return flux / np.nanmedian(flux)
    method, _, window = spec.partition("-")
    from wotan import flatten as wotan_flatten

    flat = wotan_flatten(time, flux, method=method, window_length=float(window), edge_cutoff=0.0,
                         break_tolerance=0.5)
    return np.asarray(flat, dtype=float)


def clean(time: np.ndarray, flux: np.ndarray, sigma_upper: float = 4.0):
    """Drop NaNs left by detrending and clip upward outliers only (transits go down)."""
    good = np.isfinite(flux)
    t, f = time[good], flux[good]
    med = np.median(f)
    mad = 1.4826 * np.median(np.abs(f - med))
    keep = f < med + sigma_upper * mad
    return t[keep], f[keep]
