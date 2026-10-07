"""Detrending wrappers. The method is an experiment variable, so it is named by a short spec string."""

from __future__ import annotations

import numpy as np


def prewhiten(time: np.ndarray, flux: np.ndarray, max_terms: int = 10, min_period: float = 1 / 24,
              max_period: float = 5.0, snr_stop: float = 5.0) -> np.ndarray:
    """Remove the strongest coherent sinusoids, one at a time, as is done for pulsating stars.

    Each step finds the highest Lomb-Scargle peak between min_period and max_period
    (days), fits a sinusoid at that frequency by least squares, and subtracts it. It
    stops after max_terms, or when the peak amplitude falls below snr_stop times the
    median amplitude of the spectrum (the noise level). Returns flux with those removed.
    """
    from astropy.timeseries import LombScargle

    resid = flux / np.nanmedian(flux) - 1.0
    span = time.max() - time.min()
    freq = np.arange(1 / max_period, 1 / min_period, 1 / (5 * span))
    step = freq[1] - freq[0]
    for _ in range(max_terms):
        ls = LombScargle(time, resid, normalization="psd")
        amp = np.sqrt(4 * ls.power(freq, method="fast") / time.size)  # psd -> sinusoid amplitude
        i = int(np.argmax(amp))
        if amp[i] < snr_stop * np.median(amp):
            break
        # Refine the peak on a 50x finer grid: a frequency error of one coarse step
        # would leave a beat of up to ~30% of the amplitude over the baseline.
        fine = np.linspace(freq[i] - step, freq[i] + step, 101)
        f_best = fine[int(np.argmax(ls.power(fine)))]
        w = 2 * np.pi * f_best * time
        design = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
        coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
        resid = resid - design[:, :2] @ coef[:2]
    return resid + 1.0


def flatten(time: np.ndarray, flux: np.ndarray, spec: str) -> np.ndarray:
    """Return detrended flux (trend divided out).

    Spec examples: 'none', 'biweight-1.0', 'prewhiten+biweight-1.0' (sinusoids removed
    first, then the biweight).
    """
    if spec.startswith("prewhiten+"):
        return flatten(time, prewhiten(time, flux), spec.removeprefix("prewhiten+"))
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
