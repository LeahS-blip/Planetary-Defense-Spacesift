"""Transit search methods. Every search returns a Candidate, so pipelines are interchangeable.

Interface: search(time, flux, cfg) -> Candidate
Vetters (CNN, gradient boosting; SS-0003) will take a Candidate and return a score.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float
    power: float
    sde: float  # signal detection efficiency: (peak - mean) / std of the periodogram

    def to_dict(self) -> dict:
        return {f"found_{k}": v for k, v in asdict(self).items()}


def bin_lightcurve(time: np.ndarray, flux: np.ndarray, width: float):
    """Mean-bin to `width` days; empty bins dropped. width <= 0 returns the input."""
    if width <= 0:
        return time, flux
    idx = np.floor((time - time.min()) / width).astype(int)
    counts = np.bincount(idx)
    ok = counts > 0
    t = np.bincount(idx, weights=time)[ok] / counts[ok]
    f = np.bincount(idx, weights=flux)[ok] / counts[ok]
    return t, f


def shortest_duration(period: np.ndarray | float, min_duration: float) -> np.ndarray:
    """Shortest plausible T14 at a period: a Sun-density star at b = 0.9 (0.436 of the
    central duration, which is ~13 h at 1 yr and scales as P^(1/3)), floored at
    the shortest searched duration. Denser stars (M dwarfs) need a smaller factor."""
    central = 13 / 24 * (np.asarray(period) / 365.25) ** (1 / 3)
    return np.maximum(min_duration, 0.436 * central)


def period_grid(baseline: float, pmin: float, pmax: float, min_duration: float, oversample: float) -> np.ndarray:
    """Frequency grid whose step keeps the phase drift over the baseline below
    duration / oversample, using the shortest plausible duration at each period.
    Longer periods have longer transits, so the grid coarsens there (Ofir 2014 idea)."""
    freqs = [1 / pmax]
    fmax = 1 / pmin
    while freqs[-1] < fmax:
        f = freqs[-1]
        freqs.append(f + shortest_duration(1 / f, min_duration) * f / (oversample * baseline))
    return np.sort(1 / np.asarray(freqs))


def bls_periodogram(time: np.ndarray, flux: np.ndarray, *, period_range, durations, oversample: float = 3.0,
                    bin_width: float = 0.0) -> tuple[Candidate, np.ndarray, np.ndarray]:
    """BLS search returning the best candidate and the full periodogram (periods, SDE)."""
    from astropy.timeseries import BoxLeastSquares

    t, f = bin_lightcurve(time, flux, bin_width)
    durations = np.asarray(durations, dtype=float)
    periods = period_grid(t.max() - t.min(), period_range[0], period_range[1], durations.min(), oversample)
    model = BoxLeastSquares(t, f)
    res = model.power(periods, durations, objective="snr")
    power = np.asarray(res.power)
    i = int(np.nanargmax(power))
    sde_curve = (power - np.nanmean(power)) / np.nanstd(power)
    cand = Candidate(
        period=float(res.period[i]),
        t0=float(res.transit_time[i]),
        duration=float(res.duration[i]),
        depth=float(res.depth[i]),
        power=float(power[i]),
        sde=float(sde_curve[i]),
    )
    return cand, np.asarray(res.period), sde_curve


def bls(time: np.ndarray, flux: np.ndarray, *, period_range, durations, oversample: float = 3.0,
        bin_width: float = 0.0) -> Candidate:
    return bls_periodogram(time, flux, period_range=period_range, durations=durations, oversample=oversample,
                           bin_width=bin_width)[0]


SEARCHES = {"bls": bls}
