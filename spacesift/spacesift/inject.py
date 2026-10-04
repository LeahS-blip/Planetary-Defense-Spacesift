"""Transit model, injection-parameter sampling, and injection.

The transit model integrates a quadratically limb-darkened stellar disk over
the area the planet covers, annulus by annulus. It is exact up to quadrature
error, has no compiled dependencies, and is checked against the analytic
uniform-disk case in tests/test_inject.py.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

G = 6.674e-11
M_SUN = 1.989e30
R_SUN = 6.957e8
R_EARTH = 6.371e6
DAY = 86400.0


@dataclass
class Star:
    star_id: str
    radius: float = 1.0  # R_sun
    mass: float = 1.0  # M_sun
    teff: float = 5772.0
    u1: float = 0.40  # quadratic limb darkening, Kepler band
    u2: float = 0.25


@dataclass
class Injection:
    period: float  # days
    t0: float  # days, same time system as the light curve
    rp_rs: float  # planet-to-star radius ratio k
    b: float  # impact parameter
    a_rs: float  # semi-major axis in stellar radii
    duration: float  # T14, days
    depth: float  # central depth, fractional
    radius_earth: float
    target_snr: float
    expected_snr: float
    n_transits: int

    def to_dict(self) -> dict:
        return {f"inj_{k}": v for k, v in asdict(self).items()}


# ---------------------------------------------------------------- geometry

def a_over_rs(period: float, star: Star) -> float:
    a = (G * star.mass * M_SUN * (period * DAY) ** 2 / (4 * np.pi**2)) ** (1 / 3)
    return a / (star.radius * R_SUN)


def t14(period: float, a_rs: float, k: float, b: float) -> float:
    """Total transit duration for a circular orbit (Seager & Mallen-Ornelas 2003)."""
    inc = np.arccos(b / a_rs)
    arg = np.sqrt(max((1 + k) ** 2 - b**2, 0.0)) / (a_rs * np.sin(inc))
    return period / np.pi * np.arcsin(min(arg, 1.0))


def central_depth(k: float, b: float, u1: float, u2: float) -> float:
    """Small-planet depth at mid-transit, used to map depth <-> k."""
    mu = np.sqrt(max(1 - b**2, 0.0))
    intensity = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    return k**2 * intensity / (1 - u1 / 3 - u2 / 6)


# ------------------------------------------------------------- flux model

def _blocked_fraction(z: np.ndarray, k: float, u1: float, u2: float, n_r: int = 64) -> np.ndarray:
    """Fraction of stellar flux blocked by a planet at sky separation z (stellar radii)."""
    out = np.zeros_like(z, dtype=float)
    hit = z < 1 + k
    if not hit.any():
        return out
    zz = np.maximum(z[hit], 1e-9)[:, None]
    r_lo = np.clip(zz - k, 0.0, 1.0)
    r_hi = np.clip(zz + k, 0.0, 1.0)
    # Midpoint rule in r between the overlap limits, per epoch.
    s = (np.arange(n_r) + 0.5) / n_r
    r = r_lo + (r_hi - r_lo) * s[None, :]
    dr = (r_hi - r_lo) / n_r
    cos_arg = np.clip((r**2 + zz**2 - k**2) / (2 * r * zz), -1.0, 1.0)
    arc = 2 * r * np.arccos(cos_arg)  # length of annulus r inside the planet
    mu = np.sqrt(np.clip(1 - r**2, 0.0, 1.0))
    intensity = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    blocked = (intensity * arc).sum(axis=1) * dr[:, 0]
    total = np.pi * (1 - u1 / 3 - u2 / 6)
    out[hit] = blocked / total
    return out


def transit_model(
    time: np.ndarray,
    inj: Injection,
    u1: float,
    u2: float,
    exptime: float | None = None,
    supersample: int = 1,
) -> np.ndarray:
    """Relative flux (1 out of transit). Supersamples over exptime if given."""
    if exptime and supersample > 1:
        offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
        t = (time[:, None] + offsets[None, :] * exptime).ravel()
    else:
        t, supersample = time, 1
    phase = 2 * np.pi * (t - inj.t0) / inj.period
    inc = np.arccos(inj.b / inj.a_rs)
    x = inj.a_rs * np.sin(phase)
    y = inj.a_rs * np.cos(phase) * np.cos(inc)
    z = np.sqrt(x**2 + y**2)
    z = np.where(np.cos(phase) > 0, z, np.inf)  # planet behind the star: no transit
    flux = 1 - _blocked_fraction(z, inj.rp_rs, u1, u2)
    return flux.reshape(-1, supersample).mean(axis=1)


# ------------------------------------------------------- noise and sampling

def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def cdpp(time: np.ndarray, flux_flat: np.ndarray, duration: float) -> float:
    """Noise on the transit-duration timescale: robust scatter of duration-wide bin means."""
    edges = np.arange(time.min(), time.max() + duration, duration)
    idx = np.digitize(time, edges)
    cadence = np.median(np.diff(time))
    need = max(1, int(0.5 * duration / cadence))
    sums = np.bincount(idx, weights=flux_flat)
    counts = np.bincount(idx)
    good = counts >= need
    return robust_std(sums[good] / counts[good])


def count_transits(time: np.ndarray, period: float, t0: float, duration: float) -> int:
    """Transits with at least half of their expected in-transit cadences present."""
    cadence = np.median(np.diff(time))
    need = max(1, int(0.5 * duration / cadence))
    epoch = np.round((time - t0) / period)
    in_tr = np.abs(time - (t0 + epoch * period)) < duration / 2
    _, counts = np.unique(epoch[in_tr], return_counts=True)
    return int((counts >= need).sum())


def sample_injection(
    rng: np.random.Generator,
    time: np.ndarray,
    flux_flat: np.ndarray,
    star: Star,
    snr_range: tuple[float, float],
    period_range: tuple[float, float],
    b_range: tuple[float, float],
) -> Injection:
    """Draw P, b, t0 and a target SNR, then solve for the radius that gives that SNR here."""
    period = float(np.exp(rng.uniform(*np.log(period_range))))
    b = float(rng.uniform(*b_range))
    t0 = float(time.min() + rng.uniform(0, period))
    target = float(np.exp(rng.uniform(*np.log(snr_range))))
    a_rs = a_over_rs(period, star)

    k = 0.01  # first guess; T14 depends only weakly on k, so two passes suffice
    for _ in range(2):
        dur = t14(period, a_rs, k, b)
        sigma = cdpp(time, flux_flat, dur)
        n_tr = count_transits(time, period, t0, dur)
        depth = target * sigma / np.sqrt(max(n_tr, 1))
        k = float(np.sqrt(depth / central_depth(1.0, b, star.u1, star.u2)))
    dur = t14(period, a_rs, k, b)
    n_tr = count_transits(time, period, t0, dur)
    depth = central_depth(k, b, star.u1, star.u2)
    expected = depth / cdpp(time, flux_flat, dur) * np.sqrt(n_tr)
    return Injection(
        period=period,
        t0=t0,
        rp_rs=k,
        b=b,
        a_rs=a_rs,
        duration=dur,
        depth=depth,
        radius_earth=k * star.radius * R_SUN / R_EARTH,
        target_snr=target,
        expected_snr=float(expected),
        n_transits=n_tr,
    )


def inject(time, flux, inj: Injection, star: Star, exptime: float | None, supersample: int) -> np.ndarray:
    return flux * transit_model(time, inj, star.u1, star.u2, exptime, supersample)
