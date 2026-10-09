"""One interactive trial for the UI: build a light curve, plant a planet, clean, search, grade."""

from __future__ import annotations

import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_mission, synthetic_variable
from ..detrend import clean, flatten
from ..evaluate import match
from ..inject import (R_EARTH, R_SUN, Injection, Star, a_over_rs, central_depth, cdpp, count_transits,
                      mean_over_central, p2p_noise, t14, transit_model)
from ..search import bls_periodogram

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DURATIONS_HR = [1.5, 2.5, 4.0, 6.0, 9.0]


def _star_for_kic(kic: str) -> Star:
    """Stellar radius and mass from any frozen star list that contains this KIC, else Sun-like."""
    for csv in sorted((PROJECT_ROOT / "configs" / "stars").glob("*.csv")):
        df = pd.read_csv(csv, dtype={"star_id": str})
        hit = df[df.star_id == str(kic)]
        if len(hit):
            r = hit.iloc[0]
            return Star(star_id=str(kic), radius=float(r.radius), mass=float(r.mass))
    return Star(star_id=str(kic))


def _bin(x: np.ndarray, y: np.ndarray, n: int):
    """Median-bin y on x into n bins (for folded views)."""
    edges = np.linspace(x.min(), x.max(), n + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(y[idx == i]) if np.any(idx == i) else np.nan for i in range(n)])
    keep = np.isfinite(med)
    return centers[keep], med[keep]


def _thin(x: np.ndarray, y: np.ndarray, n: int):
    """Keep the max of y in n chunks (periodograms: peaks survive thinning)."""
    if x.size <= n:
        return x, y
    chunks = np.array_split(np.arange(x.size), n)
    pick = np.array([c[np.argmax(y[c])] for c in chunks])
    return x[pick], y[pick]


def _clean_floats(a) -> list:
    return [None if not np.isfinite(v) else round(float(v), 7) for v in a]


def run_ui_trial(p: dict) -> dict:
    started = clock.perf_counter()
    seed = int(p.get("seed", 1))
    rng = np.random.default_rng(seed)

    # 1. Light curve.
    if p.get("source") == "kepler":
        kic = str(p["kic"]).strip()
        quarters = [int(q) for q in p.get("quarters", [2, 3, 4, 5])]
        lc = load_mission(kic, "kepler", quarters, PROJECT_ROOT / "cache")
        star = _star_for_kic(kic)
        sigma = p2p_noise(flatten(lc.time, lc.flux, "prewhiten+biweight-0.5"))
        noise = 1 + np.random.default_rng(seed + 1).normal(0, sigma, lc.time.size)
        label = f"KIC {kic} (Kepler Q{quarters[0]}-Q{quarters[-1]})"
    else:
        lc = synthetic_variable(rng, float(p.get("baseline_d", 120)), float(p.get("noise_ppm", 150)),
                                p.get("var_kind", "none"), float(p.get("var_period_d", 1.0)),
                                float(p.get("var_amp_ppm", 0)))
        star = Star(star_id="simulated")
        noise, sigma = lc.noise_only, float(p.get("noise_ppm", 150)) * 1e-6
        label = "Simulated Sun-like star"

    # 2. Planet: sized by target SNR or by radius; SNR on the mean in-transit depth vs white noise.
    planet = p.get("planet", {})
    inj = None
    model = np.ones_like(lc.time)
    if planet.get("enabled", True):
        period = float(planet.get("period_d", 5.0))
        b = float(planet.get("b", 0.3))
        a_rs = a_over_rs(period, star)
        t0 = float(lc.time.min() + rng.uniform(0, period))
        if planet.get("size_mode", "snr") == "radius":
            k = float(planet.get("radius_earth", 1.0)) * R_EARTH / (star.radius * R_SUN)
        else:
            target = float(planet.get("snr", 12.0))
            k = 0.01
            for _ in range(3):
                dur = t14(period, a_rs, k, b)
                n_tr = max(count_transits(lc.time, period, t0, dur), 1)
                depth_mean = target * cdpp(lc.time, noise, dur) / np.sqrt(n_tr)
                k = float(np.sqrt(depth_mean / (central_depth(1.0, b, star.u1, star.u2)
                                                * mean_over_central(k, b, a_rs, period, star.u1, star.u2))))
        dur = t14(period, a_rs, k, b)
        n_tr = count_transits(lc.time, period, t0, dur)
        depth = central_depth(k, b, star.u1, star.u2)
        snr = depth * mean_over_central(k, b, a_rs, period, star.u1, star.u2) / cdpp(lc.time, noise, dur) * np.sqrt(n_tr)
        inj = Injection(period, t0, k, b, a_rs, dur, depth, k * star.radius * R_SUN / R_EARTH, snr, snr, n_tr)
        model = transit_model(lc.time, inj, star.u1, star.u2, lc.exptime, 7)

    raw = lc.flux * model

    # 3. Clean and search.
    spec = p.get("detrend", "biweight-1.0")
    flat = flatten(lc.time, raw, spec)
    t, f = clean(lc.time, flat)
    pmin, pmax = float(p.get("period_min", 1.0)), float(p.get("period_max", 30.0))
    cand, periods, sde = bls_periodogram(t, f, period_range=(pmin, pmax), durations=np.array(DURATIONS_HR) / 24,
                                         oversample=3.0, bin_width=1 / 24)
    threshold = float(p.get("sde_threshold", 7.25))
    if inj is not None:
        status = match(inj, cand, sde_threshold=threshold)
    else:
        status = "false_alarm" if cand.sde >= threshold else "nothing_found"

    # 4. Views.
    fold_p = cand.period
    phase_h = (((t - cand.t0 + 0.5 * fold_p) % fold_p) - 0.5 * fold_p) * 24
    win = np.abs(phase_h) < max(3 * cand.duration * 24, 6)
    fx, fy = _bin(phase_h[win], f[win], 80) if win.sum() > 20 else (np.array([]), np.array([]))
    pp, ps = _thin(periods, sde, 3000)
    out = {
        "label": label,
        "status": status,
        "threshold": threshold,
        "white_noise_ppm": round(sigma * 1e6, 1),
        "candidate": {k: round(v, 6) for k, v in cand.__dict__.items()},
        "injection": None if inj is None else {
            "period_d": round(inj.period, 5), "t0": round(inj.t0, 5), "radius_earth": round(inj.radius_earth, 3),
            "depth_ppm": round(inj.depth * 1e6, 1), "duration_h": round(inj.duration * 24, 2),
            "snr": round(inj.expected_snr, 2), "n_transits": int(inj.n_transits)},
        "lc": {"t": _clean_floats(lc.time), "raw": _clean_floats((raw - 1) * 1e6),
               "model": _clean_floats((model - 1) * 1e6), "flat_t": _clean_floats(t),
               "flat": _clean_floats((f - 1) * 1e6)},
        "periodogram": {"period": _clean_floats(pp), "sde": _clean_floats(ps)},
        "fold": {"phase_h": _clean_floats(phase_h[win][:8000]), "flux": _clean_floats((f[win][:8000] - 1) * 1e6),
                 "bin_h": _clean_floats(fx), "bin_flux": _clean_floats((fy - 1) * 1e6)},
        "seconds": round(clock.perf_counter() - started, 1),
    }
    return out
