"""Recovery matching, completeness with Wilson intervals, and the gamma-CDF completeness fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .inject import Injection
from .search import Candidate

STATUSES = ("recovered", "alias_half", "alias_double", "wrong_period", "below_threshold", "not_observable")


def _epoch_offset(t_found: float, t_inj: float, period: float) -> float:
    return abs((t_found - t_inj + period / 2) % period - period / 2)


def match(inj: Injection, cand: Candidate | None, *, sde_threshold: float, period_tol: float = 0.01,
          min_transits: int = 2) -> str:
    """Classify one trial. Aliases are tracked separately, never counted as recoveries."""
    if inj.n_transits < min_transits:
        return "not_observable"
    if cand is None or cand.sde < sde_threshold:
        return "below_threshold"
    ratio = cand.period / inj.period
    for target, label in ((1.0, "recovered"), (0.5, "alias_half"), (2.0, "alias_double")):
        if abs(ratio / target - 1) < period_tol:
            # Epoch check against the injected ephemeris, on the shorter of the two periods.
            p = min(cand.period, inj.period)
            if _epoch_offset(cand.t0, inj.t0, p) < inj.duration:
                return label
    return "wrong_period"


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval; z=1 gives ~68% (1-sigma) bounds."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def completeness(trials: pd.DataFrame, by: str = "inj_expected_snr", bins=None) -> pd.DataFrame:
    """Recovered fraction per bin, over observable trials only."""
    obs = trials[trials.status != "not_observable"]
    if bins is None:
        bins = np.geomspace(obs[by].min(), obs[by].max(), 13)
    cut = pd.cut(obs[by], bins)
    g = obs.groupby(cut, observed=False)
    out = pd.DataFrame({
        "lo": bins[:-1],
        "hi": bins[1:],
        "n": g.size().to_numpy(),
        "k": g.apply(lambda d: (d.status == "recovered").sum(), include_groups=False).to_numpy(),
    })
    out["mid"] = np.sqrt(out.lo * out.hi)
    out["frac"] = out.k / out.n
    out["lo68"], out["hi68"] = wilson(out.k, out.n)
    return out


def gamma_cdf(x, a, k, theta):
    """Completeness model of Christiansen et al. (2016): a * GammaCDF(x; shape k, scale theta)."""
    return a * stats.gamma.cdf(x, k, scale=theta)


def fit_gamma_cdf(snr: np.ndarray, recovered: np.ndarray) -> dict | None:
    """Maximum-likelihood fit to per-trial outcomes (no binning)."""
    snr, y = np.asarray(snr, float), np.asarray(recovered, float)
    if len(snr) < 20 or y.sum() == 0:
        return None

    def nll(p):
        a, k, theta = p
        if not (0 < a <= 1 and k > 0 and theta > 0):
            return np.inf
        m = np.clip(gamma_cdf(snr, a, k, theta), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(m) + (1 - y) * np.log(1 - m))

    res = optimize.minimize(nll, x0=[0.95, 10.0, 1.0], method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    if not res.success:
        return None
    a, k, theta = res.x
    snr50 = stats.gamma.ppf(0.5 / a, k, scale=theta) if a > 0.5 else float("nan")
    return {"a": float(a), "k": float(k), "theta": float(theta), "snr_50pct": float(snr50)}
