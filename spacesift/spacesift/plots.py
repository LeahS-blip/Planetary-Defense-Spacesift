"""Standard plots saved next to each experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import completeness, fit_gamma_cdf, gamma_cdf


def plot_experiment(trials: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plot_completeness_snr(trials, out / "completeness_vs_snr.png")
    plot_completeness_radius_period(trials, out / "completeness_radius_period.png")


def plot_completeness_snr(trials: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for spec, d in trials.groupby("detrend"):
        c = completeness(d)
        c = c[c.n > 0]
        # Clip: at 0% or 100% the Wilson bound equals the point up to round-off.
        yerr = [np.clip(c.frac - c.lo68, 0, None), np.clip(c.hi68 - c.frac, 0, None)]
        line = ax.errorbar(c.mid, c.frac, yerr=yerr, fmt="o", capsize=3,
                           label=f"{spec}")
        obs = d[d.status != "not_observable"]
        fit = fit_gamma_cdf(obs.inj_expected_snr, obs.status == "recovered")
        if fit:
            x = np.geomspace(c.lo.min(), c.hi.max(), 200)
            ax.plot(x, gamma_cdf(x, fit["a"], fit["k"], fit["theta"]), color=line[0].get_color(), lw=1.2,
                    label=f"  gamma fit: 50% at SNR {fit['snr_50pct']:.1f}")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Expected SNR")
    ax.set_ylabel("Fraction recovered (68% Wilson interval)")
    ax.set_title("Detection completeness vs expected SNR")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_completeness_radius_period(trials: pd.DataFrame, path: Path) -> None:
    specs = sorted(trials.detrend.unique())
    fig, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4.5), squeeze=False)
    for ax, spec in zip(axes[0], specs):
        d = trials[(trials.detrend == spec) & (trials.status != "not_observable")]
        pb = np.geomspace(d.inj_period.min(), d.inj_period.max(), 8)
        rb = np.geomspace(d.inj_radius_earth.min(), d.inj_radius_earth.max(), 8)
        n, _, _ = np.histogram2d(d.inj_period, d.inj_radius_earth, [pb, rb])
        k, _, _ = np.histogram2d(d.inj_period, d.inj_radius_earth, [pb, rb], weights=d.status == "recovered")
        with np.errstate(invalid="ignore"):
            frac = np.where(n >= 5, k / n, np.nan)  # blank cells with too few trials
        m = ax.pcolormesh(pb, rb, frac.T, vmin=0, vmax=1, cmap="viridis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Radius (Earth radii)")
        ax.set_title(spec)
        fig.colorbar(m, ax=ax, label="Fraction recovered")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trial(time, raw, injected_model, flat, inj, cand, status: str, path: Path) -> None:
    """Three-panel look at one trial: raw + injected model, detrended, folded on the injected period."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(time, raw, ",k", alpha=0.5)
    axes[0].plot(time, injected_model * np.median(raw), "r-", lw=0.6)
    axes[0].set_title(f"Raw flux with injected model (status: {status})")
    axes[1].plot(time, flat, ",k", alpha=0.5)
    axes[1].set_title("Detrended")
    phase = ((time - inj.t0 + 0.5 * inj.period) % inj.period) - 0.5 * inj.period
    win = np.abs(phase) < 3 * inj.duration
    axes[2].plot(phase[win] * 24, flat[win], ".k", ms=2, alpha=0.4)
    axes[2].axvspan(-inj.duration * 12, inj.duration * 12, color="r", alpha=0.1)
    title = f"Folded on injected P={inj.period:.4f} d, depth {inj.depth * 1e6:.0f} ppm, SNR {inj.expected_snr:.1f}"
    if cand is not None:
        title += f" | found P={cand.period:.4f} d, SDE {cand.sde:.1f}"
    axes[2].set_title(title, fontsize=9)
    axes[2].set_xlabel("Hours from injected mid-transit")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
