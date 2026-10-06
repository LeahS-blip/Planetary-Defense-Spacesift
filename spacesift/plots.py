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
        _plain_log_ticks(ax.xaxis, [1, 2, 3, 5, 10, 20, 30])
        _plain_log_ticks(ax.yaxis, [0.2, 0.3, 0.5, 1, 2, 3])
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Radius (Earth radii)")
        ax.set_title(spec)
        fig.colorbar(m, ax=ax, label="Fraction recovered")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plain_log_ticks(axis, ticks) -> None:
    """Log axis labelled with plain numbers instead of overlapping 'a x 10^b' minor labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def plot_null(trials: pd.DataFrame, out: Path) -> None:
    """Survival curve of null-search SDE: the fraction of noise-only searches above each threshold."""
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (spec, kind), d in trials.groupby(["detrend", "null_kind"]):
        sde = np.sort(d.found_sde.to_numpy())
        surv = 1 - np.arange(sde.size) / sde.size
        ax.step(sde, surv, where="post", label=f"{spec} · {kind} (n={sde.size})")
    allsde = trials.found_sde.to_numpy()
    for far, ls in ((0.01, "--"), (0.001, ":")):
        thr = np.quantile(allsde, 1 - far)
        ax.axvline(thr, color="k", ls=ls, lw=1, label=f"{far:.1%} false alarms: SDE {thr:.2f}")
    ax.axvline(7, color="r", lw=1, alpha=0.6, label=f"SDE 7: {(allsde > 7).mean():.1%} false alarms")
    ax.set_yscale("log")
    ax.set_xlabel("Top-peak SDE of a noise-only search")
    ax.set_ylabel("Fraction of searches above")
    ax.set_title("False-alarm rate vs SDE threshold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "false_alarm_rate.png", dpi=150)
    plt.close(fig)


def plot_star(time, flux, flat, cand, inv_cand, star_id: str, path: Path) -> None:
    """Why does this star have a pre-injection signal? Raw, detrended, and folded on the best period."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(time, flux, ",k", alpha=0.5)
    axes[0].set_title(f"KIC {star_id}: raw PDCSAP flux")
    axes[1].plot(time, flat, ",k", alpha=0.5)
    axes[1].set_title("Detrended")
    phase = ((time - cand.t0 + 0.5 * cand.period) % cand.period) - 0.5 * cand.period
    # Show at most +-12 h (or half the period): enough for several cycles of fast
    # variability, and narrow enough that a short dip at long period stays visible.
    half = min(12.0, 12 * cand.period)
    win = np.abs(phase * 24) < half
    axes[2].plot(phase[win] * 24, flat[win], ".k", ms=1.5, alpha=0.3)
    # Binned fold so a shallow dip is visible over the scatter.
    bins = np.linspace(-half, half, 121)
    idx = np.digitize(phase * 24, bins)
    centers = 0.5 * (bins[1:] + bins[:-1])
    med = [np.median(flat[idx == i]) if (idx == i).sum() > 5 else np.nan for i in range(1, len(bins))]
    axes[2].plot(centers, med, "r-", lw=1.2)
    lo, hi = np.nanpercentile(flat, [0.5, 99.5])
    axes[2].set_ylim(lo, hi)
    axes[2].set_title(f"Folded on P={cand.period:.5f} d, SDE {cand.sde:.1f}, depth {cand.depth * 1e6:.0f} ppm, "
                      f"duration {cand.duration * 24:.1f} h | same search on inverted flux: SDE {inv_cand.sde:.1f}",
                      fontsize=9)
    axes[2].set_xlabel("Hours from mid-event")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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
