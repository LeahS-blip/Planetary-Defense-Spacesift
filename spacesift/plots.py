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


def _grid_panel(ax, cells: pd.DataFrame, kind: str, value: str, cmap: str, title: str) -> None:
    """Heatmap of one value over variability period (rows) x amplitude (columns), with numbers."""
    d = cells[cells.var_kind == kind]
    periods = sorted(d.var_period_d.unique())
    amps = sorted(d.var_amp_ppm.unique())
    grid = np.full((len(periods), len(amps)), np.nan)
    for r in d.itertuples():
        grid[periods.index(r.var_period_d), amps.index(r.var_amp_ppm)] = getattr(r, value)
    ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, aspect="auto", origin="lower")
    for i in range(len(periods)):
        for j in range(len(amps)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (grid[i, j] < 0.5) == (cmap == "viridis") else "black")
    ax.set_xticks(range(len(amps)), [f"{a:g}" for a in amps], fontsize=7)
    labels = [f"{p * 24:g} h" if p < 1 else f"{p:g} d" for p in periods]
    ax.set_yticks(range(len(periods)), labels, fontsize=7)
    ax.set_title(title, fontsize=8)


def plot_grid(cells: pd.DataFrame, trials: pd.DataFrame, out: Path) -> None:
    """SS-0002 maps: completeness and capture rate per method; best method per cell; depth kept."""
    out.mkdir(parents=True, exist_ok=True)
    specs = list(dict.fromkeys(trials.detrend))  # config order
    kinds = [k for k in ("pulsation", "spots") if (cells.var_kind == k).any()]
    for value, cmap, fname, label in (("completeness", "viridis", "completeness_maps.png", "completeness"),
                                      ("capture_rate", "magma_r", "capture_maps.png", "captured by star")):
        fig, axes = plt.subplots(len(specs), len(kinds), figsize=(4.2 * len(kinds), 2.4 * len(specs)),
                                 squeeze=False)
        for row, spec in enumerate(specs):
            ctrl = cells[(cells.detrend == spec) & (cells.var_kind == "none")]
            note = f" | no variability: {ctrl[value].iloc[0]:.2f}" if len(ctrl) else ""
            for col, kind in enumerate(kinds):
                _grid_panel(axes[row, col], cells[cells.detrend == spec], kind, value, cmap,
                            f"{spec} · {kind} · {label}{note}")
        for ax in axes[-1]:
            ax.set_xlabel("Variability amplitude (ppm)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("Variability period", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=140)
        plt.close(fig)

    # Best method per cell: highest completeness in the SNR band.
    best = cells.loc[cells.groupby("cell").completeness.idxmax()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 3.4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        d = best[best.var_kind == kind]
        periods, amps = sorted(d.var_period_d.unique()), sorted(d.var_amp_ppm.unique())
        ax.set_xlim(-0.5, len(amps) - 0.5)
        ax.set_ylim(-0.5, len(periods) - 0.5)
        for r in d.itertuples():
            ax.text(amps.index(r.var_amp_ppm), periods.index(r.var_period_d),
                    f"{r.detrend}\n{r.completeness:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(amps)), [f"{a:g}" for a in amps], fontsize=7)
        ax.set_yticks(range(len(periods)), [f"{p * 24:g} h" if p < 1 else f"{p:g} d" for p in periods], fontsize=7)
        ax.set_xlabel("Variability amplitude (ppm)", fontsize=8)
        ax.set_title(f"Best method per cell ({kind})", fontsize=9)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "best_method.png", dpi=140)
    plt.close(fig)

    # The cost side: how much transit each method keeps, by transit duration (no variability).
    ctrl = trials[(trials.var_kind == "none") & trials.detrended_depth_ratio.notna()]
    if len(ctrl):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        bins = [0, 2, 3, 4, 5, 7, 12]
        for spec in specs:
            d = ctrl[ctrl.detrend == spec]
            med = d.groupby(pd.cut(d.inj_duration * 24, bins), observed=True).detrended_depth_ratio.median()
            ax.plot([iv.mid for iv in med.index], med.to_numpy(), "o-", label=spec)
        ax.axhline(1, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("Transit duration (hours)")
        ax.set_ylabel("Transit depth kept after detrending")
        ax.set_title("Cost of each method on a quiet star")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "depth_kept.png", dpi=140)
        plt.close(fig)


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    """Part B: observed completeness on real stars vs the Part A prediction, per bin and method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, alpha=0.5)
    for spec, d in table.groupby("detrend"):
        for judged, marker in ((True, "o"), (False, "x")):
            e = d[d.judged == judged]
            if len(e):
                ax.errorbar(e.pred, e.completeness, yerr=[np.clip(e.completeness - e.lo68, 0, None),
                                                          np.clip(e.hi68 - e.completeness, 0, None)],
                            fmt=marker, capsize=2, ms=5, label=f"{spec}" + ("" if judged else " (<10 stars)"))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Predicted by Part A (simulation)")
    ax.set_ylabel("Observed on real Kepler stars")
    ax.set_title("Does the synthetic map predict real stars?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
