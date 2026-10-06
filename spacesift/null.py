"""False-alarm (null) light curves: inverted and block-scrambled versions of a detrended light curve."""

from __future__ import annotations

import numpy as np


def invert(flux: np.ndarray) -> np.ndarray:
    """Flip about 1: dips become bumps, so any surviving dip-like signal is noise or systematics."""
    return 2.0 - flux


def scramble(time: np.ndarray, flux: np.ndarray, block_d: float, rng: np.random.Generator):
    """Cut into blocks of block_d days and lay them back down in random order.

    Each block keeps its internal cadence; blocks are placed end to end one
    cadence apart, so data gaps between blocks are dropped. This destroys strict
    periodicity across blocks while keeping the local noise properties.
    """
    cadence = np.median(np.diff(time))
    edges = np.arange(time.min(), time.max() + block_d, block_d)
    idx = np.digitize(time, edges)
    blocks = [np.flatnonzero(idx == b) for b in np.unique(idx)]
    order = rng.permutation(len(blocks))
    t_out, f_out, t0 = [], [], time.min()
    for b in order:
        sel = blocks[b]
        tb = time[sel] - time[sel][0] + t0
        t_out.append(tb)
        f_out.append(flux[sel])
        t0 = tb[-1] + cadence
    return np.concatenate(t_out), np.concatenate(f_out)


def null_variants(time, flux, cfg_null, rng_for):
    """Yield (kind, k, time, flux) for each null light curve. rng_for(k) gives the k-th scramble's RNG."""
    if cfg_null.inverted:
        yield "inverted", 0, time, invert(flux)
    for k in range(cfg_null.scrambles):
        t_s, f_s = scramble(time, flux, cfg_null.block_d, rng_for(k))
        yield "scrambled", k, t_s, f_s
        yield "scrambled_inverted", k, t_s, invert(f_s)
