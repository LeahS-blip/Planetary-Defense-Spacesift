from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spacesift.data import stellar_variability, synthetic_variable
from spacesift.detrend import flatten, prewhiten
from spacesift.evaluate import match
from spacesift.inject import Star, mean_over_central, sample_injection
from spacesift.search import Candidate
from tests.test_pipeline import make_injection


def test_mean_depth_basis_hits_target():
    rng = np.random.default_rng(5)
    t = np.arange(0, 360, 29.4 / 1440)
    f = 1 + rng.normal(0, 150e-6, t.size)
    for _ in range(10):
        inj = sample_injection(rng, t, f, Star("x"), (8, 20), (1, 30), (0, 0.9), depth_basis="mean")
        assert inj.expected_snr == pytest.approx(inj.target_snr, rel=0.15)
    # Same draw, central basis: the mean basis needs a bigger planet for the same SNR.
    a = sample_injection(np.random.default_rng(1), t, f, Star("x"), (10, 10), (5, 5), (0.3, 0.3))
    b = sample_injection(np.random.default_rng(1), t, f, Star("x"), (10, 10), (5, 5), (0.3, 0.3), depth_basis="mean")
    assert b.rp_rs > a.rp_rs
    assert 0.75 < mean_over_central(0.02, 0.3, 15.0, 5.0, 0.4, 0.25) < 0.95


def test_variability_amplitudes():
    t = np.arange(0, 100, 0.02)
    rng = np.random.default_rng(0)
    p = stellar_variability(rng, t, "pulsation", 0.2, 500)
    assert np.std(p) == pytest.approx(500e-6 * np.sqrt((1 + 0.09) / 2), rel=0.05)
    s = stellar_variability(rng, t, "spots", 5.0, 500)
    assert 200e-6 < np.std(s) < 800e-6
    assert np.all(stellar_variability(rng, t, "none", 1.0, 500) == 0)


def test_synthetic_variable_keeps_noise_only():
    lc = synthetic_variable(np.random.default_rng(2), 100, 150, "pulsation", 0.2, 1000)
    assert lc.noise_only is not None and lc.noise_only.size == lc.flux.size
    assert np.std(lc.noise_only) == pytest.approx(150e-6, rel=0.05)
    assert np.std(lc.flux) > 3 * np.std(lc.noise_only)


def test_prewhiten_removes_pulsation_but_biweight_does_not():
    lc = synthetic_variable(np.random.default_rng(3), 200, 150, "pulsation", 4 / 24, 1000)
    resid_pw = flatten(lc.time, lc.flux, "prewhiten+biweight-1.0") - 1
    resid_bw = flatten(lc.time, lc.flux, "biweight-1.0") - 1
    assert np.std(resid_pw) < 1.3 * 150e-6  # back near the white noise
    assert np.std(resid_bw) > 3 * 150e-6  # a 1-day window cannot follow a 4-hour cycle
    assert np.std(prewhiten(lc.time, lc.noise_only) - 1) == pytest.approx(150e-6, rel=0.05)  # noise left alone


def test_star_signal_status():
    inj = make_injection()
    star_p = 1.1440
    cand = Candidate(star_p * 2, 0.3, 0.06, 1e-4, 1.0, 12)
    assert match(inj, cand, sde_threshold=7, star_period=star_p) == "star_signal"
    assert match(inj, cand, sde_threshold=7) == "wrong_period"
    planet = Candidate(inj.period, inj.t0, 0.1, 1e-3, 1.0, 12)
    assert match(inj, planet, sde_threshold=7, star_period=inj.period) == "recovered"  # planet wins ties


def test_small_grid_end_to_end(tmp_path: Path):
    from spacesift.analysis import analyze_grid
    from spacesift.runner import merge_shards, run_experiment

    cfg = yaml.safe_load(Path("configs/SS-0002A.yaml").read_text())
    cfg.update({"id": "GRID-TEST", "n_jobs": 1})
    cfg["synthetic"]["baseline_d"] = 90
    cfg["grid"] = {"pulsation_periods_h": [4], "spot_periods_d": [3], "amplitudes_ppm": [1000],
                   "control": True, "stars_per_cell": 2}
    cfg["injection"]["trials_per_star"] = 2
    cfg["injection"]["period_d"] = [1, 10]
    cfg["detrend"] = ["biweight-1.0", "prewhiten+biweight-1.0"]
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    for i in range(2):
        run_experiment(path, allow_dirty=True, out_root=tmp_path, shard=(i, 2))
    exp = merge_shards(tmp_path / "GRID-TEST")
    trials = pd.read_parquet(exp / "trials.parquet")
    base = pd.read_parquet(exp / "baselines.parquet")
    assert len(trials) == 3 * 2 * 2 * 2  # cells x stars x trials x methods
    assert len(base) == 3 * 2 * 2 * 2  # cells x stars x methods x (plain, inverted)
    assert set(trials.var_kind) == {"none", "pulsation", "spots"}
    out = analyze_grid(exp, snr_band=(8, 20))
    cells = pd.read_csv(out / "cells.csv")
    assert len(cells) == 3 * 2
    for f in ("completeness_maps.png", "capture_maps.png", "best_method.png", "depth_kept.png"):
        assert (out / f).exists()
