from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spacesift.data import synthetic
from spacesift.detrend import clean, flatten
from spacesift.evaluate import completeness, fit_gamma_cdf, match, wilson
from spacesift.inject import Injection, Star, a_over_rs, central_depth, inject, t14
from spacesift.search import Candidate, bls


def make_injection(P=7.3, t0=3.1, k=0.03, b=0.3, n_tr=10):
    star = Star("x")
    a = a_over_rs(P, star)
    return Injection(P, t0, k, b, a, t14(P, a, k, b), central_depth(k, b, .4, .25), 3.3, 20, 20, n_tr)


def test_bls_recovers_clear_transit():
    rng = np.random.default_rng(1)
    lc = synthetic(rng, baseline=180, noise_ppm=200, var_amp_ppm=2000, var_period=12)
    inj = make_injection()
    flux = inject(lc.time, lc.flux, inj, Star("x"), lc.exptime, 7)
    t, f = clean(lc.time, flatten(lc.time, flux, "biweight-1.0"))
    cand = bls(t, f, period_range=(1, 20), durations=np.array([1.5, 3, 5]) / 24, bin_width=1 / 24)
    assert match(inj, cand, sde_threshold=7) == "recovered"


def test_match_labels():
    inj = make_injection()
    c = lambda P, t0=inj.t0, sde=12: Candidate(P, t0, 0.1, 1e-3, 1.0, sde)
    assert match(inj, c(inj.period), sde_threshold=7) == "recovered"
    assert match(inj, c(inj.period * 2), sde_threshold=7) == "alias_double"
    assert match(inj, c(inj.period / 2), sde_threshold=7) == "alias_half"
    assert match(inj, c(inj.period * 1.3), sde_threshold=7) == "wrong_period"
    assert match(inj, c(inj.period, sde=5), sde_threshold=7) == "below_threshold"
    assert match(inj, c(inj.period, t0=inj.t0 + inj.period / 2), sde_threshold=7) == "wrong_period"
    assert match(make_injection(n_tr=1), c(inj.period), sde_threshold=7) == "not_observable"


def test_wilson_bounds():
    lo, hi = wilson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert (lo >= 0).all() and (hi <= 1).all()
    assert lo[1] < 0.5 < hi[1]


def test_gamma_fit_recovers_known_curve():
    rng = np.random.default_rng(2)
    snr = np.exp(rng.uniform(np.log(3), np.log(30), 4000))
    from spacesift.evaluate import gamma_cdf
    p = gamma_cdf(snr, 0.97, 20.0, 0.45)
    fit = fit_gamma_cdf(snr, rng.uniform(size=snr.size) < p)
    assert abs(fit["snr_50pct"] - 8.9) < 0.6  # median of Gamma(20, 0.45) scaled by a=0.97


def test_bad_star_is_recorded_not_fatal():
    from spacesift.config import ExperimentConfig
    from spacesift.runner import run_star

    cfg = ExperimentConfig(id="X", question="q", seed=1, mission="synthetic")
    rows, star_row = run_star(cfg, Star("bad", mass=0.0), 0)
    assert rows == [] and "ZeroDivisionError" in star_row["error"]


def test_runner_is_reproducible(tmp_path: Path):
    from spacesift.runner import run_experiment

    cfg = {
        "id": "SS-TEST", "question": "test", "seed": 7, "mission": "synthetic",
        "synthetic": {"n_stars": 2, "baseline_d": 120, "noise_ppm": [200, 200]},
        "injection": {"trials_per_star": 3, "period_d": [1, 15]},
        "detrend": ["biweight-1.0"], "n_jobs": 1,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    a = run_experiment(path, allow_dirty=True, out_root=tmp_path / "a")
    b = run_experiment(path, allow_dirty=True, out_root=tmp_path / "b")
    ta, tb = pd.read_parquet(a / "trials.parquet"), pd.read_parquet(b / "trials.parquet")
    assert len(ta) == 6
    pd.testing.assert_frame_equal(ta, tb)
    assert (a / "record.json").exists() and (a / "plots" / "completeness_vs_snr.png").exists()
    assert not completeness(ta).empty

    # Sharded run + merge must reproduce the unsharded trials exactly.
    from spacesift.runner import merge_shards
    for i in range(2):
        run_experiment(path, allow_dirty=True, out_root=tmp_path / "s", shard=(i, 2))
    merged = merge_shards(tmp_path / "s" / "SS-TEST")
    pd.testing.assert_frame_equal(pd.read_parquet(merged / "trials.parquet"), ta)

    # CI artifacts can arrive nested (shards/shard-1/SS-TEST/shards/shard-1/...); merge must still work.
    import shutil
    nested = tmp_path / "n" / "SS-TEST"
    for i in range(2):
        src = tmp_path / "s" / "SS-TEST" / "shards" / f"shard-{i}"
        shutil.copytree(src, nested / "shards" / f"shard-{i}" / "SS-TEST" / "shards" / f"shard-{i}")
    pd.testing.assert_frame_equal(pd.read_parquet(merge_shards(nested) / "trials.parquet"), ta)
