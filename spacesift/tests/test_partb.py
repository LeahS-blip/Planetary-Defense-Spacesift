import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacesift.config import ExperimentConfig
from spacesift.data import read_stars, synthetic_variable
from spacesift.detrend import flatten
from spacesift.inject import p2p_noise
from spacesift.runner import cell_for, measured_variability


def test_cleaned_p2p_recovers_white_noise_under_variability():
    for kind, period, amp in (("pulsation", 4 / 24, 1000), ("pulsation", 1.5 / 24, 1000), ("spots", 3.0, 1000)):
        lc = synthetic_variable(np.random.default_rng(1), 200, 150, kind, period, amp)
        raw = p2p_noise(lc.flux)
        cleaned = p2p_noise(flatten(lc.time, lc.flux, "prewhiten+biweight-0.5"))
        assert cleaned == pytest.approx(150e-6, rel=0.12), (kind, period)
        if period < 0.2:
            assert raw > 1.3 * cleaned  # fast pulsation inflates the raw estimate


def test_measured_variability_finds_pulsation():
    lc = synthetic_variable(np.random.default_rng(2), 200, 150, "pulsation", 4 / 24, 500)
    m = measured_variability(lc.time, lc.flux)
    assert m["meas_var_period_d"] == pytest.approx(4 / 24, rel=0.01)
    assert m["meas_var_amp_ppm"] == pytest.approx(500, rel=0.15)


def test_real_star_cells_from_star_list(tmp_path: Path):
    p = tmp_path / "stars.csv"
    pd.DataFrame({"star_id": ["1", "2"], "radius": [1.0, 1.1], "mass": [1.0, 1.0],
                  "cell": ["spots-P3d-A300", "dsct"], "var_kind": ["spots", "dsct"],
                  "var_period_d": [3.0, 0.0], "var_amp_ppm": [300.0, 0.0], "cat_period_d": [2.7, np.nan]}).to_csv(p, index=False)
    stars = read_stars(p)
    cfg = ExperimentConfig(id="X", question="q", seed=1, mission="kepler", stars_file=str(p))
    assert cell_for(cfg, 0, stars[0]) == {"cell": "spots-P3d-A300", "var_kind": "spots", "var_period_d": 3.0,
                                         "var_amp_ppm": 300.0}
    assert cell_for(cfg, 1, stars[1])["var_kind"] == "dsct"


def test_predict_and_compare(tmp_path: Path):
    from spacesift.partb import compare, predict

    methods = ["biweight-1.0", "prewhiten+biweight-1.0"]
    a = pd.DataFrame([{"detrend": m, "cell": i, "var_kind": k, "var_period_d": p, "var_amp_ppm": 300.0,
                       "completeness": c, "lo68": c - 0.05, "hi68": c + 0.05}
                      for i, (k, p) in enumerate([("spots", 3.0), ("pulsation", 4 / 24)])
                      for m, c in zip(methods, (0.1, 0.9))])
    cells_a = tmp_path / "a.csv"
    a.to_csv(cells_a, index=False)
    stars = tmp_path / "stars.csv"
    pd.DataFrame({"star_id": [str(i) for i in range(12)] + ["d1"], "var_kind": ["spots"] * 12 + ["dsct"],
                  "cell": ["spots-P3d-A300"] * 12 + ["dsct"], "var_period_d": [3.0] * 12 + [0.0],
                  "var_amp_ppm": [300.0] * 12 + [0.0]}).to_csv(stars, index=False)
    pred = predict(cells_a, stars, tmp_path / "pred.csv")
    assert set(pred.detrend) == set(methods) and (pred.n_stars == 12).all()

    exp = tmp_path / "B"
    out = exp / "analysis" / "far1pct"
    out.mkdir(parents=True)
    pd.DataFrame([{"detrend": m, "cell": "spots-P3d-A300", "var_kind": "spots", "var_period_d": 3.0,
                   "var_amp_ppm": 300.0, "completeness": c, "lo68": c - 0.05, "hi68": c + 0.05}
                  for m, c in zip(methods, (0.12, 0.5))]).to_csv(out / "cells.csv", index=False)
    pd.DataFrame([{"star_id": "d1", "detrend": m, "var_kind": "dsct", "status": s, "inj_expected_snr": 12.0}
                  for m in methods for s in ("recovered", "below_threshold")]).to_parquet(out / "trials.parquet")
    pd.DataFrame({"star_id": ["d1"], "meas_var_period_d": [0.17], "meas_var_amp_ppm": [280.0]}).to_parquet(
        exp / "stars.parquet")
    (out / "analysis.json").write_text(json.dumps({"snr_band": [10, 16]}))
    compare(exp, "far1pct", cells_a, tmp_path / "pred.csv")
    table = pd.read_csv(out / "comparison.csv")
    spots = table[table.cell == "spots-P3d-A300"].set_index("detrend")
    assert bool(spots.loc["biweight-1.0", "agrees_2sigma"]) and not bool(spots.loc["prewhiten+biweight-1.0", "agrees_2sigma"])
    assert (out / "plots" / "observed_vs_predicted.png").exists()
