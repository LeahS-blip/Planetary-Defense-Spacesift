from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from spacesift.null import invert, scramble


def test_scramble_keeps_values_and_breaks_order():
    rng = np.random.default_rng(0)
    t = np.arange(0, 100, 0.02)
    f = np.sin(t)
    ts, fs = scramble(t, f, 10.0, rng)
    assert np.allclose(np.sort(fs), np.sort(f))
    assert np.all(np.diff(ts) > 0)
    assert not np.allclose(fs, f)
    assert np.allclose(invert(invert(f)), f)


def _run(tmp_path: Path, name: str, extra: dict) -> Path:
    from spacesift.runner import run_experiment

    cfg = {"id": name, "question": "test", "seed": 3, "mission": "synthetic",
           "synthetic": {"n_stars": 2, "baseline_d": 120, "noise_ppm": [200, 200]},
           "injection": {"trials_per_star": 4, "period_d": [1, 15]}, "detrend": ["biweight-1.0"], **extra}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return run_experiment(path, allow_dirty=True, out_root=tmp_path)


def test_null_experiment_and_reanalysis(tmp_path: Path):
    import json
    from spacesift.analysis import analyze

    null = _run(tmp_path, "NULL", {"kind": "null", "null": {"scrambles": 2}})
    rows = pd.read_parquet(null / "trials.parquet")
    assert len(rows) == 2 * (1 + 2 * 2)  # per star: inverted + 2 x (scrambled, scrambled_inverted)
    thr = json.loads((null / "record.json").read_text())["summary"]["biweight-1.0"]["threshold_for_far"]
    assert set(thr) == {"0.01", "0.005", "0.001"}
    assert (null / "plots" / "false_alarm_rate.png").exists()

    inj = _run(tmp_path, "INJ", {})
    original = pd.read_parquet(inj / "trials.parquet")
    # Re-deriving at the experiment's own threshold must reproduce every status.
    out = analyze(inj, "same", sde_threshold=7.0)
    again = pd.read_parquet(out / "trials.parquet")
    kept = original[original.star_id.isin(again.star_id.unique())]
    assert (kept.status.to_numpy() == again.status.to_numpy()).all()
    # A threshold from the null run works end to end.
    analyze(inj, "from-null", null_exp=null, far=0.01)
