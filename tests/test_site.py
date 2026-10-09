import json
from pathlib import Path

from spacesift.site import export_experiments


def test_export_experiments(tmp_path: Path):
    exp = tmp_path / "experiments"
    for name in ("SS-9", "_scratch"):
        d = exp / name
        (d / "plots").mkdir(parents=True)
        (d / "record.json").write_text(json.dumps({"id": name, "question": "q", "n_trials": 5, "config": {"big": 1},
                                                   "summary": {"bw": {"status_counts": {"recovered": 1},
                                                                      "gamma_cdf_fit": {"snr_50pct": float("nan")}}}}))
        (d / "plots" / "a.png").write_bytes(b"png")
        (d / "shards" / "shard-0").mkdir(parents=True)
        (d / "shards" / "shard-0" / "x.png").write_bytes(b"png")
    (exp / "SS-9" / "NOTES.md").write_text("notes")
    (exp / "SS-9" / "analysis" / "far1pct").mkdir(parents=True)
    (exp / "SS-9" / "analysis" / "far1pct" / "analysis.json").write_text(json.dumps({"thresholds": {"bw": 6.3}}))
    out = tmp_path / "lab" / "data" / "experiments.json"
    export_experiments(exp, out, "/spacesift/experiments")
    data = json.loads(out.read_text())  # strict JSON: NaN must be gone
    assert [e["id"] for e in data] == ["SS-9"]  # scratch runs left out
    e = data[0]
    assert e["images"] == ["/spacesift/experiments/SS-9/plots/a.png"]  # shard plots left out
    assert e["notes"] == "notes" and e["analyses"]["far1pct"]["thresholds"] == {"bw": 6.3}
    assert "config" not in e and e["summary"]["bw"]["gamma_cdf_fit"]["snr_50pct"] is None
