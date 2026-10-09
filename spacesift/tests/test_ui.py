import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import math

from spacesift import json_safe
from spacesift.ui.server import make_handler
from spacesift.ui.trial import run_ui_trial


def test_json_safe_replaces_nan():
    out = json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2}})
    assert out == {"a": None, "b": [1.0, None], "c": {"d": 2}}
    json.loads(json.dumps(out))  # strict JSON


def test_ui_trial_pulsator_biweight_vs_prewhitening():
    base = {"source": "synthetic", "var_kind": "pulsation", "var_period_d": 4 / 24, "var_amp_ppm": 300,
            "noise_ppm": 150, "baseline_d": 90, "seed": 1,
            "planet": {"enabled": True, "size_mode": "snr", "snr": 14, "period_d": 5.0, "b": 0.3}}
    missed = run_ui_trial({**base, "detrend": "biweight-1.0"})
    found = run_ui_trial({**base, "detrend": "prewhiten+biweight-1.0"})
    assert missed["status"] != "recovered"
    assert found["status"] == "recovered"
    for d in (missed, found):
        json.loads(json.dumps(json_safe(d)))
        assert len(d["lc"]["t"]) == len(d["lc"]["raw"]) and len(d["periodogram"]["period"]) <= 3000


def test_ui_trial_without_planet():
    d = run_ui_trial({"source": "synthetic", "var_kind": "none", "baseline_d": 90, "seed": 2,
                      "planet": {"enabled": False}, "detrend": "biweight-1.0", "sde_threshold": 7.25})
    assert d["injection"] is None and d["status"] in ("nothing_found", "false_alarm")


def test_server_serves_page_and_experiments(tmp_path: Path):
    exp = tmp_path / "EXP-1"
    exp.mkdir()
    (exp / "record.json").write_text('{"id": "EXP-1", "question": "q", "n_trials": 3, "summary": {"x": NaN}}')
    (exp / "plot.png").write_bytes(b"\x89PNG fake")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([tmp_path]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"SPACESIFT" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/experiments").read())[0]["id"] == "EXP-1"
        detail = json.loads(urllib.request.urlopen(base + "/api/experiment/EXP-1").read())  # NaN made safe
        assert detail["images"] == ["plot.png"]
        assert urllib.request.urlopen(base + "/files/EXP-1/plot.png").read().startswith(b"\x89PNG")
        try:  # no escaping the experiment folder
            urllib.request.urlopen(base + "/files/EXP-1/../EXP-1/record.json/../../secret")
            assert False, "path traversal should 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
