"""Every shipped config must load: CI runs them straight from the YAML files."""

from pathlib import Path

import pytest
import yaml

from spacesift.config import load_config

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_config_loads(path):
    cfg, _ = load_config(path)
    # YAML turns bare null/yes/no/on/off into None/True/False: no key may come back non-string.
    assert all(isinstance(k, str) for k in yaml.safe_load(path.read_text()))
    if cfg.stars_file:
        assert (path.parent.parent / cfg.stars_file).exists()
