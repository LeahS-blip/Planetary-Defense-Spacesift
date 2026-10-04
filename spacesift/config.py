"""Experiment configuration. One YAML file fully defines an experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class SyntheticStars(BaseModel):
    n_stars: int = 20
    baseline_d: float = 360.0
    noise_ppm: tuple[float, float] = (100.0, 400.0)  # per cadence, drawn log-uniform per star
    var_amp_ppm: tuple[float, float] = (0.0, 0.0)
    var_period_d: tuple[float, float] = (5.0, 30.0)


class InjectionCfg(BaseModel):
    trials_per_star: int = 50
    expected_snr: tuple[float, float] = (3.0, 30.0)  # log-uniform
    period_d: tuple[float, float] = (1.0, 30.0)  # log-uniform
    b: tuple[float, float] = (0.0, 0.9)
    # Detrending used only to measure the noise that sets the target SNR, so the
    # SNR scale is the same whichever detrending is being tested.
    noise_detrend: str = "biweight-1.0"
    supersample: int = 7


class SearchCfg(BaseModel):
    method: Literal["bls"] = "bls"
    durations_hr: list[float] = Field(default_factory=lambda: [1.5, 2.5, 4.0, 6.0, 9.0])
    oversample: float = 3.0
    bin_min: float = 60.0  # bin to this many minutes before searching; 0 = no binning


class RecoveryCfg(BaseModel):
    sde_threshold: float = 7.0
    period_tol: float = 0.01
    min_transits: int = 2


class ExperimentConfig(BaseModel):
    id: str
    question: str
    seed: int
    mission: Literal["kepler", "tess", "synthetic"]
    stars_file: str | None = None
    quarters: list[int] | None = None  # Kepler quarters, or TESS sectors
    max_stars: int | None = None
    synthetic: SyntheticStars = Field(default_factory=SyntheticStars)
    injection: InjectionCfg = Field(default_factory=InjectionCfg)
    detrend: list[str] = Field(default_factory=lambda: ["biweight-1.0"])
    search: SearchCfg = Field(default_factory=SearchCfg)
    recovery: RecoveryCfg = Field(default_factory=RecoveryCfg)
    n_jobs: int = 1
    cache_dir: str = "cache"

    @model_validator(mode="after")
    def _check(self):
        if self.mission != "synthetic" and not self.stars_file:
            raise ValueError("stars_file is required for real-data missions")
        return self


def load_config(path: Path) -> tuple[ExperimentConfig, str]:
    raw = Path(path).read_bytes()
    return ExperimentConfig.model_validate(yaml.safe_load(raw)), hashlib.sha256(raw).hexdigest()
