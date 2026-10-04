"""Experiment runner: stars x trials x detrending choices -> trials.parquet + record.json."""

from __future__ import annotations

import json
import shutil
import subprocess
import time as clock
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import PREPROCESSING_VERSION, __version__
from .config import ExperimentConfig, load_config
from .data import LightCurve, load_mission, read_stars, synthetic
from .detrend import clean, flatten
from .evaluate import STATUSES, completeness, fit_gamma_cdf, match
from .inject import Injection, Star, inject, sample_injection
from .search import SEARCHES, Candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Seed-stream index for generating synthetic light curves; trials use 0..trials_per_star-1.
LIGHTCURVE_STREAM = 2**31 - 1
PACKAGES = ["spacesift", "numpy", "scipy", "pandas", "astropy", "lightkurve", "wotan", "pydantic"]


# ------------------------------------------------------------- provenance

def git_state() -> tuple[str | None, bool]:
    """(commit, dirty). No repository counts as dirty."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "spacesift", "configs"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, True


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def trial_seed(master: int, star_index: int, trial: int) -> int:
    return int(np.random.SeedSequence([master, star_index, trial]).generate_state(1)[0])


# ------------------------------------------------------------------ stars

def build_stars(cfg: ExperimentConfig) -> list[Star]:
    if cfg.mission == "synthetic":
        stars = [Star(star_id=f"SYN-{i:04d}") for i in range(cfg.synthetic.n_stars)]
    else:
        stars = read_stars(PROJECT_ROOT / cfg.stars_file)
    return stars[: cfg.max_stars] if cfg.max_stars else stars


def load_star(cfg: ExperimentConfig, star: Star, star_index: int) -> LightCurve:
    if cfg.mission == "synthetic":
        s = cfg.synthetic
        rng = np.random.default_rng(trial_seed(cfg.seed, star_index, LIGHTCURVE_STREAM))
        return synthetic(
            rng,
            baseline=s.baseline_d,
            noise_ppm=float(np.exp(rng.uniform(*np.log(s.noise_ppm)))),
            var_amp_ppm=float(rng.uniform(*s.var_amp_ppm)),
            var_period=float(rng.uniform(*s.var_period_d)),
        )
    return load_mission(star.star_id, cfg.mission, cfg.quarters, PROJECT_ROOT / cfg.cache_dir)


# ----------------------------------------------------------------- trials

def run_search(cfg: ExperimentConfig, time: np.ndarray, flux_flat: np.ndarray) -> Candidate:
    s = cfg.search
    return SEARCHES[s.method](
        time, flux_flat,
        period_range=cfg.injection.period_d,
        durations=np.asarray(s.durations_hr) / 24,
        oversample=s.oversample,
        bin_width=s.bin_min / 1440,
    )


def run_trial(cfg: ExperimentConfig, lc: LightCurve, star: Star, inj: Injection, spec: str) -> tuple[str, Candidate | None]:
    if inj.n_transits < cfg.recovery.min_transits:
        return "not_observable", None
    flux = inject(lc.time, lc.flux, inj, star, lc.exptime, cfg.injection.supersample)
    t, f = clean(lc.time, flatten(lc.time, flux, spec))
    cand = run_search(cfg, t, f)
    r = cfg.recovery
    return match(inj, cand, sde_threshold=r.sde_threshold, period_tol=r.period_tol,
                 min_transits=r.min_transits), cand


def run_star(cfg: ExperimentConfig, star: Star, star_index: int) -> tuple[list[dict], dict]:
    started = clock.perf_counter()
    try:
        lc = load_star(cfg, star, star_index)
    except Exception as exc:  # a missing star is logged, not fatal
        return [], {"star_id": star.star_id, "error": repr(exc)}

    noise_flat = flatten(lc.time, lc.flux, cfg.injection.noise_detrend)
    t_noise, f_noise = clean(lc.time, noise_flat)

    # Pre-injection search: a strong signal here means the star may host a real
    # (or instrumental) periodic signal and should be excluded from analysis.
    pre = run_search(cfg, t_noise, f_noise)
    star_row = {"star_id": star.star_id, "n_points": lc.time.size,
                "baseline_d": float(lc.time.max() - lc.time.min()),
                "pre_sde": pre.sde, "pre_period": pre.period, "error": None}

    rows = []
    for j in range(cfg.injection.trials_per_star):
        seed = trial_seed(cfg.seed, star_index, j)
        rng = np.random.default_rng(seed)
        inj = sample_injection(rng, t_noise, f_noise, star, cfg.injection.expected_snr,
                               cfg.injection.period_d, cfg.injection.b)
        for spec in cfg.detrend:
            status, cand = run_trial(cfg, lc, star, inj, spec)
            rows.append({
                "star_id": star.star_id, "star_index": star_index, "trial": j, "trial_seed": seed,
                "detrend": spec, "search": cfg.search.method, "status": status,
                **inj.to_dict(), **(cand.to_dict() if cand else {}),
            })
    star_row["seconds"] = clock.perf_counter() - started
    return rows, star_row


# ------------------------------------------------------------- experiment

def run_experiment(config_path: Path, *, allow_dirty: bool = False, out_root: Path | None = None,
                   max_stars: int | None = None, n_jobs: int | None = None,
                   shard: tuple[int, int] | None = None) -> Path:
    """Run every star (or one shard of them) and write results.

    With shard=(i, n), runs stars i, i+n, i+2n, ... into <out>/<id>/shards/shard-i;
    star indices stay global, so seeds match an unsharded run. Combine with merge_shards.
    """
    cfg, cfg_sha = load_config(config_path)
    if max_stars is not None:
        cfg.max_stars = max_stars
    if n_jobs is not None:
        cfg.n_jobs = n_jobs
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise SystemExit("Refusing to run: code or configs have uncommitted changes (or no git repo). "
                         "Commit first, or pass --dirty; the record will say so.")

    exp = (out_root or PROJECT_ROOT / "experiments") / cfg.id
    if (exp / "record.json").exists():
        raise SystemExit(f"{exp} already holds a finished experiment; bump the id or delete it.")
    out = exp / "shards" / f"shard-{shard[0]}" if shard else exp
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, exp / "config.yaml")

    indexed = list(enumerate(build_stars(cfg)))
    if shard:
        indexed = indexed[shard[0]::shard[1]]
    started = datetime.now(timezone.utc)
    label = f" (shard {shard[0]}/{shard[1]})" if shard else ""
    print(f"{cfg.id}{label}: {len(indexed)} stars x {cfg.injection.trials_per_star} trials "
          f"x {len(cfg.detrend)} detrend")
    results = Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(run_star)(cfg, s, i) for i, s in indexed)
    trials = pd.DataFrame([r for star_rows, _ in results for r in star_rows])
    star_table = pd.DataFrame([s for _, s in results])
    for s in star_table[star_table["error"].notna()].itertuples():
        print(f"  star {s.star_id} failed: {s.error}")
    trials.to_parquet(out / "trials.parquet")
    star_table.to_parquet(out / "stars.parquet")

    provenance = {
        "id": cfg.id,
        "question": cfg.question,
        "config_file": str(config_path),
        "config_sha256": cfg_sha,
        "config": cfg.model_dump(mode="json"),
        "git_commit": commit,
        "git_dirty": dirty,
        "spacesift_version": __version__,
        "preprocessing_version": PREPROCESSING_VERSION,
        "package_versions": package_versions(),
        "seed": cfg.seed,
        "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    if shard:
        provenance["shard"] = list(shard)
        (out / "shard.json").write_text(json.dumps(provenance, indent=2))
    else:
        write_record(exp, provenance, trials, star_table)
    print(f"wrote {out}")
    return out


def merge_shards(exp: Path) -> Path:
    """Combine shards/shard-*/ into the experiment's final record. Refuses mixed provenance."""
    metas = [json.loads(p.read_text()) for p in sorted((exp / "shards").glob("shard-*/shard.json"))]
    if not metas:
        raise SystemExit(f"no shards under {exp / 'shards'}")
    n = metas[0]["shard"][1]
    found = sorted(m["shard"][0] for m in metas)
    if found != list(range(n)):
        raise SystemExit(f"expected shards 0..{n - 1}, found {found}")
    for key in ("config_sha256", "git_commit", "git_dirty", "package_versions", "preprocessing_version"):
        if len({json.dumps(m[key], sort_keys=True) for m in metas}) > 1:
            raise SystemExit(f"shards disagree on {key}; they were not run from the same code/config")

    dirs = [exp / "shards" / f"shard-{i}" for i in range(n)]
    trials = pd.concat([pd.read_parquet(d / "trials.parquet") for d in dirs], ignore_index=True)
    stars = pd.concat([pd.read_parquet(d / "stars.parquet") for d in dirs], ignore_index=True)
    trials = trials.sort_values(["star_index", "trial", "detrend"], ignore_index=True)
    provenance = {k: v for k, v in metas[0].items() if k != "shard"}
    provenance["started"] = min(m["started"] for m in metas)
    provenance["finished"] = max(m["finished"] for m in metas)
    provenance["n_shards"] = n
    write_record(exp, provenance, trials, stars)
    print(f"merged {n} shards into {exp}")
    return exp


def write_record(exp: Path, provenance: dict, trials: pd.DataFrame, star_table: pd.DataFrame) -> None:
    trials.to_parquet(exp / "trials.parquet")
    star_table.to_parquet(exp / "stars.parquet")
    record = {
        **provenance,
        "n_stars": int(len(star_table)),
        "n_stars_failed": int(star_table["error"].notna().sum()) if len(star_table) else 0,
        "n_trials": int(len(trials)),
        "summary": summarize(trials) if len(trials) else {},
    }
    (exp / "record.json").write_text(json.dumps(record, indent=2))
    if len(trials):
        from .plots import plot_experiment
        plot_experiment(trials, exp / "plots")


def summarize(trials: pd.DataFrame) -> dict:
    out = {}
    for spec, d in trials.groupby("detrend"):
        obs = d[d.status != "not_observable"]
        out[spec] = {
            "status_counts": {s: int((d.status == s).sum()) for s in STATUSES},
            "gamma_cdf_fit": fit_gamma_cdf(obs.inj_expected_snr, obs.status == "recovered"),
            "completeness_vs_snr": completeness(d).round(4).to_dict("records"),
        }
    return out


