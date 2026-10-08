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
from .data import LightCurve, load_mission, read_stars, synthetic, synthetic_variable
from .detrend import clean, flatten
from .evaluate import STATUSES, completeness, fit_gamma_cdf, match
from .inject import Injection, Star, sample_injection, transit_model
from .search import SEARCHES, Candidate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Seed-stream index for generating synthetic light curves; trials use 0..trials_per_star-1.
LIGHTCURVE_STREAM = 2**31 - 1
PACKAGES = ["spacesift", "numpy", "scipy", "pandas", "astropy", "lightkurve", "wotan", "pydantic"]
# The packages whose version can change a trial's numbers (injection, detrending, search).
RESULT_PACKAGES = {"spacesift", "numpy", "scipy", "astropy", "wotan"}


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
    if cfg.grid:
        spc = cfg.grid.stars_per_cell
        stars = [Star(star_id=f"G{ci:03d}-{j:02d}") for ci in range(len(cfg.grid.cells())) for j in range(spc)]
    elif cfg.mission == "synthetic":
        stars = [Star(star_id=f"SYN-{i:04d}") for i in range(cfg.synthetic.n_stars)]
    else:
        stars = read_stars(PROJECT_ROOT / cfg.stars_file)
    return stars[: cfg.max_stars] if cfg.max_stars else stars


def cell_for(cfg: ExperimentConfig, star_index: int) -> dict:
    """The variability cell a grid star belongs to ({} outside a grid)."""
    if not cfg.grid:
        return {}
    ci = star_index // cfg.grid.stars_per_cell
    return {"cell": ci, **cfg.grid.cells()[ci]}


def load_star(cfg: ExperimentConfig, star: Star, star_index: int) -> LightCurve:
    if cfg.mission == "synthetic":
        s = cfg.synthetic
        rng = np.random.default_rng(trial_seed(cfg.seed, star_index, LIGHTCURVE_STREAM))
        if cfg.grid:
            cell = cell_for(cfg, star_index)
            return synthetic_variable(rng, baseline=s.baseline_d,
                                      noise_ppm=float(np.exp(rng.uniform(*np.log(s.noise_ppm)))),
                                      kind=cell["var_kind"], period_d=cell["var_period_d"],
                                      amp_ppm=cell["var_amp_ppm"])
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


def depth_metrics(t: np.ndarray, f_flat: np.ndarray, model: np.ndarray, inj: Injection) -> dict:
    """How much transit survives detrending, measured without any search.

    model_mean_depth: mean of (1 - model) over in-transit cadences (the injected signal).
    detrended_depth_ratio: the same mean measured on the detrended flux, over the model's.
    A ratio below 1 means detrending removed part of the transit; noise averages out
    over many trials. BLS box bias is then found_depth / model_mean_depth.
    """
    phase = np.abs(((t - inj.t0 + 0.5 * inj.period) % inj.period) - 0.5 * inj.period)
    inside = phase < inj.duration / 2
    if not inside.any():
        return {"model_mean_depth": np.nan, "detrended_depth_ratio": np.nan}
    model_mean = float(np.mean(1 - model[inside]))
    return {"model_mean_depth": model_mean,
            "detrended_depth_ratio": float(np.mean(1 - f_flat[inside]) / model_mean)}


def run_trial(cfg: ExperimentConfig, lc: LightCurve, star: Star, inj: Injection,
              spec: str, star_period: float | None = None) -> tuple[str, Candidate | None, dict]:
    if inj.n_transits < cfg.recovery.min_transits:
        return "not_observable", None, {}
    model = transit_model(lc.time, inj, star.u1, star.u2, lc.exptime, cfg.injection.supersample)
    t, f = clean(lc.time, flatten(lc.time, lc.flux * model, spec))
    metrics = depth_metrics(t, f, model[np.isin(lc.time, t)], inj)
    cand = run_search(cfg, t, f)
    r = cfg.recovery
    status = match(inj, cand, sde_threshold=r.sde_threshold, period_tol=r.period_tol, min_transits=r.min_transits,
                   star_period=star_period)
    return status, cand, metrics


def baseline_rows(cfg: ExperimentConfig, lc: LightCurve, star: Star, star_index: int) -> list[dict]:
    """Per detrending method, search the star with nothing injected: plain (its own best
    signal, used to label 'star_signal') and inverted (a false-alarm sample)."""
    from .null import invert

    rows = []
    for spec in cfg.detrend:
        t, f = clean(lc.time, flatten(lc.time, lc.flux, spec))
        for kind, flux in (("plain", f), ("inverted", invert(f))):
            cand = run_search(cfg, t, flux)
            rows.append({"star_id": star.star_id, "star_index": star_index, "detrend": spec,
                         "baseline": kind, **cell_for(cfg, star_index), **cand.to_dict()})
    return rows


def null_rows(cfg: ExperimentConfig, lc: LightCurve, star: Star, star_index: int) -> list[dict]:
    """Search every null variant of the star's detrended light curve; one row per search."""
    from .null import null_variants

    rows = []
    for spec in cfg.detrend:
        t, f = clean(lc.time, flatten(lc.time, lc.flux, spec))
        rng_for = lambda k: np.random.default_rng(trial_seed(cfg.seed, star_index, k))
        for kind, k, tv, fv in null_variants(t, f, cfg.false_alarm, rng_for):
            cand = run_search(cfg, tv, fv)
            rows.append({"star_id": star.star_id, "star_index": star_index, "trial": k, "detrend": spec,
                         "search": cfg.search.method, "status": "null", "null_kind": kind, **cand.to_dict()})
    return rows


def run_star(cfg: ExperimentConfig, star: Star, star_index: int) -> tuple[list[dict], dict, list[dict]]:
    """All trials for one star -> (trial rows, star row, baseline rows). Any failure drops
    the whole star (no partial rows) and is recorded in stars.parquet, so one bad star
    cannot end a long run."""
    try:
        return _run_star(cfg, star, star_index)
    except Exception as exc:
        return [], {"star_id": star.star_id, "error": repr(exc)}, []


def _run_star(cfg: ExperimentConfig, star: Star, star_index: int) -> tuple[list[dict], dict, list[dict]]:
    started = clock.perf_counter()
    lc = load_star(cfg, star, star_index)
    t_flat, f_flat = clean(lc.time, flatten(lc.time, lc.flux, cfg.injection.noise_detrend))
    if cfg.injection.snr_noise == "white":
        t_noise, f_noise = lc.time, lc.noise_only
    else:
        t_noise, f_noise = t_flat, f_flat
    cell = cell_for(cfg, star_index)

    # Pre-injection search: a strong signal here means the star may host a real
    # (or instrumental) periodic signal and should be excluded from analysis.
    pre = run_search(cfg, t_flat, f_flat)
    star_row = {"star_id": star.star_id, "n_points": lc.time.size,
                "baseline_d": float(lc.time.max() - lc.time.min()),
                "pre_sde": pre.sde, "pre_period": pre.period, "error": None, **cell}

    if cfg.kind == "false_alarm":
        rows = null_rows(cfg, lc, star, star_index)
        star_row["seconds"] = clock.perf_counter() - started
        return rows, star_row, []

    baselines = baseline_rows(cfg, lc, star, star_index) if cfg.baseline_searches else []
    star_period = {b["detrend"]: b["found_period"] for b in baselines if b["baseline"] == "plain"}

    rows = []
    for j in range(cfg.injection.trials_per_star):
        seed = trial_seed(cfg.seed, star_index, j)
        rng = np.random.default_rng(seed)
        inj = sample_injection(rng, t_noise, f_noise, star, cfg.injection.expected_snr,
                               cfg.injection.period_d, cfg.injection.b, cfg.injection.depth_basis)
        for spec in cfg.detrend:
            status, cand, metrics = run_trial(cfg, lc, star, inj, spec, star_period.get(spec))
            rows.append({
                "star_id": star.star_id, "star_index": star_index, "trial": j, "trial_seed": seed,
                "detrend": spec, "search": cfg.search.method, "status": status, **cell,
                **inj.to_dict(), **(cand.to_dict() if cand else {}), **metrics,
            })
    star_row["seconds"] = clock.perf_counter() - started
    return rows, star_row, baselines


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
    trials = pd.DataFrame([r for star_rows, _, _ in results for r in star_rows])
    star_table = pd.DataFrame([s for _, s, _ in results])
    baselines = pd.DataFrame([b for _, _, base in results for b in base])
    for s in star_table[star_table["error"].notna()].itertuples():
        print(f"  star {s.star_id} failed: {s.error}")
    trials.to_parquet(out / "trials.parquet")
    star_table.to_parquet(out / "stars.parquet")
    if len(baselines):
        baselines.to_parquet(out / "baselines.parquet")

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
        write_record(exp, provenance, trials, star_table, baselines)
    print(f"wrote {out}")
    return out


def merge_shards(exp: Path) -> Path:
    """Combine shards/shard-*/ into the experiment's final record. Refuses mixed provenance."""
    # Search recursively: downloaded CI artifacts may nest the shard folders.
    by_index = {}
    for p in sorted((exp / "shards").rglob("shard.json")):
        meta = json.loads(p.read_text())
        by_index[meta["shard"][0]] = (p.parent, meta)
    if not by_index:
        raise SystemExit(f"no shard.json found under {exp / 'shards'}")
    metas = [m for _, m in by_index.values()]
    n = metas[0]["shard"][1]
    found = sorted(by_index)
    if found != list(range(n)):
        raise SystemExit(f"expected shards 0..{n - 1}, found {found}")
    for key in ("config_sha256", "git_commit", "git_dirty", "preprocessing_version"):
        if len({json.dumps(m[key], sort_keys=True) for m in metas}) > 1:
            raise SystemExit(f"shards disagree on {key}; they were not run from the same code/config")
    # Packages that compute results must match exactly; others (config parsing, plotting)
    # may differ, e.g. when one shard is re-run after a release, but are recorded per shard.
    for pkg in sorted({p for m in metas for p in m["package_versions"]}):
        versions = {m["package_versions"].get(pkg) for m in metas}
        if len(versions) > 1:
            if pkg in RESULT_PACKAGES:
                raise SystemExit(f"shards ran with different {pkg} versions {sorted(map(str, versions))}; "
                                 "results are not comparable")
            print(f"  note: shards differ in {pkg} {sorted(map(str, versions))} (does not affect results)")

    dirs = [by_index[i][0] for i in range(n)]
    metas = [by_index[i][1] for i in range(n)]
    trials = pd.concat([pd.read_parquet(d / "trials.parquet") for d in dirs], ignore_index=True)
    stars = pd.concat([pd.read_parquet(d / "stars.parquet") for d in dirs], ignore_index=True)
    trials = trials.sort_values(["star_index", "trial", "detrend"], ignore_index=True)
    base_files = [d / "baselines.parquet" for d in dirs if (d / "baselines.parquet").exists()]
    baselines = pd.concat([pd.read_parquet(p) for p in base_files], ignore_index=True) if base_files else None
    if baselines is not None:
        baselines = baselines.sort_values(["star_index", "detrend", "baseline"], ignore_index=True)
    provenance = {k: v for k, v in metas[0].items() if k != "shard"}
    if len({json.dumps(m["package_versions"], sort_keys=True) for m in metas}) > 1:
        provenance["package_versions_by_shard"] = {str(i): m["package_versions"] for i, m in enumerate(metas)}
    provenance["started"] = min(m["started"] for m in metas)
    provenance["finished"] = max(m["finished"] for m in metas)
    provenance["n_shards"] = n
    write_record(exp, provenance, trials, stars, baselines)
    print(f"merged {n} shards into {exp}")
    return exp


def write_record(exp: Path, provenance: dict, trials: pd.DataFrame, star_table: pd.DataFrame,
                 baselines: pd.DataFrame | None = None) -> None:
    trials.to_parquet(exp / "trials.parquet")
    star_table.to_parquet(exp / "stars.parquet")
    if baselines is not None and len(baselines):
        baselines.to_parquet(exp / "baselines.parquet")
    is_null = "null_kind" in trials.columns
    record = {
        **provenance,
        "n_stars": int(len(star_table)),
        "n_stars_failed": int(star_table["error"].notna().sum()) if len(star_table) else 0,
        "n_trials": int(len(trials)),
        "summary": (summarize_null(trials) if is_null else summarize(trials)) if len(trials) else {},
    }
    (exp / "record.json").write_text(json.dumps(record, indent=2))
    if len(trials):
        from .plots import plot_experiment, plot_null
        (plot_null if is_null else plot_experiment)(trials, exp / "plots")


FALSE_ALARM_RATES = (0.01, 0.005, 0.001)


def summarize_null(trials: pd.DataFrame) -> dict:
    """SDE distribution of null searches and the threshold that gives each false-alarm rate.

    False-alarm rate = fraction of noise-only searches whose top peak beats the threshold.
    """
    out = {}
    for spec, d in trials.groupby("detrend"):
        sde = d.found_sde.to_numpy()
        out[spec] = {
            "n_null_searches": int(len(d)),
            "sde_quantiles_by_kind": {k: g.found_sde.quantile([.5, .9, .99]).round(3).tolist()
                                      for k, g in d.groupby("null_kind")},
            "threshold_for_far": {str(far): float(np.quantile(sde, 1 - far)) for far in FALSE_ALARM_RATES},
            "far_at_sde_7": float((sde > 7).mean()),
        }
    return out


def summarize(trials: pd.DataFrame) -> dict:
    out = {}
    for spec, d in trials.groupby("detrend"):
        obs = d[d.status != "not_observable"]
        out[spec] = {
            "status_counts": {s: int((d.status == s).sum()) for s in STATUSES},
            "gamma_cdf_fit": fit_gamma_cdf(obs.inj_expected_snr, obs.status == "recovered"),
            "completeness_vs_snr": completeness(d).round(4).to_dict("records"),
        }
        if "detrended_depth_ratio" in d.columns:
            rec = d[d.status == "recovered"]
            out[spec]["depth"] = {
                "median_detrended_depth_ratio": float(obs.detrended_depth_ratio.median()),
                "median_found_over_injected_central_depth": float((rec.found_depth / rec.inj_depth).median()),
                "median_found_over_model_mean_depth": float((rec.found_depth / rec.model_mean_depth).median()),
                "median_model_mean_over_central_depth": float((obs.model_mean_depth / obs.inj_depth).median()),
            }
    return out


