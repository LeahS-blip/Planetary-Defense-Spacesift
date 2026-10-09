# SpaceSift

An open computational laboratory for **exoplanet transit detectability**. The
core output is not a classifier score but a *completeness map*: the fraction of
injected transits a pipeline recovers, as a function of planet, star,
observing, and algorithm parameters. Plan and rationale: the SpaceSift Roadmap doc.

Current scope: phase 2 of the roadmap. This is the injection-recovery harness and the
BLS baseline for **SS-0001 (harness validation on Kepler)**.

## Setup

```
pip install -e ".[dev]"
pytest
```

Python 3.10+. `batman-package` and `transitleastsquares` are optional extras
because they need a C compiler where no prebuilt wheel exists (e.g. Python
3.14 on Windows). SpaceSift ships its own transit model (below), so neither is required.

## The UI

```
python -m spacesift.cli ui
```

This opens <http://127.0.0.1:8765/> in your browser. Stop it with Ctrl+C.

- **Explore:** choose a simulated star (quiet, pulsating, or spotted) or a real Kepler star by KIC
  number. Plant a planet, sized by signal-to-noise or by radius, pick a cleaning method, and run
  the search. The page shows whether the planet was found, plus four plots: raw brightness with
  the planted dips, the cleaned light curve, the search periodogram with its threshold, and the
  light curve folded on the period found. The presets reproduce the main SS-0002 results.
- **Results:** browse every experiment in `experiments/`: its record, analyses, notes and plots.
  Use `--experiments <folder> [<folder> ...]` to browse other folders.

## Running an experiment

```
spacesift run configs/SS-0000.yaml --dirty      # synthetic smoke test, no downloads
spacesift select-stars --n 200 --seed 1          # freeze a Kepler sample (already done)
spacesift run configs/SS-0001.yaml --max-stars 2 --dirty   # timing run
spacesift run configs/SS-0001.yaml               # the real thing, from a clean commit
spacesift replay experiments/SS-0001 --star 1868918 --trial 3   # re-run + plot one trial
```

### In the cloud (GitHub Actions)

On GitHub: **Actions → run-experiment → Run workflow**, then pick a config. The run
is split into 10 parallel shards (`spacesift run ... --shard i/10`), and a merge
job (`spacesift merge experiments/<id>`) combines them. The merge job commits the
finished `experiments/<id>/` back to the repo and uploads it as a workflow artifact.
Star indices are global, so a sharded run gives exactly the same trials as an
unsharded one (this is tested).

Each experiment writes `experiments/<id>/`:

| File | Contents |
| --- | --- |
| `config.yaml` | Exact config used |
| `record.json` | Config hash, git commit and dirty flag, package versions, seed, timings, per-detrending status counts, gamma-CDF fit, binned completeness |
| `trials.parquet` | One row per trial and detrending choice: every injected parameter (`inj_*`), the search result (`found_*`), status, and the trial's own seed |
| `stars.parquet` | Per star: points, baseline, pre-injection search SDE (screen out stars with real signals), runtime, errors |
| `plots/` | Completeness vs SNR with Wilson intervals and gamma-CDF fit; completeness over radius x period |

`run` refuses to start with uncommitted changes to `spacesift/` or `configs/`
unless you pass `--dirty`, and the record then says `git_dirty: true`. It also
refuses to overwrite a finished experiment.

## How a trial works

1. Load the star's light curve: Kepler PDCSAP long cadence via lightkurve, normalised per quarter and cached in `cache/`.
2. Measure noise once per star on a light curve detrended with `injection.noise_detrend`, so the SNR scale doesn't change with the detrending under test.
3. Draw P (log-uniform), b, t0, and a **target expected SNR**. Solve for the planet radius that gives that SNR on this star: depth = SNR · σ_CDPP(T14) / √N_transits.
4. Inject the limb-darkened transit (supersampled over the 29.4-min exposure) into the raw flux.
5. Detrend (each `detrend` spec in turn), clip upward outliers, bin to `search.bin_min`, and run BLS.
6. Classify: `recovered`, `alias_half`, `alias_double`, `wrong_period`, `below_threshold`, or `not_observable` (fewer than `min_transits` transits in the data).

Trial seeds are derived from `(seed, star_index, trial)`, so any single trial can be replayed on its own.

## Departures from the roadmap, and known limitations

- **Transit model.** Instead of batman, it numerically integrates the quadratically limb-darkened stellar disk over the planet's shadow (`spacesift/inject.py`). It is tested against the analytic uniform-disk overlap. It assumes circular orbits only.
- **Limb darkening.** Fixed at u1 = 0.40, u2 = 0.25 (roughly Sun-like, Kepler band). Claret-table coefficients per Teff/logg are a to-do.
- **Injection level.** Transits are injected into PDCSAP light curves, after the pipeline's systematics correction. Christiansen et al. inject at pixel level, so measured completeness here is likely optimistic. State this in any write-up.
- **Period grid.** Adaptive: the step keeps phase drift below duration / oversample, using the shortest plausible duration at each period (Sun-density star, b = 0.9). That's about 54k periods for 1 year of data, 1–30 d. For M dwarfs, lower the factor in `search.shortest_duration`.
- **SDE depends on the grid.** The noise-floor SDE changes with the period grid and binning. Re-check the threshold (default 7) against the pre-injection SDE distribution in `stars.parquet` whenever those settings change.
- **TLS and the CNN / gradient-boosting vetters** (SS-0003) are not implemented yet. New searches register in `search.SEARCHES`.

## Cost

Measured on Kepler Q2–Q5 (about 1 year, ~17k cadences binned to 1 h): about 16 s per BLS search on one core. So SS-0001 at 200 stars × 50 trials comes to about 45 CPU-hours, or roughly 11 h with 4 workers. Using all 17 quarters roughly quadruples the grid.
