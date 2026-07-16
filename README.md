# planetary-defense-neocp-vetting

Machine-learned vetting for the Minor Planet Center's [NEO Confirmation Page](https://www.minorplanetcenter.net/iau/NEO/toconfirm_tabular.html):
for every candidate awaiting confirmation, predict the probability it's a **real
near-Earth object** vs. a false positive (artifact, satellite, star, or main-belt
asteroid), so the volunteer follow-up network stops spending scarce telescope
time on junk. Only ~50% of NEOCP candidates are real NEOs today; in the
Rubin/LSST era that drops to ~8%.

**Benchmark to beat** (A&A 2025): exclude >80% of non-NEOs while losing ≤1% of
real NEOs. Current status and honest numbers: see [PIPELINE.md](PIPELINE.md).

## How it works

1. **Capture** (`src/capture_neocp.py`, runs 2×/day): snapshots the live NEOCP
   feed, saves each new object's astrometric tracklet *while it is still
   public* (MPC deletes them on removal — the dataset cannot be backfilled),
   and records outcomes from the Previous-NEOCP page. `data/capture/` is the
   accumulating labeled archive.
2. **Features** (`src/features.py`): the 26 digest2 sub-scores (13 orbit
   classes × raw/noid) + tracklet geometry the published benchmark ignored
   (sky-motion rate, ecliptic latitude, solar elongation, observatory code…).
3. **Train/evaluate** (`src/train.py --with-backfill`): LightGBM, time-based
   split, metrics reported at 99% NEO recall. Includes a guard against
   provenance leakage between captured and backfilled rows.
4. **Score & publish** (`src/predict.py`, `src/make_dashboard.py`): ranked
   `outputs/daily_vetting.csv` + a live dashboard (`outputs/dashboard.html`)
   with an extrapolated-position sky map.

Plus an image track (`src/fetch_css_bulk.py`, `src/motion_baseline.py`,
`src/train_cnn.py`): real-vs-artifact classification on Catalina Sky Survey
4-frame detection sequences from Zooniverse's Daily Minor Planet project.

## Setup

```
pip install numpy scikit-learn lightgbm pillow          # + torch for the CNN
git clone https://github.com/Smithsonian/digest2        # see PIPELINE.md for build notes
```

Full runbook, environment gotchas, and current results: [PIPELINE.md](PIPELINE.md).
Original research plan: `Planetary Defense v1 — Project Plan.md`.

Data credits: Minor Planet Center (NEOCP, MPECs, observations API), Catalina
Sky Survey / Zooniverse Daily Minor Planet. Please cite the MPC when using
their data and poll politely.
