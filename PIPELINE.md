# NEOCP Vetting Pipeline — status & runbook

Built 2026-07-10. Complements `Planetary Defense v1 — Project Plan.md` (v1 plan);
this file reflects what is actually implemented and the constraints discovered.

## The one-sentence architecture

Because MPC never publishes observations for objects that leave the NEOCP
unconfirmed, the labeled training set **cannot be backfilled** — it is captured
prospectively: every run snapshots the feed, saves each new object's tracklet
while it is still public, and joins outcomes from the Previous-NEOCP page as
objects resolve.

## What runs where

| Piece | Where | Why |
|---|---|---|
| `src/capture_neocp.py` | your machine (or any cron box) | direct HTTP to MPC |
| `neocp-capture` scheduled task | Cowork app, 08:00 & 20:00 daily | same capture via Claude's web tools (sandbox can't reach MPC directly) |
| `src/features.py` + digest2 | Claude sandbox or your machine | needs the digest2 binary (build once, see below) |
| `src/train.py`, `src/predict.py` | anywhere with sklearn/lightgbm | |
| `src/fetch_css_bulk.py` | **your machine only** | Zooniverse unreachable from the sandbox |
| `src/train_cnn.py` | your machine (or sandbox once torch present) | |

## Daily flow (automated)

1. Scheduled task captures: snapshot JSON → `data/capture/snapshots/`, new
   tracklets → `data/capture/tracklets/<desig>.obs80`, outcomes →
   `data/capture/outcomes.csv` (via `prev_des_raw_*.txt` + `merge_prev_des.py`).
2. Periodically (interactive session or locally):
   `python3 src/features.py <digest2-dir>` → `python3 src/dataset.py` →
   `python3 src/train.py` → `python3 src/predict.py <digest2-dir>` which writes
   `outputs/daily_vetting.csv` — today's NEOCP ranked by P(real NEO).

## Current dataset status (2026-07-13)

- 621 outcomes archived (May 16 – Jul 13).
- 41 objects captured with tracklets (Jul 10 + Jul 13 batches) → full 26
  digest2 sub-scores + 12 tracklet/geometry features. digest2 now lives
  **inside the repo** at `tools/digest2-linux/` (Linux binary + model).
- **42 labeled training rows** (34 NEO / 8 non-NEO): 19 from capture (11 of
  Thursday's objects resolved within 3 days — capture works as designed),
  23 NEO positives backfilled from MPEC circulars (discovery-night-trimmed).
- Backfill guard in train.py verified: it refused the MPEC positives until
  the capture supplied digest2-featured negatives, then merged cleanly.
- Remaining bottleneck is NEGATIVES (8 so far). They accrue only via capture
  (~10–20/day). Meaningful benchmark run at roughly 100+ negatives: ~1-2 weeks,
  faster if `backfill_designated.py` is run locally (adds ~154 MBA negatives
  + ~260 more NEO positives in one go).

### First real training run (2026-07-16, n=390)

After running `backfill_designated.py` locally (393 objects fetched): 390
labeled rows (250 NEO / 140 non-NEO), 367 with full digest2 sub-scores.
LightGBM, time-based split (75/25 by discovery date), test n=98:

| | ROC-AUC | non-NEO excluded @100% NEO recall |
|---|---|---|
| digest2 feed-score baseline | 0.52 | 2.9% |
| LightGBM (26 sub-scores + geometry) | 0.78 | 20% |

Honest caveats: (1) the feed-score baseline is handicapped here — backfilled
rows have no feed score (imputed 50), so its true operating performance is
better than 0.52 suggests; the sub-score model's advantage is real but this
comparison flatters it. (2) Negatives are mostly designated MBAs; the
junk-class (artifacts/satellites) is still underrepresented (only from
capture). (3) 20% exclusion is far from the 80% benchmark — expected at
1/50th of the paper's dataset size. Watch it climb as capture accrues.

Backfill quirks handled: precovery contamination (published astrometry
includes decades-old archival obs; discovery-tracklet trimming now windows
to ±120d of the outcome date — without this, 69/375 rows had wrong-epoch
tracklets) and digest2 runtime (~15s per 40 discovery tracklets; sandbox
requires chunked runs, see D2_PRECOMPUTED env hook in build_backfill.py).

## Environment gotchas (hard-won, do not rediscover)

- **MPC CDN staleness**: `neocp.txt` was 12 days stale, `neocp.json` 3 days,
  HTML live. Always append `?nocache=<unixtime>`. (The JSON feed is now VALID
  JSON — Appendix A.1 of the v1 plan is outdated on that point.)
- **Tracklet CGI**: `https://cgi.minorplanetcenter.net/cgi-bin/showobsorbs.cgi?Obj=<desig>&obs=y`
  (the `www.` host returns empty). Lines come back WITHOUT the 5-space 80-col
  prefix (75 chars); prepend 5 spaces before feeding digest2. Returns empty
  once an object is removed — capture or lose it.
- **digest2 build**: `git clone https://github.com/Smithsonian/digest2` — the
  Makefile needs libxml2 headers only for ADES XML input; if unavailable,
  strip the `#include <libxml/...>` lines from digest2.h, remove d2ades.c from
  the Makefile SRC and `-lxml2` from LDFLAGS (we only feed 80-col). Use the
  model files from the repo's `population/` dir, NOT the stale ones in
  `digest2/`: `./digest2 -m population/digest2.model -o population/digest2.obscodes -c <config>`.
  Config for the 26 A&A features: `headings/rms/raw/noid` + the 13 class names.
- **Labels**: designated-with-MPEC ⇒ NEO; designated-without-MPEC ⇒ almost
  always an identification with a known non-NEO (treat as 0, verify vs MPCORB
  later); "was not confirmed" contains some lost real NEOs (irreducible label
  noise, shared with the A&A paper). Removal reasons are free-form text —
  `merge_prev_des.py` prints UNPARSED lines; watch for new phrasings.
- **Cowork sandbox network**: only package registries + github.com. MPC/
  Zooniverse/IRSA/Wayback all unreachable (Wayback is blocklisted for
  Claude's fetch tool too — historical snapshots are NOT an option).

## Image side (CSS / Zooniverse)

- `src/fetch_css_bulk.py --max-subjects 2000` (local machine, ~30–60 min,
  1–2 GB) → `data/css_zooniverse/subject_*/` + `css_subjects.csv` index.
- `src/motion_baseline.py` — no-ML physics baseline (blob tracking + linear-
  motion consistency). Smoke-tested on the 2 sample subjects: the labeled
  artifact correctly scores p_real=0.04. This is the number the CNN must beat.
- `src/train_cnn.py` — 4-frames-as-channels small CNN (~300k params, CPU-ok)
  reporting the controlled comparison digest2-only vs image-only vs fused on
  the same test split. Needs a few hundred labeled subjects.
- Green-circle overlay is masked in `css_dataset.py` (leakage hygiene).

## Next milestones

1. (auto) capture accumulates; check `outcomes.csv` growth weekly
2. run `fetch_css_bulk.py` locally → train CNN + baseline comparison
3. at ~500+ labeled tracklet rows: first honest time-split benchmark run
4. verify designated-other labels against MPCORB; decide comet handling
5. Rubin-era stress test: reweight test priors to ~8% NEO fraction
