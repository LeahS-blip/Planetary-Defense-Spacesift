# Planetary Defense — NEOCP False-Positive Vetting Classifier

**Project plan & research summary (v1)**
Author context: Miro Shverdin · Drafted 2026-07-04

---

## 1. Motivation & problem framing

### 1.1 The original idea and its hard limit
The starting goal was to *"examine NASA's feed to detect potential impacts of meteors not currently tracked by NASA."* A key reality check shaped the entire project:

- **NASA/JPL catalog feeds (NEO Feed API, Close Approach Data (CAD) API, Sentry) contain only already-detected, catalogued objects.** By definition they cannot surface objects that are "not tracked." Finding un-catalogued objects requires *discovery-stage* data, not the finished catalog.

### 1.2 Catalog completeness is worst exactly where it matters
- ~140 m+ objects (regional devastation class): the catalog is only **~40% complete** (below the Congressional 90% mandate).
- ~20 m **Chelyabinsk-class** objects: only a **few percent** are catalogued. Chelyabinsk (2013) itself was **completely undetected** until atmospheric entry (arrived from the sunward direction).

### 1.3 Where un-catalogued objects actually appear
The real discovery-stage sources:

| Source | Content | Un-catalogued? | Machine-readable |
|---|---|---|---|
| **MPC NEO Confirmation Page (NEOCP)** | Newly-detected objects awaiting orbit confirmation | ✅ Yes — core source | ✅ Yes (txt/JSON/API) |
| ATLAS / Pan-STARRS / Catalina survey nights | Raw transient detections pre-orbit | ✅ Yes | Partly |
| ZTF / Rubin (LSST) alert streams | Every moving/transient source per exposure | ✅ Yes | ✅ Yes (Avro/Kafka) |
| CNEOS Fireball & Bolide feed | Atmospheric impacts already detected | N/A (post-impact) | ✅ Yes (JSON) |
| JPL NEO / CAD / Sentry | Catalogued objects only | ❌ No | ✅ Yes |

### 1.4 The bottleneck: follow-up, not detection
Pipeline: **Surveys detect → objects land on NEOCP → someone must re-observe within ~24–72 h → MPC/JPL computes orbit → catalogued (or lost).**

The choke point is the **follow-up** step:
- NEOCP objects are often faint and fast-moving; without re-observation within hours–days, positional uncertainty balloons and the object is **lost**.
- Detection is about to massively outpace follow-up: **Rubin/LSST** will raise the NEO discovery rate by ~an order of magnitude, without a matching increase in follow-up capacity.
- Follow-up is done by a small, largely volunteer/small-observatory network — capacity-constrained by telescope time and human attention.

### 1.5 What already exists (do NOT reinvent)
**NEOfixer** (Catalina Sky Survey / University of Arizona) is an operational, funded follow-up **broker** that already ingests the MPC catalog + NEOCP + PCCP and outputs **site-specific ranked target lists** in real time. A generic "triage/ranking" tool would duplicate it.

- NEOfixer FAQ: https://neofixer.arizona.edu/faq
- PDC 2021 paper: https://ui.adsabs.harvard.edu/abs/2021plde.confE.206C/abstract

### 1.6 The open gap this project targets
NEOfixer ranks *what to observe*. It does **not** fully solve *which candidates are even worth ranking*:

- **Only ~50% of NEOCP candidates turn out to be real NEOs** — the rest are artifacts / non-NEOs that waste scarce follow-up time.
- **Rubin/LSST will make this far worse:** a 2024 study projects **only ~8.3%** of objects flagged for follow-up will be NEOs, the rest a flood of faint main-belt contaminants — and explicitly calls for better *prioritization/filtering strategies*.
  - arXiv 2408.12517: https://arxiv.org/html/2408.12517v1
- **Published benchmark to beat:** a 2025 A&A paper used ML on Digest2 parameters to exclude **>80% of non-NEOs while losing only ~1% of true NEOs** (~95% precision).
  - A&A 2025: https://www.aanda.org/articles/aa/full_html/2025/06/aa54311-25/aa54311-25.html

**This project = an open-source ML classifier that predicts, for each NEOCP object, real-NEO vs. false-positive**, so the follow-up network stops wasting time on junk. It complements NEOfixer (improves the input to ranking) rather than duplicating it. No telescope required — data + software only.

---

## 2. Data sources (all verified live 2026-07-04)

### 2.1 Live NEOCP feed (current candidates)
- **Plain text:** `https://www.minorplanetcenter.net/iau/NEO/neocp.txt`
- **JSON:** `https://www.minorplanetcenter.net/Extended_Files/neocp.json`

Fields per object (from live JSON): `Temp_Desig`, `Score` (Digest2, 0–100), `Discovery_year/month/day`, `R.A.`, `Decl.`, `V` (apparent mag), `Updated`, `NObs` (# observations), `Arc` (days), `H` (absolute mag → size proxy), `Not_Seen_dys` (staleness).

> NEOCP posting threshold: objects with a Digest2 score > 65 (or suspected comets) are posted. Only ~half are ultimately confirmed as NEOs.

### 2.2 Historical labels & archive (for training)
- **MPC NEOCP observations API:** `https://data.minorplanetcenter.net/api/get-obs-neocp` (POST `trksubs`, `output_format`)
  - Docs: https://docs.minorplanetcenter.net/mpc-ops-docs/apis/get-obs-neocp/
- **`neocp_els` table** (per-object discovery obs string + Digest2 + comet/artsat flags): https://www.minorplanetcenter.org/mpcops/documentation/neocp-els
- **Previous NEOCP + removal reasons** (the LABEL source — confirmed NEO vs. removed as non-NEO/artifact): https://www.minorplanetcenter.org/mpcops/documentation/neocp-obs-archive
- **Digest2 tool & sub-parameters** (recompute richer features than the single score): https://docs.minorplanetcenter.net/tutorials/notebooks/mpc_tutorial_digest2/

### 2.3 Supporting / context feeds (optional, for impact-relevance features)
- JPL CAD API: `https://ssd-api.jpl.nasa.gov/cad.api`
- NASA NeoWs Feed: `https://api.nasa.gov/neo/rest/v1/feed` (get a personal api_key; DEMO_KEY is rate-limited)
- CNEOS Fireball API: `https://ssd-api.jpl.nasa.gov/fireball.api`

---

## 3. Method

### 3.1 Task definition
Binary classification per NEOCP tracklet: **real NEO (1)** vs. **non-NEO / artifact (0)**, using only features available at (or shortly after) posting time. Output a calibrated probability so observers can set their own precision/recall trade-off.

### 3.2 Features
From the live feed and Digest2 archive:
- **Digest2 score** and, ideally, its **sub-parameters** across the 14 orbit classes (the single most informative signal per the A&A paper).
- **Arc length** (`Arc`, days) and **NObs** — short arc / few obs = poorly constrained.
- **Staleness** (`Not_Seen_dys`) — proxy for loss risk.
- **Apparent magnitude** `V` — feasibility / detectability.
- **Absolute magnitude** `H` → size proxy (impact relevance).
- **Sky position** (`R.A.`, `Decl.`) and derived **sky-plane motion rate** (fast movers behave differently).
- **Uncertainty-growth rate** (derived from successive feed pulls, if time-series captured).

### 3.3 Labels
Join NEOCP postings to the **Previous NEOCP / removal-reason archive**: objects that received a designation / were confirmed as NEOs → positive; objects removed as non-NEOs (main-belt, artsat, bad tracklet, artifact) → negative.

### 3.4 Models
Reproduce then try to beat the published benchmark:
1. **Baseline:** raw Digest2 score threshold (the status quo).
2. **Digest2-parameter filter** (reproduce the A&A rule-based >80% non-NEO exclusion).
3. **ML classifiers:** gradient-boosting (XGBoost/LightGBM), random forest, plus a small neural net; compare. Class imbalance handling + probability calibration.

### 3.5 Evaluation
- **Primary target (beat / match):** exclude >80% of non-NEOs while losing ≤~1% of true NEOs; precision ≈95%.
- Report precision/recall, PR-AUC, ROC-AUC, confusion matrix at operating points.
- **Time-based split** (train on earlier NEOCP history, test on later) — NOT random split — to simulate real deployment and avoid leakage.
- **Rubin-era stress test:** re-weight the test set toward the projected ~8% NEO fraction and report how the filter holds up.

---

## 4. Deliverables & repository structure

```
planetary-defense-neocp-vetting/
├── README.md                  # problem, benchmark, results table, how to run
├── LICENSE                    # MIT or Apache-2.0 (open source)
├── requirements.txt
├── sample_fetch_neocp.py      # WORKING: pull one live NEOCP frame, display as table
├── visualize_neocp.py         # WORKING: 4-panel visualization (sky map, scores, size, staleness)
├── data/
│   ├── raw/                   # cached NEOCP pulls + archive downloads (gitignored)
│   └── processed/             # feature tables, train/test splits
├── src/
│   ├── fetch_neocp.py         # pull live neocp.json / neocp.txt
│   ├── fetch_archive.py       # pull Previous-NEOCP + removal reasons (labels)
│   ├── features.py            # feature engineering (Digest2 params, arc, motion, etc.)
│   ├── labels.py              # join postings → confirmed/removed outcomes
│   ├── dataset.py             # build time-split train/test tables
│   ├── train.py               # baseline + rule filter + ML models
│   ├── evaluate.py            # metrics vs. benchmark, plots
│   └── predict.py             # score today's live NEOCP list
├── notebooks/
│   └── eda.ipynb              # exploration, benchmark reproduction
├── models/                    # serialized trained models + calibration
├── outputs/
│   └── daily_vetting.csv      # ranked real-NEO probability for current NEOCP
└── tests/
```

### Existing scripts (created 2026-07-04)

| Script | Dependencies | What it does | Output |
|--------|-------------|-------------|--------|
| `sample_fetch_neocp.py` | stdlib only | Fetches one live NEOCP text feed frame, parses the fixed-width format, displays sorted table with Digest2 scores | Terminal table |
| `visualize_neocp.py` | matplotlib, numpy | Fetches live NEOCP, produces 4-panel plot: Aitoff sky map, score bar chart, H-vs-V magnitude scatter, staleness-vs-arc bubble chart | `neocp_snapshot.png` |
| `fetch_zooniverse_css.py` | stdlib only | Pulls N subjects (4-frame CSS detection sequences + expert labels + metadata) from Zooniverse API | `data/css_zooniverse/subject_*/` (frames + JSON) |

Suggested `predict.py` output: today's NEOCP objects sorted by predicted real-NEO probability, with the Digest2 score alongside for comparison — the tangible artifact to share.

---

## 5. How to contribute the result (audience & channels)

- **Public GitHub repo** (model + pipeline + reproducible benchmark results).
- **Minor Planet Mailing List (mpml)** on groups.io — where NEOfixer itself was announced; the active follow-up community. https://groups.io/g/mpml/
- Offer as **input to the MPC's NEOCP posting criteria** and/or as a filtering layer feeding **NEOfixer**.
- Optional, higher-barrier: obtain an **MPC observatory code** and contribute actual follow-up astrometry (adds the scarce resource directly).

---

## 6. Roadmap (suggested milestones)

1. **M1 — Data spine:** `fetch_neocp.py` + `fetch_archive.py`; cache a labeled historical dataset.
2. **M2 — Baselines:** raw Digest2 threshold + reproduce A&A rule-based filter; establish the benchmark numbers.
3. **M3 — ML models:** train GBM/RF/NN; calibrate; time-based evaluation; beat/match benchmark.
4. **M4 — Live scoring:** `predict.py` producing `daily_vetting.csv` from the live feed.
5. **M5 — Publish:** README with results table, LICENSE, mpml announcement, optional preprint.

---

## 7. Key references
- MPC NEOCP (live): https://minorplanetcenter.net/iau/NEO/toconfirm_tabular.html
- MPC processing / Digest2 > 65 posting rule: https://docs.minorplanetcenter.net/mpc-ops-docs/astrometry/mpc-processing/
- Digest2 classifier tutorial: https://docs.minorplanetcenter.net/tutorials/notebooks/mpc_tutorial_digest2/
- NEOCP observations API: https://docs.minorplanetcenter.net/mpc-ops-docs/apis/get-obs-neocp/
- NEOfixer (existing broker) FAQ: https://neofixer.arizona.edu/faq
- NEOfixer PDC 2021: https://ui.adsabs.harvard.edu/abs/2021plde.confE.206C/abstract
- ML for NEOCP vetting (A&A 2025, benchmark): https://www.aanda.org/articles/aa/full_html/2025/06/aa54311-25/aa54311-25.html
- Rubin/LSST follow-up impact (arXiv 2408.12517): https://arxiv.org/html/2408.12517v1
- NASA NeoWs API: https://api.nasa.gov/
- JPL CAD API: https://ssd-api.jpl.nasa.gov/doc/cad.html
- CNEOS Fireballs API: https://ssd-api.jpl.nasa.gov/doc/fireball.html


---

## Appendix B — A&A 2025 benchmark details (from paper review)

Reviewed 2026-07-04. Source: https://www.aanda.org/articles/aa/full_html/2025/06/aa54311-25/aa54311-25.html

### B.1 Features used

The paper uses **26 Digest2 sub-parameter scores** as ML inputs — 13 orbital categories
each in two variants ("raw" and "noid"):

1. Interesting (Int) — raw & noid
2. NEO — raw & noid
3. Mars crossers (MC) — raw & noid
4. Hungarias (Hun) — raw & noid
5. Phocaeas (Pho) — raw & noid
6. Hansa family (Han) — raw & noid
7. Inner main belt (MBA1) — raw & noid
8. Pallas family (Pal) — raw & noid
9. Central main belt (MBA2) — raw & noid
10. Outer main belt (MBA3) — raw & noid
11. Hildas (Hil) — raw & noid
12. Jupiter trojans (Jtr) — raw & noid
13. Jupiter-family comets (JFC) — raw & noid

These are computed from astrometric tracklet data (RA, Dec, magnitude, epoch in MPC
80-column format) run through the Digest2 tool. **No image/pixel data is used.** The paper
does not discuss image-based features or explain why they were excluded.

### B.2 Models trained

| Model | Details |
|---|---|
| **XGBoost (GBM)** | Binary logistic, 100 estimators, lr=0.1, max_depth=5 |
| **Random Forest** | scikit-learn default |
| **SGD** | Linear classifier, binary cross-entropy loss |
| **Neural Network** | 2 hidden layers x 64 ReLU neurons, Adam optimizer, lr=0.001 |

Ensemble methods also investigated (model stacking with logistic regression meta-learner,
voting mechanisms).

### B.3 Results

- Accuracy: 0.9261 (RF) to 0.9329 (NN)
- Precision for NEO identification: **0.95–0.98**; for non-NEOs: 0.81–0.86
- **SGD selected as best** — fewest misidentified NEOs: 189 of 3428 (5.5%)
- Of those 189, 140 were correctly reclassified on follow-up tracklets → **net loss ~1%**
- Combined filter + ML: **>80% non-NEO reduction** on NEOCP
- Digest2 filters alone: ~20% non-NEO removal at >98% precision
- Projected 2024 impact: 884 non-NEOs would have been correctly prevented from posting

### B.4 Gaps / opportunities for improvement

The benchmark's feature set is deliberately narrow (Digest2 scores only). Features NOT
used that are available from NEOCP metadata:
- Arc length, NObs, staleness (Not_Seen_dys)
- Apparent magnitude (V), absolute magnitude (H)
- Sky-plane motion rate (derivable from RA/Dec + epoch)
- Uncertainty growth over time (requires time-series polling)

**Image/pixel data** could help catch artifacts that fool Digest2 (cosmic rays, diffraction
spikes, hot pixels that survive survey-level filtering). However this requires:
- Cross-matching NEOCP postings to survey cutout services (ZTF, ATLAS, Pan-STARRS)
- Handling heterogeneous image formats and FOVs
- Training a CNN/vision model alongside or upstream of the tabular classifier

Recommended approach: tabular model first (match/beat benchmark with additional metadata
features), then evaluate whether error cases justify image-based v2.

---

## Appendix A — Known data gotchas (read before coding)

These are real quirks observed in the live data on 2026-07-04. They will break naive
implementations — handle them explicitly.

### A.1 The NEOCP "JSON" feed is malformed
`https://www.minorplanetcenter.net/Extended_Files/neocp.json` does **not** return valid
JSON. Records are concatenated key/value lines **without** enclosing array brackets,
without commas between objects, and without object braces. A plain `json.loads()` fails.

Observed raw shape (no `[`, no `{`, no separators between records):
```
"Temp_Desig": "CEQHXE2",
"Score": 86,
...
"Not_Seen_dys": 0.052
"Temp_Desig": "MAS0006",     <-- next record starts with no comma / brace
"Score": 35,
```

**Recommended fix — parse the plain-text feed instead:**
`https://www.minorplanetcenter.net/iau/NEO/neocp.txt`. Columns in order: `Temp_Desig`,
`Score`, `Discovery Y M D.d`, `R.A.`, `Decl.`, `V`, `Updated ... UT` (free text with
spaces), `NObs`, `Arc`, `H`, `Not_Seen_dys`. The "Updated July 4.23 UT" field contains
spaces — tokenize by known field structure, not a naive `split()`.

If you must use the JSON endpoint, reconstruct valid JSON: start a new record at each
`"Temp_Desig"` key, wrap each group in `{}`, join key/value lines with commas, wrap the
set in `[...]`.

### A.2 Labels require joining live postings to the outcome archive
The live NEOCP feed has **no ground-truth label** — an object is only known to be a real
NEO or a false positive *after* it leaves the page. Building the labeled training set is
the hardest part of the project:

- Positive (real NEO): object received a provisional/permanent designation -> in catalog.
- Negative (non-NEO / artifact): removed with a non-NEO reason (main-belt, artificial
  satellite, bad tracklet, failed orbit fit, or "no designation assigned").
- Source of truth: the **Previous NEOCP** and **removal-reason** tables (`neocp_els`,
  `neocp-obs-archive`). Join on the temporary designation (`Temp_Desig` / `trksub`).

**Caution — temp designations are ephemeral/recycled.** A `Temp_Desig` is only meaningful
for the window the object is on the page; it is not a stable global key across long spans.
Capture the discovery date alongside it to disambiguate.

### A.3 Build a time-series capture, not a single snapshot
Several of the most informative features (uncertainty growth, fading, staleness
trajectory) require **successive** feed pulls. Stand up `fetch_neocp.py` as a poller
(e.g. hourly) early and archive every timestamped snapshot, so you accumulate real
time-series history before the models are ready.

### A.4 Rate limits & etiquette
- NASA NeoWs `DEMO_KEY` is heavily rate-limited (~30/hr, 50/day). Get a free personal key
  at https://api.nasa.gov/ for real use.
- MPC asks that you cite the relevant URL + the MPC when using their data, and avoid
  hammering endpoints. Cache aggressively, poll politely, set a descriptive `User-Agent`.

---

## Appendix C — Image data pipeline (v2 approach, validated 2026-07-04)

### C.1 What's available

| Survey | Image type | Public API | Pixel scale | Depth | Coverage of NEOCP |
|--------|-----------|-----------|-------------|-------|-------------------|
| **ZTF** | Science + **difference images** | Yes (IRSA IBE) | ~1.0"/px | r~20.5 | Many NEOCP objects 2018–2024 |
| **Pan-STARRS** | Stacked reference only | Yes (STScI) | 0.25"/px | r~23.2 | All-sky (but asteroid stacked away) |
| **ATLAS** | Forced photometry only | Yes | — | o~19.5 | No public image cutouts |

### C.2 ZTF difference image pipeline (validated working)

The key insight: ZTF provides **difference images** (science minus reference) where an
asteroid appears as a positive residual against a flat background. This is the actual
detection-level data.

**Step 1 — Find exposures at a position and time:**
```
GET https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci?POS={ra_deg},{dec_deg}
    &WHERE=obsdate BETWEEN '{date_start}' AND '{date_end}'&ct=csv
```
Returns: `filefracday`, `field`, `ccdid`, `qid`, `filtercode`, `obsdate`.

**Step 2 — Construct difference image cutout URL:**
```
https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{year}/{monthday}/{fracday}/
  ztf_{filefracday}_{paddedfield}_{filtercode}_c{paddedccdid}_o_q{qid}_scimrefdiffimg.fits.fz
  ?center={ra},{dec}&size=60arcsec&gzip=false
```
The suffix `_scimrefdiffimg.fits.fz` is the key — it selects the difference image
product instead of the science image (`_sciimg.fits`).

**Step 3 — Read the FITS cutout:**
- Image data is in HDU[1] (not HDU[0])
- Returns a 60x60 pixel stamp at ~1 arcsec/pixel
- Values: flux difference in DN; positive = new source, negative = subtraction residual

### C.3 Validated example: 2019 AQ3

Successfully retrieved the difference image for **2019 AQ3** (Atira-class NEO, discovered
by ZTF on 2019-01-04, V=18.96):
- Position: RA=325.663, Dec=+0.353
- Exposure: `ztf_20190104084051_000442_zr_c15_o_q1`
- Result: 60x60 px cutout with clear positive residual visible (range -48 to +87 DN,
  std=12.5 DN above flat background)
- The source appeared offset from center by ~20px because the reported MPC position
  corresponds to a different timestamp than the exposure; fast-moving NEOs require
  **ephemeris propagation** to the exact exposure midpoint

### C.4 Challenges for building an image-based training set

1. **Observatory code matching** — NEOCP doesn't say which survey detected the object.
   Must query MPC observations API for the observatory code, then check if it's ZTF (I41).
   Only ZTF-detected objects have accessible difference images.

2. **Temporal precision** — Fast-moving NEOs shift significantly between exposures.
   The reported NEOCP position must be propagated to the exact exposure timestamp, or
   the source won't be centered in the cutout. Requires knowing the object's motion rate.

3. **Faintness** — Most NEOCP objects are V=20–22, near or beyond ZTF's limit (r~20.5).
   The asteroid may be only a few DN above noise in a single exposure — barely
   distinguishable from background fluctuations without multi-frame stacking.

4. **ZTF temporal coverage** — ZTF public data spans 2018–~2024. No difference images
   available for current (2026) NEOCP objects unless a successor survey provides similar
   access.

5. **Pan-STARRS stacks are NOT useful for moving objects** — Co-adds average many nights;
   the asteroid smears out and disappears. Only useful as contextual background
   (star density, nearby bright sources that could cause confusion).

### C.5 Catalina Sky Survey images via Zooniverse (validated 2026-07-04)

**Major discovery:** CSS detection images are publicly accessible through the Zooniverse
"Daily Minor Planet" project (project ID 19413) — no authentication required for images
or metadata. This is a ready-made labeled dataset for an image-based classifier.

**API access:**
```
GET https://www.zooniverse.org/api/subjects?project_id=19413&workflow_id=22354&page_size=100
Headers: Accept: application/vnd.api+json; version=1
```

**What each subject contains:**

| Field | Content |
|-------|---------|
| `locations[]` | 4 PNG image URLs (817x817 px each, ~95KB) — the detection sequence |
| `!sType` | Object classification: `"NEO"`, `"MBA"` (main-belt) |
| `!sKeyword` | Outcome: `"normal"`, `"rjct"` (rejected as false positive) |
| `fDigest2` | Digest2 score (0–100) |
| `sMag` | Apparent magnitude |
| `#feedback_2_answer` | Expert ground-truth label: `"1"` = not real asteroid |
| `#feedback_2_successMessage` | e.g. "Correct! This is not a real asteroid." |
| `sDetsFile` | CSS internal detection file reference (includes date + field) |

**Dataset scale:**
- **~20,940 subjects** (detection sequences) with 4 frames each = ~84,000 images
- **~2,853 retired subjects** with consensus labels from 167,671 volunteer classifications
- Expert feedback labels embedded in metadata for training data (no auth needed)
- Covers G96 (Mt. Lemmon) observations, actively updated (subjects from June 2026 present)

**Image format:**
- 4 exposures taken ~6 minutes apart of the same field
- 817x817 pixel PNG cutouts centered on the candidate detection
- Green circle overlay marks the candidate position in each frame
- A real asteroid moves between frames; artifacts/noise do not

**Workflow structure (3 workflows, all binary):**
- Workflow 22354: "Validation" — "Is this a real object?" Yes/No
- Workflow 31240: "Bok Validation" — same question, different telescope
- Workflow 24646: "feedback" — same, with expert feedback labels

**Key advantages over ZTF difference images:**
1. No ephemeris propagation needed — images are pre-centered on the candidate
2. Temporal sequence (4 frames) directly shows motion — the key discriminator
3. Labels are included (expert + volunteer consensus)
4. Same classification task as NEOCP vetting ("real or fake?")
5. Tabular features (Digest2, magnitude) included alongside images
6. Active and current — new subjects added nightly

**Limitation:** Requires authentication to access per-subject *volunteer classification
counts* (how many said Yes vs No). But expert feedback labels in metadata are sufficient
for supervised training.

### C.6 Recommendation (updated)

**Phase 1 (current project):** Tabular classifier using Digest2 sub-params + additional
metadata features. Match/beat the A&A 2025 benchmark.

**Phase 2 (image-enhanced — now feasible):** Use Zooniverse/CSS dataset:
- Pull all ~20,940 subjects via API (images + metadata + expert labels)
- Train a CNN or vision model on the 4-frame sequences
- Compare: image-only vs. tabular-only vs. fused (image embeddings + tabular features)
- This directly addresses whether pixel data improves over Digest2-only classification
- The CSS dataset provides both the images AND the Digest2 scores, enabling a controlled
  comparison on the same objects

**Phase 3 (optional — ZTF cross-validation):** For objects detected by both CSS and ZTF,
cross-validate using ZTF difference images to test generalization across surveys.
