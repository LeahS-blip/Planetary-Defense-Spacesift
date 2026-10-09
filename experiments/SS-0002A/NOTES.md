# SS-0002A notes

Write-up: SpaceSift Roadmap doc, Research archive, "SS-0002A results".

## Analysis

```
spacesift analyze-grid experiments/SS-0002A --null experiments/SS-0002A-null --name far1pct
```

Use `analysis/far1pct/`. Thresholds (1% false-alarm rate on noise-only stars, from SS-0002A-null):
none 6.02, biweight 0.25 d 6.75, 0.5 d 6.42, 1 d 6.33, 2 d 6.39, prewhitening + biweight 1 d 6.33.

**Do not use the pooled default** (`analyze-grid` without `--null`). Strongly variable stars score
high even with their flux inverted. In this grid, where half the cells vary strongly, pooling
put every biweight threshold near SDE 24 and wiped out detections even on quiet stars.

## Results (completeness at SNR 10-16)

- Biweight at any window fails on pulsations at 100 ppm or more, for periods from 2 h to 2 d.
  Prewhitening recovers 99% or more in all 24 pulsation cells.
- Prewhitening fails on fast spots: rotation of 1 d at 300 ppm or more, and 3 d at 1,000 ppm.
  There the 0.25-day window is best, but it recovers only 64% on quiet stars.
- 30 ppm pulsations at 16 h to 1 d drop the 1-day biweight to 1-10%, while the same amplitude
  at 2-8 h leaves 88-100%. Variability on transit-like timescales does the most damage.
- In failing biweight cells the search returns the star's own signal in 88-100% of trials, and
  60-100% of planet-free inverted searches beat the threshold.

## Caveats

The simulated pulsators are two constant sinusoids, the ideal case for prewhitening. The noise
is white only. Part B (real Kepler variable stars) is the test.

## Provenance

- 49,200 trials, commit 18bf1f9, run 37556705026 (20 shards).
- Shard 11 failed at `pip install` and was re-run 36 h later. pydantic 2.14.0 was released in
  between, so it differs across shards. This is recorded in `record.json`
  (`package_versions_by_shard`) and does not affect any results.
- Merged by spacesift-merge on commit 5becf13.
