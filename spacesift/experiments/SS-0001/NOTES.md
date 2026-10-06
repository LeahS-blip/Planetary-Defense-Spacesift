# SS-0001 notes

## Stars with a pre-injection signal (SDE > 7 before anything was injected)

Plots: `stars/KIC<id>.png` (made with `spacesift inspect-star configs/SS-0001.yaml --star <id>`).
"Inverted SDE" is the same search on flux flipped about 1. A real transit or eclipse
has no counterpart in inverted flux, while symmetric variability scores about the same.

| KIC | SDE | Inverted SDE | Best period | What it is |
| --- | --- | --- | --- | --- |
| 6205852 | 17.4 | 16.0 | 1.144 d | Coherent ~4.6 h oscillation, ~80 ppm. Too fast for the 1-day biweight to remove; BLS locks onto 6 cycles. Not a transit. |
| 4280576 | 15.7 | 15.2 | 1.240 d | Coherent ~3.7 h oscillation, ~200 ppm, amplitude changes by quarter (likely light from a nearby pulsator in the aperture). Not a transit. |
| 11233898 | 8.3 | 4.8 | 28.08 d | Dip-only but no visible event in the fold; ~13 possible transits. Consistent with a noise peak at long period. |
| 11606040 | 7.3 | 4.4 | 25.74 d | Dip-only, depth 37 ppm at 9 h duration; marginal. |
| 10714051 | 7.3 | 8.2 | 1.925 d | Symmetric (inverted scores higher): residual variability. |
| 7817711 | 7.2 | 6.5 | 2.752 d | Symmetric: residual variability. |
| 9911639 | 7.2 | 6.9 | 1.150 d | Symmetric: residual variability. |

**Finding:** the two strongest pre-injection signals are short-period stellar oscillations
that survive biweight detrending with a 1-day window. Fast coherent variability is
therefore a concrete false-alarm source for this pipeline, and probably also a source of
missed injections on those stars. That makes it a candidate failure mode for SS-0002.

## Open items

- Detection threshold: set from SS-0001-null at a fixed false-alarm rate, then
  `spacesift analyze experiments/SS-0001 --null experiments/SS-0001-null --name far1pct`.
- Recovered transits come out ~21% shallower than injected (median found/injected
  depth 0.79): detrending suppression vs BLS box-shape bias still to be separated.
- 4% of injections above SNR 16 are missed (103 trials): inspect with `spacesift replay`.
