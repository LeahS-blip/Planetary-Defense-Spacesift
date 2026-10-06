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

## Detection threshold (from SS-0001-null, 1,800 noise-only searches)

| False-alarm rate | SDE threshold | Stars excluded | 50% completeness at SNR | Plateau |
| --- | --- | --- | --- | --- |
| 1.3% (old fixed SDE 7) | 7.00 | none | 10.67 | 0.962 |
| 1% | 7.25 | 5 | 10.85 | 0.973 |
| 0.1% (unreliable, see below) | 9.88 | 2 | 13.50 | 0.960 |

Results: `analysis/far0.01/` and `analysis/far0.001/` (`spacesift analyze ... --null experiments/SS-0001-null --far <rate>`).

- **The SS-0001 curve holds up.** Moving from SDE 7 to a calibrated 1% false-alarm threshold shifts the 50% point by 0.2 in SNR.
- **The pulsating stars hide every planet.** On KIC 6205852 and 4280576, all 50 trials each returned the stellar oscillation period instead of the injected planet, whatever its SNR. Together they produced 100 of the 259 "wrong period" results. This is variability *masking* detection, not just causing false alarms.
- **The 0.1% threshold is driven by those two stars.** Their inverted searches (SDE 15–16) are the only null samples above ~8.5. Scrambling breaks their coherent oscillation, so the scrambled tails end near 8.5. With 1,800 samples, 0.1% rests on ~2 points. Re-derive it with the pulsators removed from the null set, and with more scrambles, before using it.

## Open items

- Recovered transits come out ~21% shallower than injected (median found/injected
  depth 0.79): detrending suppression vs BLS box-shape bias still to be separated.
- 4% of injections above SNR 16 are missed (103 trials): inspect with `spacesift replay`.
