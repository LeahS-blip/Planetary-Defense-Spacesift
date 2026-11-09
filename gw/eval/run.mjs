// Reproducible evaluation of the Gravitational Wave Lab's search, for the Evidence page.
// Fixed seeds, so the same code gives the same numbers. Run from the repo root:  node gw/eval/run.mjs
//   1. calibration: loudest trigger in 2,000 noise-only stretches -> the 1% false-alarm threshold
//   2. an independent check of that false-alarm rate on 2,000 different noise stretches
//   3. completeness: 200 hidden mergers at each loudness (SNR 3-12), chirp mass 15-60 Msun
//   4. how well detected mergers' chirp masses are measured
import { writeFileSync } from "node:fs";
import * as G from "../gw.js";

const THRESHOLD = 5.4; // the value the lab uses (gw/index.html)
const templates = G.bank(48).map(G.prepare);
const loudest = (seed, opts = {}) => G.search(G.simulate({ seed, ...opts }).data, templates).best;
const t0 = Date.now();

const calib = Array.from({ length: 2000 }, (_, i) => loudest(100000 + i).snr);
const check = Array.from({ length: 2000 }, (_, i) => loudest(300000 + i).snr);
const faK = check.filter((s) => s > THRESHOLD).length;

const SNRS = [3, 4, 5, 6, 7, 8, 9, 10, 12], PER = 200, rows = [], mcErr = [];
SNRS.forEach((snr, j) => {
  let k = 0;
  for (let i = 0; i < PER; i++) {
    const seed = 500000 + j * 10000 + i, r = G.rng(seed * 7 + 3), mc = 15 + r() * 45;
    const sim = G.simulate({ seed, mc, snr }), { best } = G.search(sim.data, templates);
    // Found only if it crossed the threshold at the right time; otherwise it found noise, not the merger.
    if (best.snr > THRESHOLD && Math.abs(best.merger - sim.truth.merger) < 0.1) { k++; mcErr.push({ snr, rel: Math.abs(best.mc - mc) / mc }); }
  }
  const [lo, hi] = G.wilson(k, PER);
  rows.push({ snr, found: k, n: PER, p: k / PER, ci: [lo, hi] });
});
// SNR at which half are found, by linear interpolation between the bracketing loudnesses.
const i50 = rows.findIndex((r) => r.p >= 0.5);
const snr50 = i50 > 0 ? rows[i50 - 1].snr + (0.5 - rows[i50 - 1].p) / (rows[i50].p - rows[i50 - 1].p) * (rows[i50].snr - rows[i50 - 1].snr) : null;
const med = (a) => (a.length ? G.quantile(a, 0.5) : null);

const out = {
  generated: new Date().toISOString().slice(0, 10), threshold: THRESHOLD, templates: 48, stretchSeconds: 4,
  calibration: { n: calib.length, median: G.quantile(calib, 0.5), p99: G.quantile(calib, 0.99), max: Math.max(...calib) },
  falseAlarm: { n: check.length, k: faK, rate: faK / check.length, ci: G.wilson(faK, check.length) },
  completeness: rows, snr50,
  chirpMassError: { below7: med(mcErr.filter((e) => e.snr < 7).map((e) => e.rel)), from7: med(mcErr.filter((e) => e.snr >= 7).map((e) => e.rel)), n: mcErr.length },
  seconds: Math.round((Date.now() - t0) / 1000),
};
writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(out, null, 1) + "\n");
console.log(JSON.stringify({ p99: out.calibration.p99.toFixed(2), falseAlarm: out.falseAlarm.rate, snr50: snr50 && snr50.toFixed(2), complete: rows.map((r) => `${r.snr}:${r.p.toFixed(2)}`).join(" "), mcErr: out.chirpMassError, seconds: out.seconds }));
