// Calibrate the JS search's detection threshold: top-peak SDE of noise-only simulated
// stars, per method; threshold = 99th percentile (1% false alarms). SDE depends on how
// many trial periods are searched, so calibrate each baseline the demo offers.
// Run:  node lab/test/calibrate.mjs <n> <baseline_days>   (uses all CPU cores)
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { availableParallelism } from "node:os";
import * as E from "../engine.js";

const METHODS = ["none", "biweight-0.25", "biweight-0.5", "biweight-1.0", "biweight-2.0", "prewhiten+biweight-1.0"];

if (isMainThread) {
  const n = Number(process.argv[2] || 240), baseline = Number(process.argv[3] || 120), cores = availableParallelism();
  const seeds = Array.from({ length: n }, (_, i) => 100000 + i);
  const chunks = Array.from({ length: cores }, (_, c) => seeds.filter((_, i) => i % cores === c));
  const results = (await Promise.all(chunks.map((s) => new Promise((res, rej) => {
    const w = new Worker(new URL(import.meta.url), { workerData: { seeds: s, baseline } });
    w.on("message", res); w.on("error", rej);
  })))).flat();
  const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
  const out = {};
  for (const m of METHODS) {
    const v = results.map((r) => r[m]);
    out[m] = { n: v.length, median: +q(v, 0.5).toFixed(3), p99: +q(v, 0.99).toFixed(3), max: +Math.max(...v).toFixed(3) };
  }
  console.log(JSON.stringify({ baseline, ...out }, null, 2));
} else {
  const rows = workerData.seeds.map((seed) => {
    const lc = E.syntheticStar({ baseline: workerData.baseline, noisePpm: 150, seed });
    const row = {};
    for (const m of METHODS) {
      const [t, f] = E.clean(lc.time, E.flatten(lc.time, lc.flux, m));
      row[m] = E.bls(t, f).candidate.sde;
    }
    return row;
  });
  parentPort.postMessage(rows);
}
