// Calibrate the lab's detection threshold: loudest trigger in noise-only data over the whole bank;
// threshold = 99th percentile (1% false alarms per 4-second stretch). Run:  node gw/test/calibrate.mjs [n]
import * as G from "../gw.js";
const n = Number(process.argv[2] || 2000), templates = G.bank(48).map(G.prepare), loud = [];
for (let s = 0; s < n; s++) loud.push(G.search(G.simulate({ seed: 100000 + s }).data, templates).best.snr);
const q = (p) => G.quantile(loud, p).toFixed(2);
console.log(JSON.stringify({ n, median: q(0.5), p90: q(0.9), p99: q(0.99), p999: q(0.999), max: Math.max(...loud).toFixed(2) }));
