// Reproducible evaluation of the Dark Matter Detective's survey, for the Evidence page.
// Deterministic (grid fits on the real SPARC data). Run from the repo root:  node darkmatter/eval/run.mjs
import { readFileSync, writeFileSync } from "node:fs";
import * as D from "../dm.js";

const data = JSON.parse(readFileSync(new URL("../data/sparc.json", import.meta.url)));
const s = D.survey(data.galaxies), n = s.length;
const count = (f) => s.filter(f).length;
const ratios = s.map((x) => x.budget.ratio).sort((a, b) => a - b);
// "Explained by visible matter alone" needs both a decent fit and a believable stellar mass-to-light ratio.
const visOK = (x) => x.vis.rchi2 <= 2 && x.vis.yd <= D.Y_MAX_PLAUSIBLE;
const out = {
  generated: new Date().toISOString().slice(0, 10), source: data.source, sample: { total: data.galaxies.length, quality: n, cut: "Q < 3 and inclination >= 30 deg" },
  yDisk: D.Y_DISK, yMaxPlausible: D.Y_MAX_PLAUSIBLE,
  visibleOnly: { badFit: count((x) => x.vis.rchi2 > 5), needsHeavyStars: count((x) => x.vis.yd > D.Y_MAX_PLAUSIBLE), explained: count(visOK), medianRchi2: D.quantile(s.map((x) => x.vis.rchi2), 0.5) },
  halo: { badFit: count((x) => x.halo.rchi2 > 5), goodFit: count((x) => x.halo.rchi2 <= 2), medianRchi2: D.quantile(s.map((x) => x.halo.rchi2), 0.5) },
  haloBetter: count((x) => x.halo.rchi2 < x.vis.rchi2),
  massRatio: { median: D.quantile(ratios, 0.5), p10: D.quantile(ratios, 0.1), p90: D.quantile(ratios, 0.9), over2: count((x) => x.budget.ratio > 2), values: ratios.map((r) => +r.toFixed(2)) },
  haloFailures: s.filter((x) => x.halo.rchi2 > 5).map((x) => ({ name: x.name, rchi2: +x.halo.rchi2.toFixed(1) })).sort((a, b) => b.rchi2 - a.rchi2).slice(0, 8),
};
writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(out) + "\n");
const { values, ...brief } = out.massRatio;
console.log(JSON.stringify({ n, visibleOnly: out.visibleOnly, halo: out.halo, haloBetter: out.haloBetter, massRatio: brief, haloFailures: out.haloFailures.map((f) => f.name).join(", ") }));
