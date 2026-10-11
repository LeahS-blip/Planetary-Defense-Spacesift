// Evaluate the Spectrum Lab's star classifier on real SDSS spectra of known type.
//   node spectra/eval/eval_sdss.mjs           (downloads the 11 spectra on first run, ~2 MB, to spectra/eval/cache/)
// Writes spectra/eval/results.json, which the Evidence page reads.
//
// Test set: the highest signal-to-noise SDSS DR17 stars of each class, chosen by a SkyServer query
// (class='STAR', run2d='26', zWarning=0, ordered by snMedian), before any tuning. "Truth" is SDSS's own
// pipeline subclass, which is itself a template fit, so agreement within a subclass or two is the realistic goal.
// Methods compared:
//   templates : the current uploader (123 real stellar spectra, SDSS/Indo-US library)
//   baseline  : the first uploader, a simplified physics model (blackbody + ~40 lines) fit over T x [Fe/H]
//   colors    : a naive baseline that relies on the overall continuum shape ("temperature from color"): the
//               same templates with only a scale factor (polynomial degree 0) and no feature weighting
// Robustness: each spectrum is also tilted by exp(+/-0.6 x) across the range, as dust or a bad calibration would,
// and the worse of the two answers is scored.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseFITS, fitTemplates, loadTemplates } from "../upload.js";
import { fitStar as baselineFit } from "./baseline_model.js";

const here = path.dirname(fileURLToPath(import.meta.url));
export const TEST_SET = [
  // [plate, mjd, fiber, SDSS subclass]
  [2815, 54414, 630, "O"], [2904, 54574, 511, "B6"], [2682, 54401, 569, "B9"], [394, 51913, 246, "A0"],
  [2867, 54479, 271, "F5"], [2913, 54526, 298, "F9"], [2876, 54581, 579, "G2"], [2904, 54574, 109, "K5"],
  [2865, 54503, 157, "K7"], [2871, 54536, 245, "M1"], [1637, 52992, 260, "M4"],
];

/** Spectral type as a number of subclasses from O0 (O0 = 0, B0 = 10, ... M9 = 69). SDSS "O" alone counts as O5. */
export function typeIndex(type) {
  const m = /^(sd)?([OBAFGKM])(\d+(?:\.\d+)?)?/.exec(type || "");
  if (!m) return NaN;
  return "OBAFGKM".indexOf(m[2]) * 10 + (m[3] === undefined ? 5 : +m[3]);
}

async function fetchSpectrum([plate, mjd, fiber]) {
  const p = String(plate).padStart(4, "0"), f = String(fiber).padStart(4, "0");
  const name = `spec-${p}-${mjd}-${f}.fits`, file = path.join(here, "cache", name);
  if (!fs.existsSync(file)) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const url = `https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/${p}/${name}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`download failed: ${url} (${r.status})`);
    fs.writeFileSync(file, Buffer.from(await r.arrayBuffer()));
  }
  const b = fs.readFileSync(file);
  return parseFITS(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
}

const D = path.join(here, "..", "data");
const bin = fs.readFileSync(path.join(D, "templates.bin"));
const lib = loadTemplates(JSON.parse(fs.readFileSync(path.join(D, "templates.json"), "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));

// A deliberately mis-calibrated copy of each spectrum (as dust or a bad flux calibration would do) to test robustness.
const tilt = (s, k) => ({ ...s, flux: s.flux.map((v, i) => v * Math.exp(k * (s.wl[i] - 640) / 260)) });

const rows = [];
for (const star of TEST_SET) {
  const s = await fetchSpectrum(star);
  const truth = star[3];
  const tpl = fitTemplates(s, lib);
  const col = fitTemplates(s, lib, { deg: 0, weight: 1 });
  const base = baselineFit(s);
  const tiltA = fitTemplates(tilt(s, 0.6), lib), tiltB = fitTemplates(tilt(s, -0.6), lib);
  const colA = fitTemplates(tilt(s, 0.6), lib, { deg: 0, weight: 1 }), colB = fitTemplates(tilt(s, -0.6), lib, { deg: 0, weight: 1 });
  const row = {
    id: `${star[0]}-${star[1]}-${star[2]}`, truth,
    templates: tpl.best.type, baseline: base.cls.name, colors: col.best.type,
    tilted: [tiltA.best.type, tiltB.best.type], colorsTilted: [colA.best.type, colB.best.type],
  };
  for (const k of ["templates", "baseline", "colors"]) row[k + "Err"] = Math.abs(typeIndex(row[k]) - typeIndex(truth));
  row.tiltErr = Math.max(...row.tilted.map((t) => Math.abs(typeIndex(t) - typeIndex(truth))));
  row.colorsTiltErr = Math.max(...row.colorsTilted.map((t) => Math.abs(typeIndex(t) - typeIndex(truth))));
  rows.push(row);
  console.log(`${row.id.padEnd(16)} SDSS ${truth.padEnd(3)} templates ${row.templates.padEnd(9)} (${row.templatesErr})  baseline ${row.baseline.padEnd(4)} (${row.baselineErr})  colors ${row.colors.padEnd(9)} (${row.colorsErr})  tilted ${row.tilted.join("/")}`);
}

const summary = (key) => {
  const e = rows.map((r) => r[key]);
  return {
    within1: e.filter((x) => x <= 1).length, within3: e.filter((x) => x <= 3).length, n: e.length,
    medianErr: [...e].sort((a, b) => a - b)[Math.floor(e.length / 2)], meanErr: +(e.reduce((a, b) => a + b, 0) / e.length).toFixed(1),
    worst: Math.max(...e),
  };
};
const out = {
  description: "Spectral type errors (in subclasses; one full class = 10) on 11 SDSS DR17 stars, O to M. Truth = SDSS pipeline subclass.",
  generated: new Date().toISOString().slice(0, 10),
  methods: { templates: summary("templatesErr"), baseline: summary("baselineErr"), colors: summary("colorsErr"), templatesTilted: summary("tiltErr"), colorsTilted: summary("colorsTiltErr") },
  rows,
};
fs.writeFileSync(path.join(here, "results.json"), JSON.stringify(out, null, 1));
console.log(JSON.stringify(out.methods, null, 1));
