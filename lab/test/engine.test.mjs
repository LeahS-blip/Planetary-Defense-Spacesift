// Checks the JavaScript engine against the Python pipeline (fixtures.json) and on
// end-to-end behaviour. Run from the repo root:  node lab/test/engine.test.mjs
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import * as E from "../engine.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url)));
let passed = 0;
const test = (name, fn) => { const t0 = Date.now(); fn(); passed++; console.log(`ok  ${name}  (${Date.now() - t0} ms)`); };
const close = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);

test("transit model matches Python (blocked fraction, shape factor)", () => {
  for (const c of fx.blocked) {
    const inj = { period: 1e9, t0: 0, k: c.k, b: c.b, aRs: 1e9, duration: 1e9 };
    c.z.forEach((z, i) => {
      // Evaluate the JS model at a separation z by placing the planet directly.
      const js = 1 - E.transitModel(Float64Array.of(0), { ...inj, b: z * 1e9 / 1e9, aRs: 1e9 }, 0.4, 0.25)[0];
      close(js, c.blocked[i], 2e-3 * c.k * c.k + 1e-9, `blocked k=${c.k} z=${z}`);
    });
    close(E.meanOverCentral(c.k, c.b, 15, 5, 0.4, 0.25), c.mean_over_central, 2e-3, `meanOverCentral k=${c.k}`);
  }
});

test("geometry matches Python (a/R*, T14, central depth)", () => {
  for (const g of fx.geometry) {
    const a = E.aOverRs(g.P, fx.star);
    close(a, g.a_rs, 1e-9 * g.a_rs, "a/R*");
    close(E.t14(g.P, a, g.k, g.b), g.t14, 1e-9, "T14");
    close(E.centralDepth(g.k, g.b, 0.4, 0.25), g.central_depth, 1e-12, "depth");
  }
});

test("biweight matches wotan to well under the noise", () => {
  const { time, flux, flat } = fx.biweight;
  for (const spec of Object.keys(flat)) {
    const js = E.flatten(Float64Array.from(time), Float64Array.from(flux), spec);
    let worst = 0, sum = 0;
    js.forEach((v, i) => { const d = Math.abs(v - flat[spec][i]); worst = Math.max(worst, d); sum += d; });
    // Noise is 150 ppm; trend differences must be a small fraction of it.
    assert.ok(sum / js.length < 5e-6, `${spec} mean |diff| ${(sum / js.length * 1e6).toFixed(2)} ppm`);
    assert.ok(worst < 4e-5, `${spec} worst |diff| ${(worst * 1e6).toFixed(1)} ppm`);
  }
});

test("prewhitening removes a pulsation; biweight cannot", () => {
  const lc = E.syntheticStar({ baseline: 90, noisePpm: 150, kind: "pulsation", periodD: 4 / 24, ampPpm: 1000, seed: 3 });
  const pw = E.flatten(lc.time, lc.flux, "prewhiten+biweight-1.0"), bw = E.flatten(lc.time, lc.flux, "biweight-1.0");
  assert.ok(E.robustStd(pw) < 1.3 * 150e-6, `prewhitened scatter ${E.robustStd(pw) * 1e6} ppm`);
  assert.ok(E.robustStd(bw) > 3 * 150e-6, `biweight scatter ${E.robustStd(bw) * 1e6} ppm`);
});

test("noise-only star: white-noise sizing hits its target SNR", () => {
  const lc = E.syntheticStar({ baseline: 120, noisePpm: 150, seed: 5 });
  const inj = E.makeInjection({ time: lc.time, noise: lc.noiseOnly, star: E.DEFAULT_STAR, periodD: 6.1, b: 0.3, t0: 2.0, snr: 12 });
  close(inj.snr, 12, 1.5, "expected SNR");
});

const star = { lc: null };
test("end to end: pulsator missed by biweight, found after prewhitening (as in SS-0002A)", () => {
  const lc = E.syntheticStar({ baseline: 120, noisePpm: 150, kind: "pulsation", periodD: 4 / 24, ampPpm: 300, seed: 1 });
  const L = { time: lc.time, flux: lc.flux, noise: lc.noiseOnly, sigma: lc.sigma };
  const planet = { enabled: true, sizeMode: "snr", snr: 14, periodD: 7.3, b: 0.3 };
  const miss = E.runTrial(L, E.DEFAULT_STAR, { planet, detrend: "biweight-1.0", threshold: 7, seed: 1 });
  const hit = E.runTrial(L, E.DEFAULT_STAR, { planet, detrend: "prewhiten+biweight-1.0", threshold: 7, seed: 1 });
  console.log(`    biweight: ${miss.status} (P=${miss.candidate.period.toFixed(4)}, SDE ${miss.candidate.sde.toFixed(1)}, ${miss.seconds}s)`);
  console.log(`    prewhiten: ${hit.status} (P=${hit.candidate.period.toFixed(4)}, SDE ${hit.candidate.sde.toFixed(1)}, ${hit.seconds}s)`);
  assert.notEqual(miss.status, "recovered");
  assert.equal(hit.status, "recovered");
});

test("quiet star: a clear planet is recovered", () => {
  const lc = E.syntheticStar({ baseline: 120, noisePpm: 150, seed: 9 });
  const r = E.runTrial({ time: lc.time, flux: lc.flux, noise: lc.noiseOnly, sigma: lc.sigma }, E.DEFAULT_STAR,
                       { planet: { enabled: true, sizeMode: "snr", snr: 20, periodD: 4.4, b: 0.2 }, detrend: "biweight-1.0", threshold: 7, seed: 2 });
  assert.equal(r.status, "recovered", `got ${r.status}, P=${r.candidate.period}`);
});

console.log(`\n${passed} passed`);
