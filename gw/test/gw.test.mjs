// Checks the Gravitational Wave Lab engine. Run from the repo root:  node gw/test/gw.test.mjs
import assert from "node:assert/strict";
import * as G from "../gw.js";

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

// 1. Physics against hand-checked numbers.
// GW150914 had a detector-frame chirp mass near 30 Msun. 0.1 s before merger its frequency should be ~43 Hz,
// and ISCO for an equal-mass 69 Msun binary is 4400 Hz / 69 = ~64 Hz.
close(G.fOfTau(0.1, 30), 42.9, 0.5, "f(0.1 s) for Mc = 30");
close(G.fIsco(30), 4397 / G.totalMass(30), 1, "f_isco");
close(G.tauOfF(G.fOfTau(0.37, 22), 22), 0.37, 1e-9, "tau(f(tau)) round trip");
// Every template in the bank fits inside the 4 s window.
for (const mc of [G.MC_MIN, G.MC_MAX]) assert.ok(G.chirp(mc).len < G.N, `chirp for Mc ${mc} fits`);

// 2. FFT round trip and Parseval.
{
  const r = G.rng(3), re = new Float64Array(1024).map(() => r.normal()), im = new Float64Array(1024), x = Float64Array.from(re);
  G.fft(re, im); let e = 0; for (let i = 0; i < 1024; i++) e += re[i] ** 2 + im[i] ** 2;
  close(e / 1024, G.norm(x) ** 2, 1e-6, "Parseval");
  G.fft(re, im, true); for (let i = 0; i < 1024; i++) close(re[i], x[i], 1e-9, "inverse FFT");
}

// 3. The injected SNR is what we asked for, and the filter recovers it.
const templates = G.bank(48).map(G.prepare);
{
  const sim = G.simulate({ seed: 11, mc: 30, snr: 20 });
  close(G.norm(sim.truth.wave) * sim.truth.amp, 20, 1e-9, "optimal SNR");
  const { best } = G.search(sim.data, templates);
  close(best.snr, 20, 2.5, "recovered SNR at Mc 30, SNR 20");
  close(best.mc, 30, 2, "recovered chirp mass");
  close(best.start, sim.truth.start, 20, "recovered start sample");
}

// 4. Noise-only: the loudest trigger over the whole bank. Prints the distribution used for the page's threshold.
{
  const t0 = Date.now(), loud = [];
  for (let s = 0; s < 200; s++) loud.push(G.search(G.simulate({ seed: 5000 + s }).data, templates).best.snr);
  const ms = (Date.now() - t0) / 200;
  console.log(`noise-only loudest SNR: median ${G.quantile(loud, 0.5).toFixed(2)}, 99th pct ${G.quantile(loud, 0.99).toFixed(2)}, max ${Math.max(...loud).toFixed(2)}; ${ms.toFixed(1)} ms per search`);
  assert.ok(G.quantile(loud, 0.5) > 3.5 && G.quantile(loud, 0.5) < 6, "noise peaks in the expected range");
}

// 5. Detection probability rises with SNR.
{
  const found = (snr) => { let k = 0; for (let s = 0; s < 40; s++) { const sim = G.simulate({ seed: 900 + s, mc: 15 + (s % 9) * 5, snr }); if (G.search(sim.data, templates).best.snr > 6.5) k++; } return k / 40; };
  const lo = found(5), hi = found(10);
  console.log(`found at SNR 5: ${lo}, SNR 10: ${hi}`);
  assert.ok(hi > 0.9 && lo < hi, "detection probability rises with SNR");
}
console.log("gw engine: all checks passed");
