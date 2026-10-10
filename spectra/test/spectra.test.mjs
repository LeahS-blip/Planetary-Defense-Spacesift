// Run: node --test spectra/test/
import test from "node:test";
import assert from "node:assert/strict";
import * as S from "../spectra.js";

test("blackbody peaks where Wien's law says", () => {
  const wl = S.grid(200, 3000, 1);
  for (const T of [3000, 5772, 10000]) {
    const b = wl.map((l) => S.planck(l, T));
    const peak = wl[b.indexOf(Math.max(...b))];
    assert.ok(Math.abs(peak - S.wienPeakNm(T)) <= 1, `${T} K: ${peak} vs ${S.wienPeakNm(T)}`);
  }
});

test("spectral classes of well-known stars", () => {
  assert.equal(S.spectralClass(5772).name, "G2"); // Sun
  assert.equal(S.spectralClass(9600).letter, "A"); // Vega
  assert.equal(S.spectralClass(22400).letter, "B"); // Spica
  assert.equal(S.spectralClass(3600).letter, "M"); // Betelgeuse
  assert.equal(S.spectralClass(4290).letter, "K"); // Arcturus
});

test("B-V color round-trips and matches the Sun", () => {
  assert.ok(Math.abs(S.bvFromT(5772) - 0.65) < 0.02);
  for (const T of [3500, 5000, 8000, 12000]) assert.ok(Math.abs(S.tFromBv(S.bvFromT(T)) - T) < 1);
  assert.ok(S.bvFromT(4000) > S.bvFromT(9000)); // cooler is redder
});

test("star colors: hot blue, Sun near white, cool red", () => {
  const [rh, , bh] = S.starRGB(30000), [rc, , bc] = S.starRGB(3000), [rs, gs, bs] = S.starRGB(5772);
  assert.ok(bh > rh && rc > bc);
  assert.ok(Math.min(rs, gs, bs) > 0.75, "Sun should look nearly white");
});

test("line strengths follow the Harvard sequence", () => {
  const top = (T) => S.features(T)[0].id;
  assert.equal(top(42000), "HeII");
  assert.equal(top(20000), "HeI");
  assert.equal(top(9500), "H");
  assert.equal(top(5772), "CaII");
  const tio = (T) => S.features(T).find((f) => f.id === "TiO").strength;
  assert.ok(tio(3000) > 0.4 && tio(6000) === 0);
});

test("metallicity deepens metal lines only", () => {
  const d = (id, feh) => S.features(5772, feh).find((f) => f.id === id).strength;
  assert.ok(d("Fe", 0.5) > d("Fe", 0) && d("Fe", 0) > d("Fe", -1.5));
  assert.equal(d("H", 0.5), d("H", -1.5));
});

test("spectrum is normalized and has a dip at H-alpha in an A star", () => {
  const s = S.starSpectrum(9500);
  assert.ok(Math.abs(Math.max(...s.continuum) - 1) < 1e-9);
  const i = s.wl.indexOf(656.25);
  assert.ok(s.flux[i] / s.continuum[i] < 0.6);
});

test("every asteroid template classifies as itself", () => {
  for (const cls of Object.keys(S.ASTEROID_CLASSES)) {
    assert.equal(S.classifySpectrum(S.reflectance(S.ASTEROID_CLASSES[cls]))[0].cls, cls);
  }
});

test("template colors land in the right SDSS color group", () => {
  const expect = { S: "S", Q: "S", A: "S", V: "V", C: "C", B: "C", X: "C", D: "D" };
  for (const [cls, group] of Object.entries(expect)) {
    const c = S.sdssColors(S.reflectance(S.ASTEROID_CLASSES[cls]));
    assert.equal(c.group, group, `${cls}: a*=${c.astar.toFixed(3)} i-z=${c.iz.toFixed(3)}`);
  }
});

test("noisy mystery asteroids are usually identified", () => {
  const rng = S.makeRng(7);
  let hits = 0;
  for (let k = 0; k < 200; k++) {
    const m = S.mysteryAsteroid(rng);
    if (S.classifySpectrum(m.R)[0].cls === m.cls) hits++;
  }
  assert.ok(hits / 200 > 0.75, `hit rate ${hits / 200}`);
});
