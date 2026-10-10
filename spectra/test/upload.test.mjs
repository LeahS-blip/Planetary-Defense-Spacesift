// Run: node --test spectra/test/upload.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import * as S from "../spectra.js";
import * as U from "../upload.js";

// Simulated observation: model star, instrument tilt, noise, optional wavelength unit.
function observe(T, feh, { v = 0, noise = 0.01, seed = 1, lo = 380, hi = 900, step = 0.15 } = {}) {
  const rng = S.makeRng(seed), wl = S.grid(lo, hi, step);
  const sp = S.starSpectrum(T, feh, wl, v);
  const flux = sp.flux.map((f, i) => { const x = (wl[i] - 640) / 260; return f * (1 + 0.4 * x - 0.3 * x * x) * (1 + noise * rng.normal()) * 1e-15; });
  return { wl, flux };
}

test("text parser: headers, comments, separators and units", () => {
  const rows = (f) => S.grid(400, 800, 1).map((l) => f(l)).join("\n");
  const a = U.parseSpectrumText("# comment\nwavelength,flux\n" + rows((l) => `${l * 10},${1 + l / 1e4}`));
  assert.equal(a.unit, "ångströms"); assert.ok(Math.abs(a.wl[0] - 400) < 1e-9);
  const b = U.parseSpectrumText(rows((l) => `${l}\t${2}`));
  assert.equal(b.unit, "nanometers");
  const c = U.parseSpectrumText(rows((l) => `${l / 1000}  3`));
  assert.equal(c.unit, "microns"); assert.ok(Math.abs(c.wl[10] - 410) < 1e-6);
  const d = U.parseSpectrumText(rows((l) => `${Math.log10(l * 10)};4`));
  assert.equal(d.unit, "log₁₀ ångströms"); assert.ok(Math.abs(d.wl[0] - 400) < 1e-6);
  assert.throws(() => U.parseSpectrumText("hello\nworld"), /two columns/);
});

// Minimal FITS writer for tests.
function card(k, v) {
  const val = typeof v === "string" ? `'${v.padEnd(8)}'` : typeof v === "boolean" ? (v ? "T" : "F") : String(v);
  return (k.padEnd(8) + "= " + val.padStart(typeof v === "string" ? 0 : 20)).padEnd(80);
}
function header(cards) {
  let s = cards.map(([k, v]) => card(k, v)).join("") + "END".padEnd(80);
  while (s.length % 2880) s += " ";
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
}
function pad(bytes) { const out = new Uint8Array(Math.ceil(bytes.length / 2880) * 2880); out.set(bytes); return out; }
function concat(parts) {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}
function f32(arr) { const b = new Uint8Array(arr.length * 4), dv = new DataView(b.buffer); arr.forEach((v, i) => dv.setFloat32(4 * i, v)); return b; }

test("FITS 1D image with a linear wavelength solution", () => {
  const n = 500, flux = Array.from({ length: n }, (_, i) => 1 + i / n);
  const buf = concat([header([["SIMPLE", true], ["BITPIX", -32], ["NAXIS", 1], ["NAXIS1", n], ["CRVAL1", 4000], ["CDELT1", 2], ["CRPIX1", 1], ["OBJECT", "TESTSTAR"]]), pad(f32(flux))]);
  const s = U.parseFITS(buf);
  assert.equal(s.object, "TESTSTAR"); assert.equal(s.wl.length, n);
  assert.ok(Math.abs(s.wl[0] - 400) < 1e-6 && Math.abs(s.wl[1] - 400.2) < 1e-6);
  assert.ok(Math.abs(s.flux[250] - 1.5) < 1e-6);
});

test("FITS binary table with loglam and flux columns (SDSS style)", () => {
  const n = 300, loglam = Array.from({ length: n }, (_, i) => 3.6 + i * 1e-3), flux = Array.from({ length: n }, (_, i) => 10 + i);
  const rows = new Uint8Array(n * 8), dv = new DataView(rows.buffer);
  for (let i = 0; i < n; i++) { dv.setFloat32(8 * i, flux[i]); dv.setFloat32(8 * i + 4, loglam[i]); }
  const buf = concat([
    header([["SIMPLE", true], ["BITPIX", 8], ["NAXIS", 0], ["EXTEND", true]]),
    header([["XTENSION", "BINTABLE"], ["BITPIX", 8], ["NAXIS", 2], ["NAXIS1", 8], ["NAXIS2", n], ["PCOUNT", 0], ["GCOUNT", 1], ["TFIELDS", 2],
            ["TTYPE1", "flux"], ["TFORM1", "E"], ["TTYPE2", "loglam"], ["TFORM2", "E"]]),
    pad(rows),
  ]);
  const s = U.parseFITS(buf);
  assert.equal(s.wl.length, n);
  assert.ok(Math.abs(s.wl[0] - 10 ** 3.6 / 10) < 0.01);
  assert.equal(s.flux[5], 15);
});

test("fit recovers the spectral class of simulated stars", () => {
  for (const [T, seed] of [[3300, 1], [4400, 2], [5800, 3], [7000, 4], [9000, 5], [15000, 6], [35000, 7]]) {
    const fit = U.fitStar(U.toNm(...Object.values(observe(T, 0, { v: 30, seed }))));
    const order = "OBAFGKM", off = Math.abs(order.indexOf(fit.cls.letter) - order.indexOf(S.spectralClass(T).letter));
    assert.ok(off === 0, `${T} K fitted as ${fit.cls.name} (${fit.T} K)`);
    assert.ok(Math.abs(fit.T - T) / T < 0.15, `${T} K fitted at ${fit.T} K`);
    assert.ok(fit.Trange[0] <= fit.T && fit.T <= fit.Trange[1]);
  }
});

test("fit works on partial coverage and reports detected lines", () => {
  const fit = U.fitStar(observe(5800, 0, { lo: 380, hi: 600, seed: 9 }));
  assert.equal(fit.cls.letter, "G");
  const ca = fit.lines.find((l) => l.id === "CaII");
  assert.ok(ca.depth > 3 * ca.err && ca.depth > 0.1, JSON.stringify(ca));
  assert.equal(fit.quality, "good");
});

test("fit rejects spectra outside the visible range", () => {
  assert.throws(() => U.fitStar({ wl: S.grid(1000, 2000, 1), flux: S.grid(1000, 2000, 1).map(() => 1) }), /80 nm/);
});

test("example CSV parses and fits as a K star", () => {
  const fit = U.fitStar(U.parseSpectrumText(U.exampleCSV()));
  assert.equal(fit.cls.letter, "K");
  assert.ok(Math.abs(fit.v - 40) <= 50, `v = ${fit.v}`);
});
