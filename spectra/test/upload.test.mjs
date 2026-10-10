// Run: node --test spectra/test/upload.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import * as S from "../spectra.js";
import fs from "node:fs";
import * as U from "../upload.js";

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

const D = new URL("../data/", import.meta.url);
const bin = fs.readFileSync(new URL("templates.bin", D));
const LIB = U.loadTemplates(JSON.parse(fs.readFileSync(new URL("templates.json", D), "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));

test("template library decodes", () => {
  assert.equal(LIB.list.length, 123);
  assert.equal(LIB.wl.length, 521);
  assert.ok(LIB.list.every((t) => t.flux.length === 521 && Math.max(...t.flux) === 1));
  assert.deepEqual([...new Set(LIB.list.filter((t) => t.kind === "star").map((t) => t.letter))].sort(), [..."ABFGKMO"]);
});

test("templates are recovered from tilted, noisy, finely sampled copies of themselves", () => {
  const rng = S.makeRng(3);
  for (const type of ["O8", "B5V", "A1V", "F6V", "G4V", "K3V", "M2III", "Carbon", "WDhotter"]) {
    const t = LIB.list.find((x) => x.type === type);
    const wl = S.grid(380, 900, 0.2);
    const flux = wl.map((l) => {
      const i = Math.min(520, Math.max(0, Math.round(l - 380)));
      return t.flux[i] * Math.exp(0.5 * (l - 640) / 260) * (1 + 0.02 * rng.normal());
    });
    const fit = U.fitTemplates({ wl, flux }, LIB);
    assert.equal(fit.best.letter ?? fit.best.kind, t.letter ?? t.kind, `${type} matched as ${fit.best.type}`);
  }
});

test("real SDSS K5 star: classified as K, with the right lines found", () => {
  const obs = U.parseSpectrumText(fs.readFileSync(new URL("example-sdss-k5.csv", D), "utf8"));
  assert.equal(obs.unit, "ångströms");
  const fit = U.fitTemplates(obs, LIB);
  assert.equal(fit.best.letter, "K");
  assert.equal(fit.quality, "good");
  const found = fit.lines.filter((l) => l.found).map((l) => l.id);
  for (const id of ["Na", "Mg", "CaII"]) assert.ok(found.includes(id), `missing ${id}: ${found}`);
  for (const id of ["HeII", "HeI", "TiO"]) assert.ok(!found.includes(id), `spurious ${id}`);
  // Same star seen at 3 nm resolution, over only part of the range.
  const lowres = { wl: [], flux: [] };
  for (let i = 0; i < obs.wl.length; i += 20) if (obs.wl[i] < 700) { lowres.wl.push(obs.wl[i]); lowres.flux.push(obs.flux[i]); }
  assert.equal(U.fitTemplates(lowres, LIB).best.letter, "K");
});

test("fit rejects spectra outside the visible range or without signal", () => {
  assert.throws(() => U.fitTemplates({ wl: S.grid(1000, 2000, 1), flux: S.grid(1000, 2000, 1).map(() => 1) }, LIB), /80 nm/);
  assert.throws(() => U.fitTemplates({ wl: S.grid(400, 800, 1), flux: S.grid(400, 800, 1).map(() => -1) }, LIB), /zero or negative/);
});
