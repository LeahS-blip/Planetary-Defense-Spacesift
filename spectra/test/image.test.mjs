// Run: node --test spectra/test/image.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import * as S from "../spectra.js";
import * as I from "../image.js";
import * as U from "../upload.js";

const D = new URL("../data/", import.meta.url);
const bin = fs.readFileSync(new URL("templates.bin", D));
const LIB = U.loadTemplates(JSON.parse(fs.readFileSync(new URL("templates.json", D), "utf8")), bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.length));

// The Sun's Fraunhofer lines [nm, width px], as drawn in classic textbook diagrams.
export const FRAUNHOFER = [[393.4, 3], [396.8, 3], [410.2, 1], [422.7, 1], [430.8, 3], [434.0, 1], [438.4, 1], [466.8, 1], [486.1, 3],
  [495.8, 1], [516.7, 1], [517.3, 1], [518.4, 1], [527.0, 3], [532.8, 1], [552.0, 1], [589.0, 3], [589.6, 3], [627.7, 1], [656.3, 3],
  [686.7, 3], [718.5, 1], [759.4, 3]];

/**
 * Draw a diagram-style spectrum picture: white page, grey label rows, a rainbow strip spanning lo..hi nm
 * with black lines. `shift` mis-renders the rainbow colors (every drawing tool's rainbow differs).
 */
function drawDiagram({ W = 1024, H = 300, lo = 383, hi = 767, lines = FRAUNHOFER, shift = 15, reverse = false, darkEnds = true } = {}) {
  const data = new Uint8ClampedArray(W * H * 4).fill(255);
  const set = (x, y, r, g, b) => { const p = 4 * (y * W + x); data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = 255; };
  const nmAt = (x) => (reverse ? hi - ((hi - lo) * x) / (W - 1) : lo + ((hi - lo) * x) / (W - 1));
  const xAt = (l) => Math.round(reverse ? ((hi - l) / (hi - lo)) * (W - 1) : ((l - lo) / (hi - lo)) * (W - 1));
  for (let x = 0; x < W; x++) {
    const l = nmAt(x);
    let [r, g, b] = S.wavelengthRGB(Math.min(700, 380 + (l - 380) * 0.92 + shift));
    if (darkEnds && l > 690) { const k = Math.max(0.25, 1 - (l - 690) / 100); r *= k; }
    for (let y = 52; y < 212; y++) set(x, y, 255 * r, 255 * g, 255 * b);
  }
  for (const [l, w] of lines) {
    const x = xAt(l);
    for (let dx = 0; dx < w; dx++) for (let y = 40; y < 222; y++) if (x + dx >= 0 && x + dx < W) set(x + dx, y, 0, 0, 0);
  }
  for (let x = 0; x < W; x += 7) for (let y = 230; y < 260; y++) if ((x + y) % 5 < 2) set(x, y, 40, 40, 40); // "text"
  return { width: W, height: H, data };
}

test("hue mapping runs violet to red", () => {
  assert.ok(I.hueToNm(270) < I.hueToNm(200) && I.hueToNm(200) < I.hueToNm(100) && I.hueToNm(100) < I.hueToNm(30));
  assert.ok(Number.isNaN(I.hueToNm(2)));
});

test("finds the strip and ignores the white page and labels", () => {
  const s = I.findStrip(drawDiagram());
  assert.ok(s.y0 >= 52 && s.y1 < 212 && s.x0 === 0 && s.x1 === 1023, JSON.stringify(s));
  assert.equal(I.findStrip({ width: 200, height: 100, data: new Uint8ClampedArray(200 * 100 * 4).fill(255) }), null);
});

test("wavelength scale is locked in by the dark lines", () => {
  for (const [lo, hi, rev] of [[383, 767, false], [400, 700, false], [370, 760, true]]) {
    const r = I.readImageSpectrum(drawDiagram({ lo, hi, reverse: rev }));
    assert.equal(r.cal.method, "lines");
    assert.ok(Math.abs(r.cal.left - lo) < 4 && Math.abs(r.cal.right - hi) < 4, `${lo}-${hi}: got ${r.cal.left.toFixed(1)}-${r.cal.right.toFixed(1)}`);
    assert.equal(r.cal.reversed, rev);
  }
});

test("a drawing of the Sun's spectrum is classified as a Sun-like star", () => {
  const r = I.readImageSpectrum(drawDiagram());
  const fit = U.fitTemplates(r, LIB, { normalized: true });
  assert.ok(["F", "G", "K"].includes(fit.best.letter), `matched ${fit.best.type}`);
  const found = fit.lines.filter((l) => l.found).map((l) => l.id);
  for (const id of ["CaII", "Na", "H"]) assert.ok(found.includes(id), `missing ${id}: ${found}`);
});

test("a hot star drawing (hydrogen only) is classified as hot", () => {
  const balmer = [[397.0, 4], [410.2, 4], [434.0, 5], [486.1, 6], [656.3, 6]];
  const fit = U.fitTemplates(I.readImageSpectrum(drawDiagram({ lines: balmer })), LIB, { normalized: true });
  assert.ok(["B", "A"].includes(fit.best.letter), `matched ${fit.best.type}`);
});
