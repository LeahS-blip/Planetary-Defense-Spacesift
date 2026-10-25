// Tests for the light-curve reader (uploads): text tables, magnitudes, and a hand-built Kepler-style FITS file.
// Run:  node lab/test/lightcurve.test.mjs
import { parseLightCurveText, parseLightCurveFITS, normalize } from "../lightcurve.js";
const assert = (c, m) => { if (!c) { console.error("FAIL:", m); process.exitCode = 1; } else console.log("ok:", m); };

// 1. CSV with header, Kepler-like times (BKJD) and flux in e-/s
let csv = "time,pdcsap_flux,pdcsap_flux_err\n";
for (let i = 0; i < 1000; i++) csv += `${(131.5 + i * 0.0204).toFixed(5)},${(12000 + Math.sin(i / 50) * 10).toFixed(2)},3.1\n`;
let r = parseLightCurveText(csv), n = normalize(r);
assert(r.source.includes("pdcsap_flux"), "CSV header picks pdcsap_flux: " + r.source);
assert(Math.abs(n.time[0]) < 1e-9 && Math.abs(n.spanD - 999 * 0.0204) < 1e-6, "time starts at 0, span " + n.spanD.toFixed(2) + " d");
assert(Math.abs(n.cadenceMin - 0.0204 * 1440) < 0.01, "cadence " + n.cadenceMin.toFixed(1) + " min");
assert(Math.abs(n.flux.reduce((a, b) => a + b) / n.n - 1) < 0.01, "flux normalised near 1");

// 2. Headerless whitespace table, with comments and a NaN row
let txt = "# my star\n";
for (let i = 0; i < 400; i++) txt += `${2459000 + i * 0.01}   ${1 + (i % 7 === 0 ? 0.001 : 0)}\n`;
txt += "2459004.1  nan\n";
r = parseLightCurveText(txt); n = normalize(r);
assert(n.n === 400 && r.source === "the first two columns", "headerless table, NaN dropped: " + n.n);

// 3. Magnitudes are converted to flux (brighter = bigger flux)
let mags = "BJD,mag\n";
for (let i = 0; i < 300; i++) mags += `${i * 0.02},${i === 150 ? 10.9 : 11}\n`;
r = parseLightCurveText(mags); n = normalize(r);
assert(r.source.includes("magnitudes"), "mag column detected");
assert(n.flux[150] > n.flux[149], "smaller magnitude -> brighter flux");

// 4. Too short / too few points
try { normalize({ time: [0, 1, 2], flux: [1, 1, 1] }); assert(false, "short data rejected"); } catch (e) { assert(/usable points/.test(e.message), "too few points rejected"); }

// 5. A minimal FITS: primary HDU + BINTABLE with TIME (D), PDCSAP_FLUX (E), QUALITY (J); one flagged row
function card(k, v) { const val = typeof v === "string" ? `'${v.padEnd(8)}'` : String(v); return (k.padEnd(8) + "= " + val.padStart(20)).padEnd(80); }
function block(cards) { let s = cards.join("") + "END".padEnd(80); while (s.length % 2880) s += " "; return Buffer.from(s, "ascii"); }
const rows = 500, rowLen = 8 + 4 + 4;
const prim = block([("SIMPLE  = " + "T".padStart(20)).padEnd(80), card("BITPIX", 8), card("NAXIS", 0), card("OBJECT", "KIC 123")]);
const ext = block([card("XTENSION", "BINTABLE"), card("BITPIX", 8), card("NAXIS", 2), card("NAXIS1", rowLen), card("NAXIS2", rows), card("PCOUNT", 0), card("GCOUNT", 1), card("TFIELDS", 3),
  card("TTYPE1", "TIME"), card("TFORM1", "D"), card("TTYPE2", "PDCSAP_FLUX"), card("TFORM2", "E"), card("TTYPE3", "QUALITY"), card("TFORM3", "J")]);
const data = Buffer.alloc(Math.ceil((rows * rowLen) / 2880) * 2880);
for (let i = 0; i < rows; i++) { data.writeDoubleBE(300 + i * 0.0204, i * rowLen); data.writeFloatBE(i === 10 ? 999999 : 5000 + (i % 3), i * rowLen + 8); data.writeInt32BE(i === 10 ? 128 : 0, i * rowLen + 12); }
const fits = Buffer.concat([prim, ext, data]);
r = parseLightCurveFITS(fits.buffer.slice(fits.byteOffset, fits.byteOffset + fits.length));
assert(r.time.length === rows - 1 && r.flagged === 1, `FITS read ${r.time.length} rows, ${r.flagged} flagged dropped`);
assert(r.source.includes("pdcsap_flux") && r.object === "KIC 123", "FITS columns and object: " + r.source + " / " + r.object);
assert(Math.max(...r.flux) < 6000, "flagged outlier excluded");
