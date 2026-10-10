// SpaceSift Spectrum Lab: classify an uploaded star spectrum.
//
// Reads two-column text (CSV, TSV, whitespace) or FITS (1D images with a wavelength solution, LAMOST-style
// stacked images, or binary tables such as SDSS spectra), then fits the lab's star model over a grid of
// temperatures and metallicities. Each model is multiplied by a smooth polynomial fitted to the data, so the
// unknown shape of the instrument's response (and dust reddening) is absorbed and the absorption lines
// decide the match. Everything runs in the browser; files never leave the device.

import { SPECIES, lineDepth as modelDepth } from "./spectra.js";

export const FIT_RANGE = [380, 900];
// Bands where Earth's atmosphere absorbs (oxygen and water), masked from the fit.
export const TELLURIC = [[686, 695], [715, 735], [758, 772], [810, 835]];

function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
}

/** Sort, drop bad values and convert wavelengths to nm, guessing the unit from their typical size. */
export function toNm(w, f) {
  const pts = [];
  for (let i = 0; i < w.length; i++) if (Number.isFinite(w[i]) && Number.isFinite(f[i])) pts.push([w[i], f[i]]);
  pts.sort((a, b) => a[0] - b[0]);
  let wl = pts.map((p) => p[0]);
  const flux = pts.map((p) => p[1]);
  const m = median(wl);
  let unit = "nanometers";
  if (m > 3.3 && m < 4.2) { wl = wl.map((x) => 10 ** x / 10); unit = "log₁₀ ångströms"; }
  else if (m > 1500) { wl = wl.map((x) => x / 10); unit = "ångströms"; }
  else if (m < 1e-4) { wl = wl.map((x) => x * 1e9); unit = "meters"; }
  else if (m < 5) { wl = wl.map((x) => x * 1000); unit = "microns"; }
  return { wl, flux, unit };
}

/** Two numeric columns (wavelength, flux) from CSV/TSV/whitespace text; header and comment lines are skipped. */
export function parseSpectrumText(text) {
  const w = [], f = [];
  for (const line of text.split(/\r?\n/)) {
    const s = line.trim();
    if (!s || /^[#%!;]/.test(s)) continue;
    const parts = s.split(/[\s,;|]+/).filter(Boolean).map(Number);
    if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) continue;
    w.push(parts[0]); f.push(parts[1]);
  }
  if (w.length < 20) throw new Error("Couldn't find two columns of numbers (wavelength, then brightness) in this file.");
  return toNm(w, f);
}

// ---------------------------------------------------------------- FITS

const ascii = (u8, a, n) => String.fromCharCode(...u8.subarray(a, a + n));
function cardValue(s) {
  const str = /^\s*'((?:[^']|'')*)'/.exec(s);
  if (str) return str[1].replace(/''/g, "'").trim();
  const v = s.split("/")[0].trim();
  if (v === "T") return true;
  if (v === "F") return false;
  const n = Number(v.replace(/D/i, "E"));
  return Number.isNaN(n) ? v : n;
}

const TSIZE = { L: 1, B: 1, I: 2, J: 4, K: 8, A: 1, E: 4, D: 8, C: 8, M: 16, P: 8, Q: 16 };

function readTable(dv, { h, off }) {
  const cols = [];
  let pos = 0;
  for (let i = 1; i <= h.TFIELDS; i++) {
    const m = /^(\d*)([LXBIJKAEDCMPQ])/.exec(String(h["TFORM" + i]).trim());
    if (!m) return null;
    const rep = m[1] === "" ? 1 : +m[1], t = m[2];
    cols.push({ name: String(h["TTYPE" + i] || "").trim().toLowerCase(), t, rep, pos, scale: h["TSCAL" + i] ?? 1, zero: h["TZERO" + i] ?? 0 });
    pos += t === "X" ? Math.ceil(rep / 8) : t === "P" || t === "Q" ? TSIZE[t] : rep * TSIZE[t];
  }
  const numeric = (c) => c && "BIJKED".includes(c.t);
  const fc = cols.find((c) => c.name === "flux") || cols.find((c) => /^(flux|spec|counts)/.test(c.name));
  const wc = cols.find((c) => /^(loglam|wave|wavelength|lambda|wl|wavelen)$/.test(c.name));
  if (!numeric(fc) || !numeric(wc)) return null;
  const read = (c, row, k) => {
    const p = off + row * h.NAXIS1 + c.pos + k * TSIZE[c.t];
    const v = { E: () => dv.getFloat32(p), D: () => dv.getFloat64(p), J: () => dv.getInt32(p), I: () => dv.getInt16(p),
                K: () => Number(dv.getBigInt64(p)), B: () => dv.getUint8(p) }[c.t]();
    return v * c.scale + c.zero;
  };
  const wl = [], flux = [];
  if (h.NAXIS2 === 1 && fc.rep > 1) {
    for (let k = 0; k < Math.min(fc.rep, wc.rep); k++) { flux.push(read(fc, 0, k)); wl.push(read(wc, 0, k)); }
  } else {
    for (let r = 0; r < h.NAXIS2; r++) { flux.push(read(fc, r, 0)); wl.push(read(wc, r, 0)); }
  }
  return { wl, flux, source: `table columns "${wc.name}" and "${fc.name}"` };
}

function readImage(dv, { h, off }) {
  const n1 = h.NAXIS1;
  if (!n1 || n1 < 20) return null;
  const bp = h.BITPIX, bs = Math.abs(bp) / 8, sc = h.BSCALE ?? 1, z = h.BZERO ?? 0;
  const px = (i) => {
    const p = off + i * bs;
    const v = bp === -32 ? dv.getFloat32(p) : bp === -64 ? dv.getFloat64(p) : bp === 16 ? dv.getInt16(p) : bp === 32 ? dv.getInt32(p) : bp === 8 ? dv.getUint8(p) : NaN;
    return v * sc + z;
  };
  const flux = Array.from({ length: n1 }, (_, i) => px(i));
  let wl;
  if (h.COEFF0 !== undefined) {
    wl = flux.map((_, i) => 10 ** (h.COEFF0 + h.COEFF1 * i)); // SDSS: log10 wavelength solution
  } else if (h.CRVAL1 !== undefined) {
    const d = h.CDELT1 ?? h.CD1_1 ?? 1, r = h.CRPIX1 ?? 1;
    wl = flux.map((_, i) => h.CRVAL1 + (i + 1 - r) * d);
    if (h["DC-FLAG"] === 1) wl = wl.map((x) => 10 ** x); // IRAF log-linear
  } else if ((h.NAXIS2 ?? 1) >= 3) {
    wl = Array.from({ length: n1 }, (_, i) => px(2 * n1 + i)); // LAMOST: rows are flux, inverse variance, wavelength
  } else return null;
  return { wl, flux, source: "image" };
}

/** Spectrum from a FITS file (ArrayBuffer). */
export function parseFITS(buf) {
  const u8 = new Uint8Array(buf), dv = new DataView(buf);
  const hdus = [];
  let off = 0;
  while (off + 2880 <= u8.length) {
    const h = {};
    let done = false;
    while (!done && off + 2880 <= u8.length) {
      for (let i = 0; i < 36; i++) {
        const card = ascii(u8, off + 80 * i, 80), key = card.slice(0, 8).trim();
        if (key === "END") { done = true; break; }
        if (card.slice(8, 10) === "= ") h[key] = cardValue(card.slice(10));
      }
      off += 2880;
    }
    if (!done) break;
    let size = 0;
    if (h.NAXIS > 0) {
      size = Math.abs(h.BITPIX) / 8;
      for (let i = 1; i <= h.NAXIS; i++) size *= h["NAXIS" + i];
    }
    size = (size + (h.PCOUNT || 0)) * (h.GCOUNT || 1);
    hdus.push({ h, off });
    off += Math.ceil(size / 2880) * 2880;
  }
  if (!hdus.length || !("SIMPLE" in hdus[0].h)) throw new Error("This doesn't look like a FITS file.");
  const object = String(hdus[0].h.OBJECT || hdus[0].h.OBJNAME || "").trim();
  for (const hdu of hdus) {
    const kind = hdu.h.XTENSION;
    const r = kind === "BINTABLE" ? readTable(dv, hdu) : !kind || kind === "IMAGE" ? readImage(dv, hdu) : null;
    if (r && r.wl.length >= 20) return { ...toNm(r.wl, r.flux), object, source: r.source };
  }
  throw new Error("Found no spectrum (wavelength and flux) in this FITS file.");
}

/** Read a File/Blob of any supported type, un-gzipping if needed. */
export async function readSpectrumFile(file) {
  let buf = await file.arrayBuffer();
  let u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    if (typeof DecompressionStream === "undefined") throw new Error("This browser can't open .gz files; unzip it first.");
    buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    u8 = new Uint8Array(buf);
  }
  if (ascii(u8, 0, 6) === "SIMPLE") return parseFITS(buf);
  return parseSpectrumText(new TextDecoder().decode(u8));
}

// ---------------------------------------------------------------- template library

/** Decode the template library from templates.json (parsed) and templates.bin (ArrayBuffer). */
export function loadTemplates(meta, bin) {
  const u16 = new Uint16Array(bin), n = meta.n;
  const wl = Float64Array.from({ length: n }, (_, i) => meta.wl0 + i * meta.step);
  const list = meta.templates.map((t, k) => ({ ...t, flux: Float64Array.from(u16.subarray(k * n, (k + 1) * n), (v) => v / 65535) }));
  return { wl, list, source: meta.source };
}

let libPromise = null;
/** Fetch and decode the template library once (browser and worker). */
export function getLibrary() {
  libPromise ??= Promise.all([
    fetch(new URL("./data/templates.json", import.meta.url)).then((r) => r.json()),
    fetch(new URL("./data/templates.bin", import.meta.url)).then((r) => r.arrayBuffer()),
  ]).then(([m, b]) => loadTemplates(m, b));
  return libPromise;
}

// ---------------------------------------------------------------- fitting helpers

function smooth(y, sigmaSamples) {
  if (sigmaSamples < 0.3) return Float64Array.from(y);
  const r = Math.ceil(3 * sigmaSamples), k = [];
  for (let i = -r; i <= r; i++) k.push(Math.exp(-0.5 * (i / sigmaSamples) ** 2));
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    let s = 0, w = 0;
    for (let j = -r; j <= r; j++) { const t = i + j; if (t >= 0 && t < y.length) { s += y[t] * k[j + r]; w += k[j + r]; } }
    out[i] = s / w;
  }
  return out;
}

/** Average y over bins of half-width `half` centered on g (NaN where a bin is empty). */
function binTo(x, y, g, half) {
  const out = new Float64Array(g.length).fill(NaN);
  let j = 0;
  for (let k = 0; k < g.length; k++) {
    while (j < x.length && x[j] < g[k] - half) j++;
    let s = 0, n = 0;
    for (let i = j; i < x.length && x[i] < g[k] + half; i++) { if (Number.isFinite(y[i])) { s += y[i]; n++; } }
    if (n) out[k] = s / n;
  }
  return out;
}

function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    if (Math.abs(M[c][c]) < 1e-300) return null;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/** Least-squares fit of data d ≈ m · P(x), P a polynomial of degree deg. Returns {chi2, fit, poly}. */
function fitPoly(d, m, x, mask, deg, w = null) {
  const p = deg + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
  const pw = new Array(p);
  for (let i = 0; i < d.length; i++) {
    if (!mask[i]) continue;
    const wi = w ? w[i] : 1;
    pw[0] = m[i];
    for (let k = 1; k < p; k++) pw[k] = pw[k - 1] * x[i];
    for (let r = 0; r < p; r++) { b[r] += wi * pw[r] * d[i]; for (let c = 0; c < p; c++) A[r][c] += wi * pw[r] * pw[c]; }
  }
  const beta = solve(A, b);
  if (!beta) return { chi2: Infinity };
  const poly = x.map((xi) => beta.reduce((s, c, k) => s + c * xi ** k, 0));
  let chi2 = 0;
  for (let i = 0; i < d.length; i++) if (mask[i]) chi2 += (w ? w[i] : 1) * (d[i] - m[i] * poly[i]) ** 2;
  return { chi2, poly, fit: m.map((v, i) => v * poly[i]) };
}

/** Running upper envelope (85th percentile over ±half samples, then smoothed): a pseudo-continuum. */
function envelope(y, half) {
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const w = [];
    for (let j = Math.max(0, i - half); j <= Math.min(y.length - 1, i + half); j++) if (Number.isFinite(y[j])) w.push(y[j]);
    w.sort((a, b) => a - b);
    out[i] = w.length ? w[Math.floor(0.85 * (w.length - 1))] : NaN;
  }
  return smooth(out, half / 2);
}

/**
 * Line strength as astronomers measure it: mean brightness inside the line versus a straight line drawn
 * between side bands on either side (1 = no line). Model-independent.
 */
function lineDepth(g, y, ok, c, halfWidth, band) {
  const mean = (a, b) => {
    let s = 0, n = 0;
    for (let i = 0; i < g.length; i++) if (g[i] >= a && g[i] <= b && ok[i]) { s += y[i]; n++; }
    return n ? { v: s / n, n } : null;
  };
  let inL, left, right, xl, xr, xc;
  if (band) { // molecular band head: absorbed redward of c, compared with the bright side just blueward
    inL = mean(c + 1, c + 10); left = mean(c - 9, c - 2); right = null; xc = c + 5.5; xl = c - 5.5;
  } else {
    const w = Math.max(halfWidth, 1);
    inL = mean(c - w, c + w); left = mean(c - w - 5, c - w - 1); right = mean(c + w + 1, c + w + 5);
    xl = c - w - 3; xr = c + w + 3; xc = c;
  }
  if (!inL || !left) return null;
  const cont = right ? left.v + ((right.v - left.v) * (xc - xl)) / (xr - xl) : left.v;
  return cont > 0 ? { depth: 1 - inL.v / cont, n: inL.n } : null;
}

// ---------------------------------------------------------------- fitting

// Classification features [center nm, half-width nm]: Ca II K/H + H epsilon, H delta, He I 402.6, G band,
// H gamma, He I 438.8 and 447.1 + Mg II 448.1, He II 468.6, H beta, Mg b, Na D, H alpha, TiO 705, Ca II triplet.
// Lines measured for each element: the cleanest ones at about 1 nm resolution (blended ones left out).
const MEASURE = { HeII: [468.6, 541.2], TiO: [544.8, 615.9, 705.4], Fe: [438.4, 527.0, 532.8], CaII: [393.4, 396.8, 854.2, 866.2] };

export const DIAGNOSTIC = [[395, 5], [410.2, 4], [402.6, 1.5], [430.5, 2.5], [434.0, 4], [438.8, 1.5], [447.6, 2], [468.6, 1.5],
  [486.1, 5], [517.3, 3], [589.3, 2], [656.3, 5], [707, 5], [854, 6]];

const LUM_NAMES = { V: "dwarf", IV: "subgiant", III: "giant", II: "bright giant", Ib: "supergiant", Iab: "supergiant", Ia: "bright supergiant", sd: "metal-poor subdwarf" };
export const lumName = (lum) => LUM_NAMES[lum] || "";

/**
 * Match an observed spectrum {wl (nm), flux} against the template library. Each template is multiplied
 * by a low-order polynomial fitted to the data, which absorbs differences in flux calibration and dust
 * reddening while keeping the absorption lines and the broad shape of molecular bands.
 */
export function fitTemplates(obs, lib, opts = {}) {
  const wl0 = lib.wl[0], wl1 = lib.wl[lib.wl.length - 1];
  const lo = Math.max(wl0, obs.wl[0]), hi = Math.min(wl1, obs.wl[obs.wl.length - 1]);
  if (!(hi - lo >= 80)) {
    throw new Error(`This spectrum covers ${Math.round(obs.wl[0])}–${Math.round(obs.wl[obs.wl.length - 1])} nm. The lab needs at least 80 nm of visible light between ${wl0} and ${wl1} nm.`);
  }
  const gaps = [];
  for (let i = 1; i < obs.wl.length; i++) if (obs.wl[i] >= lo && obs.wl[i] <= hi) gaps.push(obs.wl[i] - obs.wl[i - 1]);
  const step = median(gaps);
  const idx = [];
  for (let i = 0; i < lib.wl.length; i++) if (lib.wl[i] >= lo + 0.5 && lib.wl[i] <= hi - 0.5) idx.push(i);
  const g = idx.map((i) => lib.wl[i]);
  const half = Math.max(0.5, step / 2);
  const d = binTo(obs.wl, obs.flux, g, half);
  // Coarser data than the 1 nm grid: blur the templates to match.
  const sig = step > 1.2 ? step / 2.3548 : 0;
  const temps = lib.list.map((t) => {
    const f = sig ? smooth(t.flux, sig) : t.flux;
    return idx.map((i) => f[i]);
  });

  const telluric = (l) => TELLURIC.some(([a, b]) => l >= a && l <= b);
  let mask = g.map((l, i) => Number.isFinite(d[i]) && !telluric(l));
  const n0 = mask.filter(Boolean).length;
  if (n0 < 40) throw new Error("Too few usable points between 380 and 900 nm to classify this spectrum.");
  const scale = median(d.filter((_, i) => mask[i]).map(Math.abs)) || 1;
  const dn = Array.from(d, (v) => v / scale);
  if (median(dn.filter((_, i) => mask[i])) <= 0) throw new Error("The brightness column is mostly zero or negative, so there is nothing to match.");
  const mid = (lo + hi) / 2, hw = (hi - lo) / 2;
  const x = g.map((l) => (l - mid) / hw);
  const deg = opts.deg ?? 2;
  // Extra weight on the features astronomers classify by (hydrogen, helium, calcium, magnesium, molecules).
  const wt = opts.weight ?? 3;
  const w = g.map((l) => (DIAGNOSTIC.some(([c, h]) => Math.abs(l - c) <= h) ? wt : 1));

  const scan = () => temps.map((t, k) => ({ k, chi2: fitPoly(dn, t, x, mask, deg, w).chi2 })).sort((a, b) => a.chi2 - b.chi2);
  let res = scan();
  // Clip outliers (emission lines, cosmic rays, sky residuals) against the best match, then rescan.
  const f0 = fitPoly(dn, temps[res[0].k], x, mask, deg, w);
  const absr = dn.map((v, i) => (mask[i] ? Math.abs(v - f0.fit[i]) : NaN)).filter(Number.isFinite);
  const sigR = 1.4826 * median(absr) || 1e-9;
  mask = mask.map((ok, i) => ok && Math.abs(dn[i] - f0.fit[i]) < 5 * sigR);
  res = scan();

  const N = mask.filter(Boolean).length;
  const best = res[0], bt = lib.list[best.k], bf = fitPoly(dn, temps[best.k], x, mask, deg, w);
  const wsum = w.reduce((s, v, i) => s + (mask[i] ? v : 0), 0);
  // Templates never match real stars perfectly, so formal error bars are meaningless; instead count as
  // "close" every template whose misfit is within 25% of the best one (tuned on SDSS stars of known type).
  const close = res.filter((r) => Math.sqrt(r.chi2 / best.chi2) <= 1.25).map((r) => lib.list[r.k]);
  const stars = close.filter((t) => t.kind === "star");
  const teffs = stars.map((t) => t.teff);
  const level = median(bf.fit.filter((_, i) => mask[i]).map(Math.abs));
  const relRms = Math.sqrt(best.chi2 / wsum) / level;
  // Pixel-to-pixel noise from second differences (insensitive to real spectral features), so the quality
  // rating reflects how well the template matches, not how noisy the file is.
  const dd = [];
  for (let i = 1; i < dn.length - 1; i++) if (mask[i - 1] && mask[i] && mask[i + 1]) dd.push(Math.abs(dn[i] - 0.5 * (dn[i - 1] + dn[i + 1])));
  const relNoise = (1.4826 * median(dd)) / Math.sqrt(1.5) / level;
  const misfit = Math.sqrt(Math.max(0, relRms ** 2 - relNoise ** 2));

  // Pseudo-continuum for the rainbow strip and line measurements.
  const env = envelope(Array.from(dn, (v, i) => (mask[i] || telluric(g[i]) ? v : NaN)), 12);
  const norm = dn.map((v, i) => v / env[i]);
  const okAll = dn.map((v) => Number.isFinite(v));
  const noise = sigR / median(Array.from(env).filter(Number.isFinite));
  const tPrior = bt.teff ?? (bt.kind === "wd" ? 15000 : 3000);
  const lines = SPECIES.map((sp) => {
    let sd = 0, sm = 0, n = 0, k = 0;
    for (const c of MEASURE[sp.id] || sp.nm) {
      if (c < lo + 6 || c > hi - 6 || telluric(c)) continue;
      const a = lineDepth(g, dn, okAll, c, sp.fwhm / 2 || 1, sp.band);
      const b = lineDepth(g, bf.fit, okAll, c, sp.fwhm / 2 || 1, sp.band);
      if (!a || !b) continue;
      sd += a.depth; sm += b.depth; n += a.n; k++;
    }
    // Only credit an element when the matched temperature allows its lines (at this resolution,
    // neighboring lines blend, e.g. iron lines near the helium II line in a cool star).
    const plausible = sp.id === "H" ? tPrior > 3500 : modelDepth(sp, tPrior) > 0.05; // hydrogen shows in all but the coolest stars
    return k ? { id: sp.id, depth: sd / k, expected: sm / k, n, err: noise / Math.sqrt(n / k), plausible }
             : { id: sp.id, depth: NaN, expected: NaN, n: 0, err: NaN, plausible };
  });
  for (const l of lines) l.found = l.n > 0 && l.plausible && l.depth > Math.max(0.03, 3 * l.err);

  return {
    best: { ...bt, flux: undefined },
    matches: res.slice(0, 5).map((r) => ({ ...lib.list[r.k], flux: undefined, score: Math.sqrt(r.chi2 / best.chi2) })),
    T: bt.teff, Trange: teffs.length ? [Math.min(...teffs), Math.max(...teffs)] : null,
    classes: [...new Set(stars.map((t) => t.letter))],
    lums: [...new Set(stars.map((t) => t.lum).filter(Boolean))],
    quality: misfit < 0.04 ? "good" : misfit < 0.08 ? "fair" : "poor", relRms, relNoise, misfit, noisy: relNoise > 0.05,
    coverage: [lo, hi], step, N, clipped: n0 - N, deg,
    grid: { wl: g, data: dn, model: bf.fit, norm, mask },
    lines,
  };
}
