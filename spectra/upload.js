// SpaceSift Spectrum Lab: classify an uploaded star spectrum.
//
// Reads two-column text (CSV, TSV, whitespace) or FITS (1D images with a wavelength solution, LAMOST-style
// stacked images, or binary tables such as SDSS spectra), then fits the lab's star model over a grid of
// temperatures and metallicities. Each model is multiplied by a smooth polynomial fitted to the data, so the
// unknown shape of the instrument's response (and dust reddening) is absorbed and the absorption lines
// decide the match. Everything runs in the browser; files never leave the device.

import { SPECIES, starSpectrum, spectralClass, makeRng, grid } from "./spectra.js";

export const FIT_RANGE = [380, 900];
// Bands where Earth's atmosphere absorbs (oxygen and water), masked from the fit.
export const TELLURIC = [[686, 695], [715, 735], [758, 772], [810, 835]];
const FINE = grid(370, 910, 0.25);

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

// ---------------------------------------------------------------- fitting

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
    for (let i = j; i < x.length && x[i] < g[k] + half; i++) { s += y[i]; n++; }
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
function fitPoly(d, m, x, mask, deg) {
  const p = deg + 1, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
  const pw = new Array(p);
  for (let i = 0; i < d.length; i++) {
    if (!mask[i]) continue;
    pw[0] = m[i];
    for (let k = 1; k < p; k++) pw[k] = pw[k - 1] * x[i];
    for (let r = 0; r < p; r++) { b[r] += pw[r] * d[i]; for (let c = 0; c < p; c++) A[r][c] += pw[r] * pw[c]; }
  }
  const beta = solve(A, b);
  if (!beta) return { chi2: Infinity };
  const poly = x.map((xi) => beta.reduce((s, c, k) => s + c * xi ** k, 0));
  let chi2 = 0;
  for (let i = 0; i < d.length; i++) if (mask[i]) chi2 += (d[i] - m[i] * poly[i]) ** 2;
  return { chi2, poly, fit: m.map((v, i) => v * poly[i]) };
}

export const FIT_TEMPS = Array.from({ length: 64 }, (_, i) => Math.round(10 ** (Math.log10(2500) + (i / 63) * (Math.log10(45000) - Math.log10(2500)))));
export const FIT_FEH = [-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5];

/**
 * Fit the star model to an observed spectrum {wl (nm), flux}. Returns the best temperature, metallicity and
 * velocity, a likely temperature range, fit quality, the data and model on the fit grid (both divided by the
 * fitted continuum), and the depth of each element's lines as measured in the data.
 */
export function fitStar(obs) {
  const lo = Math.max(FIT_RANGE[0], obs.wl[0]), hi = Math.min(FIT_RANGE[1], obs.wl[obs.wl.length - 1]);
  if (!(hi - lo >= 80)) {
    throw new Error(`This spectrum covers ${Math.round(obs.wl[0])}–${Math.round(obs.wl[obs.wl.length - 1])} nm. The lab needs at least 80 nm of visible light between 380 and 900 nm.`);
  }
  const inRange = [];
  for (let i = 1; i < obs.wl.length; i++) if (obs.wl[i] >= lo && obs.wl[i] <= hi) inRange.push(obs.wl[i] - obs.wl[i - 1]);
  const step = median(inRange);
  const gstep = Math.max(0.5, step);
  const g = grid(lo + gstep / 2, hi - gstep / 2, gstep);
  const d = binTo(obs.wl, obs.flux, g, gstep / 2);
  const mid = (lo + hi) / 2, half = (hi - lo) / 2;
  const x = g.map((l) => (l - mid) / half);
  const telluric = (l) => TELLURIC.some(([a, b]) => l >= a && l <= b);
  let mask = g.map((l, i) => Number.isFinite(d[i]) && !telluric(l));
  const n0 = mask.filter(Boolean).length;
  if (n0 < 40) throw new Error("Too few usable points in 380–900 nm to classify this spectrum.");
  const scale = median(d.filter((v, i) => mask[i]).map(Math.abs)) || 1;
  const dn = Array.from(d, (v) => v / scale);
  const deg = hi - lo > 300 ? 3 : 2;
  // Instrument resolution: assume about two pixels per resolution element.
  const sigFine = Math.max(2 * step, 0.25) / 2.3548 / 0.25;

  const cache = new Map();
  const model = (T, feh, v) => {
    const key = `${T}|${feh}|${v}`;
    if (!cache.has(key)) {
      const sp = starSpectrum(T, feh, FINE, v);
      cache.set(key, { m: binTo(FINE, smooth(sp.flux, sigFine), g, gstep / 2), c: binTo(FINE, sp.continuum, g, gstep / 2) });
    }
    return cache.get(key);
  };
  const scan = (v) => {
    const out = [];
    for (const T of FIT_TEMPS) for (const feh of FIT_FEH) out.push({ T, feh, chi2: fitPoly(dn, model(T, feh, v).m, x, mask, deg).chi2 });
    return out.sort((a, b) => a.chi2 - b.chi2);
  };

  let v = 0, res = scan(v);
  // Radial velocity: only resolvable when the bins are fine enough.
  if (gstep <= 1) {
    let best = { v: 0, chi2: res[0].chi2 };
    for (let vv = -500; vv <= 500; vv += 25) {
      const c = fitPoly(dn, model(res[0].T, res[0].feh, vv).m, x, mask, deg).chi2;
      if (c < best.chi2) best = { v: vv, chi2: c };
    }
    v = best.v;
  }
  // Clip outliers (emission lines, cosmic rays, sky residuals), then rescan.
  const f0 = fitPoly(dn, model(res[0].T, res[0].feh, v).m, x, mask, deg);
  const r = dn.map((di, i) => (mask[i] ? di - f0.fit[i] : NaN)).filter(Number.isFinite);
  const sig = 1.4826 * median(r.map(Math.abs)) || 1e-9;
  mask = mask.map((ok, i) => ok && Math.abs(dn[i] - f0.fit[i]) < 4 * sig);
  res = scan(v);

  const best = res[0], N = mask.filter(Boolean).length;
  const bm = model(best.T, best.feh, v), bf = fitPoly(dn, bm.m, x, mask, deg);
  const s2 = best.chi2 / Math.max(1, N - deg - 1);
  // Model mismatch makes neighboring residuals correlated, so count at most ~150 independent points.
  const thresh = 9 * s2 * Math.max(1, N / 150);
  const ok = res.filter((m) => m.chi2 - best.chi2 <= thresh);
  const Ts = ok.map((m) => m.T);
  const cont = bm.c.map((c, i) => c * bf.poly[i]);
  const normData = dn.map((v2, i) => v2 / cont[i]);
  const normModel = bm.m.map((m2, i) => m2 / bm.c[i]);
  const relRms = Math.sqrt(best.chi2 / N) / median(bf.fit.filter((_, i) => mask[i]).map(Math.abs));
  const noise = 1.4826 * median(normData.map((nd, i) => (mask[i] ? Math.abs(nd - normModel[i]) : NaN)).filter(Number.isFinite));

  // Line depths measured in the data, next to what the best model expects.
  const z = 1 + v / 299792.458;
  const lines = SPECIES.map((sp) => {
    let sd = 0, sm = 0, n = 0;
    for (const c0 of sp.nm) {
      const c = c0 * z;
      const a = sp.band ? c : c - Math.max(sp.fwhm / 2, gstep), b = sp.band ? c + 15 : c + Math.max(sp.fwhm / 2, gstep);
      for (let i = 0; i < g.length; i++) if (g[i] >= a && g[i] <= b && mask[i]) { sd += 1 - normData[i]; sm += 1 - normModel[i]; n++; }
    }
    return n ? { id: sp.id, depth: sd / n, expected: sm / n, n, err: noise / Math.sqrt(n) } : { id: sp.id, depth: NaN, expected: NaN, n: 0, err: NaN };
  });

  return {
    T: best.T, feh: best.feh, v, cls: spectralClass(best.T),
    Trange: [Math.min(...Ts), Math.max(...Ts)], classes: [...new Set(Ts.map((t) => spectralClass(t).letter))],
    fehRange: [Math.min(...ok.map((m) => m.feh)), Math.max(...ok.map((m) => m.feh))],
    quality: relRms < 0.03 ? "good" : relRms < 0.07 ? "fair" : "poor", relRms,
    atEdge: best.T <= FIT_TEMPS[1] || best.T >= FIT_TEMPS[FIT_TEMPS.length - 2],
    coverage: [lo, hi], step, gstep, N, clipped: n0 - N,
    grid: { wl: g, data: normData, model: normModel, mask },
    lines,
  };
}

/** A simulated observed spectrum as CSV, for trying the uploader: instrument tilt, noise, a cosmic ray. */
export function exampleCSV(T = 4400, feh = -0.3, vKms = 40, seed = 11) {
  const rng = makeRng(seed);
  const wl = grid(380, 900, 0.12);
  const sp = starSpectrum(T, feh, wl, vKms);
  const rows = ["# Simulated spectrum for the SpaceSift Spectrum Lab", `# (made from the lab's model: ${T} K, [Fe/H] = ${feh}, ${vKms} km/s)`, "wavelength_angstrom,flux"];
  sp.flux.forEach((f, i) => {
    const xx = (wl[i] - 640) / 260;
    let y = f * (1 + 0.5 * xx - 0.35 * xx * xx) * (1 + 0.02 * rng.normal()) * 3.2e-15;
    if (i === 1500 || i === 2900) y *= 2.5;
    rows.push(`${(wl[i] * 10).toFixed(2)},${y.toExponential(5)}`);
  });
  return rows.join("\n");
}
