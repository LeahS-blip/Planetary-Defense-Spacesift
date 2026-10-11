// Baseline for evaluation only: the Spectrum Lab's first upload classifier (commit acf6891), which fit the lab's
// simplified physics model (blackbody + ~40 lines) times a polynomial over a temperature x metallicity grid.
// Kept so eval_sdss.mjs can compare it with the template matcher that replaced it.
import { SPECIES, starSpectrum, spectralClass, grid } from "../spectra.js";

export const FIT_RANGE = [380, 900];
// Bands where Earth's atmosphere absorbs (oxygen and water), masked from the fit.
export const TELLURIC = [[686, 695], [715, 735], [758, 772], [810, 835]];
const FINE = grid(370, 910, 0.25);

function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
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
