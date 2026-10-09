// SpaceSift Transit Lab engine: a JavaScript port of the Python trial pipeline
// (spacesift/inject.py, data.py, detrend.py, search.py, evaluate.py) so the demo runs
// in the visitor's browser. Experiments themselves use the Python pipeline; this port
// is checked against it in lab/test/engine.test.mjs.

const G = 6.674e-11, M_SUN = 1.989e30, R_SUN = 6.957e8, R_EARTH = 6.371e6, DAY = 86400;
export const CADENCE = 29.4244 / 1440; // Kepler long cadence, days
export const DURATIONS_HR = [1.5, 2.5, 4.0, 6.0, 9.0];

// ------------------------------------------------------------------ random numbers

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRng(seed) {
  const u = mulberry32((seed >>> 0) || 1);
  let spare = null;
  return {
    uniform: (a = 0, b = 1) => a + (b - a) * u(),
    normal() {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r === 0 || r >= 1);
      const f = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * f;
      return x * f;
    },
  };
}

// ------------------------------------------------------------------ statistics

export function median(a) {
  const s = Float64Array.from(a).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

export function robustStd(a) {
  const m = median(a);
  return 1.4826 * median(Array.from(a, (x) => Math.abs(x - m)));
}

export function p2pNoise(flux) {
  const d = new Float64Array(flux.length - 1);
  for (let i = 1; i < flux.length; i++) d[i - 1] = flux[i] - flux[i - 1];
  return robustStd(d) / Math.SQRT2;
}

// ------------------------------------------------------------------ synthetic stars

function smoothNoise(rng, t, tau) {
  // Unit-variance curve varying on `tau` days: Catmull-Rom spline through N(0,1) knots.
  const t0 = t[0] - tau, nk = Math.ceil((t[t.length - 1] - t0) / tau) + 4;
  const k = Array.from({ length: nk }, () => rng.normal());
  const out = new Float64Array(t.length);
  for (let n = 0; n < t.length; n++) {
    const u = (t[n] - t0) / tau, i = Math.floor(u), f = u - i;
    const p0 = k[Math.max(i - 1, 0)], p1 = k[i], p2 = k[Math.min(i + 1, nk - 1)], p3 = k[Math.min(i + 2, nk - 1)];
    out[n] = 0.5 * (2 * p1 + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f);
  }
  return out;
}

export function stellarVariability(rng, t, kind, periodD, ampPpm) {
  // Same models as spacesift.data.stellar_variability: amp = semi-amplitude of the fundamental.
  const n = t.length, out = new Float64Array(n);
  if (kind === "none" || !ampPpm) return out;
  const A = ampPpm * 1e-6;
  if (kind === "pulsation") {
    const p1 = rng.uniform(0, 2 * Math.PI), p2 = rng.uniform(0, 2 * Math.PI);
    for (let i = 0; i < n; i++) { const w = (2 * Math.PI * t[i]) / periodD; out[i] = A * (Math.sin(w + p1) + 0.3 * Math.sin(2 * w + p2)); }
    return out;
  }
  const tau = 3 * periodD;
  const a1 = smoothNoise(rng, t, tau), a2 = smoothNoise(rng, t, tau), ph1 = smoothNoise(rng, t, tau), ph2 = smoothNoise(rng, t, tau);
  for (let i = 0; i < n; i++) {
    const w = (2 * Math.PI * t[i]) / periodD;
    out[i] = A * (Math.max(0, 1 + 0.4 * a1[i]) * Math.sin(w + ph1[i]) + 0.5 * Math.max(0, 1 + 0.4 * a2[i]) * Math.sin(2 * w + ph2[i]));
  }
  return out;
}

export function syntheticStar({ baseline = 120, noisePpm = 150, kind = "none", periodD = 1, ampPpm = 0, seed = 1, gapFraction = 0.05 }) {
  const rng = makeRng(seed);
  const all = [];
  for (let x = 0; x < baseline; x += CADENCE) all.push(x);
  const nGaps = Math.max(1, Math.floor(baseline / 90));
  const gaps = [];
  for (let g = 0; g < nGaps; g++) { const s = rng.uniform(0, baseline); gaps.push([s, s + (gapFraction * baseline) / nGaps]); }
  const t = all.filter((x) => !gaps.some(([s, e]) => x > s && x < e));
  const n = t.length, sig = noisePpm * 1e-6;
  const noise = new Float64Array(n);
  for (let i = 0; i < n; i++) noise[i] = sig * rng.normal();
  const v = stellarVariability(rng, t, kind, periodD, ampPpm);
  const time = Float64Array.from(t), flux = new Float64Array(n), noiseOnly = new Float64Array(n);
  for (let i = 0; i < n; i++) { flux[i] = 1 + v[i] + noise[i]; noiseOnly[i] = 1 + noise[i]; }
  return { time, flux, noiseOnly, sigma: sig };
}

// ------------------------------------------------------------------ transit model

export function aOverRs(P, star) {
  const a = Math.cbrt((G * star.mass * M_SUN * (P * DAY) ** 2) / (4 * Math.PI ** 2));
  return a / (star.radius * R_SUN);
}

export function t14(P, aRs, k, b) {
  const inc = Math.acos(b / aRs);
  const arg = Math.sqrt(Math.max((1 + k) ** 2 - b * b, 0)) / (aRs * Math.sin(inc));
  return (P / Math.PI) * Math.asin(Math.min(arg, 1));
}

export function centralDepth(k, b, u1, u2) {
  const mu = Math.sqrt(Math.max(1 - b * b, 0));
  return (k * k * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)) / (1 - u1 / 3 - u2 / 6);
}

function blockedFraction(z, k, u1, u2, nr = 48) {
  // Limb-darkened stellar flux covered by a planet at separation z: annulus integration.
  if (z >= 1 + k) return 0;
  const zz = Math.max(z, 1e-9);
  const lo = Math.min(Math.max(zz - k, 0), 1), hi = Math.min(Math.max(zz + k, 0), 1);
  const dr = (hi - lo) / nr;
  let s = 0;
  for (let j = 0; j < nr; j++) {
    const r = lo + (j + 0.5) * dr;
    const c = Math.min(1, Math.max(-1, (r * r + zz * zz - k * k) / (2 * r * zz)));
    const mu = Math.sqrt(Math.max(1 - r * r, 0));
    s += (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) * 2 * r * Math.acos(c);
  }
  return (s * dr) / (Math.PI * (1 - u1 / 3 - u2 / 6));
}

export function transitModel(time, inj, u1, u2, exptime = 0, supersample = 1) {
  const n = time.length, out = new Float64Array(n).fill(1);
  const ci = Math.cos(Math.acos(inj.b / inj.aRs));
  const offs = exptime && supersample > 1
    ? Array.from({ length: supersample }, (_, j) => ((j + 0.5) / supersample - 0.5) * exptime) : [0];
  for (let i = 0; i < n; i++) {
    const ph = (time[i] - inj.t0) / inj.period;
    if (Math.abs(ph - Math.round(ph)) * inj.period > inj.duration) continue; // far from any transit
    let s = 0;
    for (const o of offs) {
      const phase = (2 * Math.PI * (time[i] + o - inj.t0)) / inj.period;
      const z = Math.cos(phase) > 0 ? Math.hypot(inj.aRs * Math.sin(phase), inj.aRs * Math.cos(phase) * ci) : Infinity;
      s += 1 - blockedFraction(z, inj.k, u1, u2);
    }
    out[i] = s / offs.length;
  }
  return out;
}

export function meanOverCentral(k, b, aRs, P, u1, u2) {
  const dur = t14(P, aRs, k, b);
  const inj = { period: P, t0: 0, k, b, aRs, duration: dur };
  const t = Float64Array.from({ length: 399 }, (_, i) => -dur / 2 + ((i + 1) * dur) / 400);
  const f = transitModel(t, inj, u1, u2);
  let m = 0; for (const v of f) m += 1 - v;
  return m / f.length / centralDepth(k, b, u1, u2);
}

// ------------------------------------------------------------------ noise and sizing

export function cdpp(time, flux, dur) {
  // Robust scatter of duration-wide bin means (spacesift.inject.cdpp).
  const cadence = median(Array.from({ length: Math.min(time.length - 1, 2000) }, (_, i) => time[i + 1] - time[i]));
  const need = Math.max(1, Math.floor((0.5 * dur) / cadence));
  const sums = new Map(), counts = new Map();
  for (let i = 0; i < time.length; i++) {
    const b = Math.floor((time[i] - time[0]) / dur);
    sums.set(b, (sums.get(b) || 0) + flux[i]); counts.set(b, (counts.get(b) || 0) + 1);
  }
  const means = [];
  for (const [b, c] of counts) if (c >= need) means.push(sums.get(b) / c);
  return robustStd(means);
}

export function countTransits(time, P, t0, dur) {
  const cadence = time[1] - time[0];
  const need = Math.max(1, Math.floor((0.5 * dur) / cadence));
  const per = new Map();
  for (const t of time) {
    const e = Math.round((t - t0) / P);
    if (Math.abs(t - (t0 + e * P)) < dur / 2) per.set(e, (per.get(e) || 0) + 1);
  }
  let n = 0; for (const c of per.values()) if (c >= need) n++;
  return n;
}

export function makeInjection({ time, noise, star, periodD, b, t0, snr = null, radiusEarth = null }) {
  // SNR on the mean in-transit depth against `noise` (white noise around 1), as in SS-0002.
  const { u1, u2 } = star;
  const aRs = aOverRs(periodD, star);
  let k;
  if (radiusEarth != null) k = (radiusEarth * R_EARTH) / (star.radius * R_SUN);
  else {
    k = 0.01;
    for (let pass = 0; pass < 3; pass++) {
      const dur = t14(periodD, aRs, k, b);
      const nTr = Math.max(countTransits(time, periodD, t0, dur), 1);
      const depthMean = (snr * cdpp(time, noise, dur)) / Math.sqrt(nTr);
      k = Math.sqrt(depthMean / (centralDepth(1, b, u1, u2) * meanOverCentral(k, b, aRs, periodD, u1, u2)));
    }
  }
  const duration = t14(periodD, aRs, k, b);
  const nTransits = countTransits(time, periodD, t0, duration);
  const depth = centralDepth(k, b, u1, u2);
  const expectedSnr = ((depth * meanOverCentral(k, b, aRs, periodD, u1, u2)) / cdpp(time, noise, duration)) * Math.sqrt(nTransits);
  return { period: periodD, t0, k, b, aRs, duration, depth, radiusEarth: (k * star.radius * R_SUN) / R_EARTH, snr: expectedSnr, nTransits };
}

// ------------------------------------------------------------------ cleaning

function biweightLocation(vals, c = 5, iters = 10) {
  // Tukey biweight location, iterated from the median (wotan's 'biweight', cval 5).
  let loc = median(vals);
  for (let it = 0; it < iters; it++) {
    const mad = median(vals.map((v) => Math.abs(v - loc)));
    if (mad === 0) return loc;
    let num = 0, den = 0;
    for (const v of vals) {
      const u = (v - loc) / (c * mad);
      if (Math.abs(u) < 1) { const w = (1 - u * u) ** 2; num += w * v; den += w; }
    }
    const next = num / den;
    if (Math.abs(next - loc) < 1e-7 * Math.abs(loc || 1)) return next;
    loc = next;
  }
  return loc;
}

export function biweightTrend(time, flux, window, breakTolerance = 0.5) {
  // Sliding time-window biweight location; segments split at gaps > breakTolerance days.
  const n = time.length, trend = new Float64Array(n);
  let segStart = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || time[i] - time[i - 1] > breakTolerance) {
      let lo = segStart, hi = segStart;
      for (let j = segStart; j < i; j++) {
        while (time[lo] < time[j] - window / 2) lo++;
        while (hi < i && time[hi] <= time[j] + window / 2) hi++;
        trend[j] = biweightLocation(Array.from(flux.subarray(lo, hi)));
      }
      segStart = i;
    }
  }
  return trend;
}

export let prewhitenTermsUsed = 0; // for profiling

export function prewhiten(time, flux, { maxTerms = 10, minPeriod = 1 / 24, maxPeriod = 5, snrStop = 5 } = {}) {
  prewhitenTermsUsed = 0;
  // Remove the strongest sinusoids one at a time (spacesift.detrend.prewhiten).
  const n = time.length, med = median(flux);
  const resid = Float64Array.from(flux, (v) => v / med - 1);
  const f0 = 1 / maxPeriod;
  const amplitude = (f) => {
    let c = 0, s = 0;
    for (let i = 0; i < n; i++) { const w = 2 * Math.PI * f * time[i]; c += resid[i] * Math.cos(w); s += resid[i] * Math.sin(w); }
    return (2 / n) * Math.hypot(c, s);
  };
  // Samples sit on a fixed cadence (Kepler, and the synthetic stars), so the spectrum is an
  // FFT of the light curve placed on that grid with zeros in the gaps: the same sums as a
  // direct transform, in milliseconds. Zero-padding to >= 3x the baseline gives the grid step.
  const cadence = median(Array.from({ length: Math.min(n - 1, 2000) }, (_, i) => time[i + 1] - time[i]));
  const idx = Int32Array.from(time, (t) => Math.round((t - time[0]) / cadence));
  let nfft = 1; while (nfft < 3 * (idx[n - 1] + 1)) nfft <<= 1;
  const fStep = 1 / (nfft * cadence), kLo = Math.ceil(f0 / fStep), kHi = Math.min(nfft / 2, Math.floor(1 / minPeriod / fStep));
  for (let term = 0; term < maxTerms; term++) {
    const re = new Float64Array(nfft), im = new Float64Array(nfft);
    for (let i = 0; i < n; i++) re[idx[i]] += resid[i];
    fft(re, im);
    const amp = new Float64Array(kHi - kLo + 1);
    let best = 0;
    for (let k = kLo; k <= kHi; k++) { const a = (2 / n) * Math.hypot(re[k], im[k]); amp[k - kLo] = a; if (a > amp[best]) best = k - kLo; }
    if (amp[best] < snrStop * median(amp)) break;
    prewhitenTermsUsed++;
    // Refine the peak with direct sums on a fine grid around it.
    const fPeak = (best + kLo) * fStep;
    let fBest = fPeak, aBest = -1;
    for (let j = 0; j <= 100; j++) { const f = fPeak + ((j - 50) / 50) * fStep; const a = amplitude(f); if (a > aBest) { aBest = a; fBest = f; } }
    // Least squares for a sin + b cos + const at fBest, then subtract the sinusoid.
    let Sss = 0, Scc = 0, Ssc = 0, Ss = 0, Sc = 0, Sys = 0, Syc = 0, Sy = 0;
    for (let i = 0; i < n; i++) {
      const w = 2 * Math.PI * fBest * time[i], s = Math.sin(w), c = Math.cos(w), y = resid[i];
      Sss += s * s; Scc += c * c; Ssc += s * c; Ss += s; Sc += c; Sys += y * s; Syc += y * c; Sy += y;
    }
    const [a, b] = solve3([[Sss, Ssc, Ss], [Ssc, Scc, Sc], [Ss, Sc, n]], [Sys, Syc, Sy]);
    for (let i = 0; i < n; i++) { const w = 2 * Math.PI * fBest * time[i]; resid[i] -= a * Math.sin(w) + b * Math.cos(w); }
  }
  return Float64Array.from(resid, (v) => v + 1);
}

function fft(re, im) {
  // In-place iterative radix-2 FFT (length a power of two), e^{-i...} convention.
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function solve3(A, y) {
  const M = A.map((r, i) => [...r, y[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

export function flatten(time, flux, spec) {
  // 'none' | 'biweight-<days>' | 'prewhiten+biweight-<days>'; returns flux / trend.
  if (spec.startsWith("prewhiten+")) return flatten(time, prewhiten(time, flux), spec.slice("prewhiten+".length));
  if (spec === "none") { const m = median(flux); return Float64Array.from(flux, (v) => v / m); }
  const trend = biweightTrend(time, flux, parseFloat(spec.split("-")[1]));
  return Float64Array.from(flux, (v, i) => v / trend[i]);
}

export function clean(time, flux, sigmaUpper = 4) {
  // Drop non-finite points and upward outliers only (transits go down).
  const keepT = [], keepF = [];
  const good = Array.from(flux).filter(Number.isFinite);
  const med = median(good), mad = 1.4826 * median(good.map((v) => Math.abs(v - med)));
  for (let i = 0; i < flux.length; i++) if (Number.isFinite(flux[i]) && flux[i] < med + sigmaUpper * mad) { keepT.push(time[i]); keepF.push(flux[i]); }
  return [Float64Array.from(keepT), Float64Array.from(keepF)];
}

// ------------------------------------------------------------------ search

export function shortestDuration(P, minDur) {
  return Math.max(minDur, 0.436 * (13 / 24) * Math.cbrt(P / 365.25));
}

export function periodGrid(baseline, pmin, pmax, minDur, oversample) {
  const out = [];
  let f = 1 / pmax;
  while (f < 1 / pmin) { out.push(1 / f); f += (shortestDuration(1 / f, minDur) * f) / (oversample * baseline); }
  return out.reverse();
}

function binLightcurve(time, flux, width) {
  const t0 = time[0], sums = new Map();
  for (let i = 0; i < time.length; i++) {
    const b = Math.floor((time[i] - t0) / width), e = sums.get(b) || [0, 0, 0];
    e[0] += time[i]; e[1] += flux[i]; e[2]++; sums.set(b, e);
  }
  const keys = [...sums.keys()].sort((a, b) => a - b);
  return [Float64Array.from(keys, (k) => sums.get(k)[0] / sums.get(k)[2]), Float64Array.from(keys, (k) => sums.get(k)[1] / sums.get(k)[2])];
}

// The search is split into three steps so the browser can spread blsPower over several
// workers: blsSetup (bin + period grid), blsPower (any slice of the periods), blsCombine.

export function blsSetup(time, flux, { periodRange = [1, 30], durationsHr = DURATIONS_HR, oversample = 3, binWidth = 1 / 24 } = {}) {
  const [t, f] = binLightcurve(time, flux, binWidth);
  const dmin = Math.min(...durationsHr) / 24;
  const periods = Float64Array.from(periodGrid(t[t.length - 1] - t[0], periodRange[0], periodRange[1], dmin, oversample));
  return { t, f, periods, durationsHr };
}

export function blsPower(t, f, periods, durationsHr = DURATIONS_HR) {
  // Box least squares, 'snr' objective, phase-binned (bin = shortest duration / 5).
  // Returns per period: power, and the best duration / phase start (for t0).
  const n = t.length;
  let mean = 0; for (const v of f) mean += v; mean /= n;
  const durs = durationsHr.map((d) => d / 24), bw = Math.min(...durs) / 5;
  const power = new Float64Array(periods.length), bestDur = new Float64Array(periods.length);
  const bestStart = new Float64Array(periods.length), bestNb = new Float64Array(periods.length), bestDepth = new Float64Array(periods.length);
  // Work buffers sized for the longest period, reused for every period (no per-period allocation).
  let maxNb = 1; for (const P of periods) maxNb = Math.max(maxNb, Math.ceil(P / bw));
  const sumB = new Float64Array(maxNb), cntB = new Float64Array(maxNb);
  const cs = new Float64Array(2 * maxNb + 1), cc = new Float64Array(2 * maxNb + 1);
  const dt = Float64Array.from(t, (x) => x - t[0]), df0 = Float64Array.from(f, (x) => x - mean);
  for (let pi = 0; pi < periods.length; pi++) {
    const P = periods[pi], nb = Math.max(1, Math.ceil(P / bw)), inv = 1 / P;
    sumB.fill(0, 0, nb); cntB.fill(0, 0, nb);
    for (let i = 0; i < n; i++) {
      const x = dt[i] * inv;
      let b = ((x - Math.floor(x)) * nb) | 0;
      if (b >= nb) b = nb - 1;
      sumB[b] += df0[i]; cntB[b]++;
    }
    // Cumulative sums over the doubled array handle wrap-around.
    for (let j = 0; j < 2 * nb; j++) { const jj = j < nb ? j : j - nb; cs[j + 1] = cs[j] + sumB[jj]; cc[j + 1] = cc[j] + cntB[jj]; }
    let bp = -Infinity, bd = 0, bs = 0, bdepth = 0;
    for (const d of durs) {
      const w = Math.max(1, Math.round((d / P) * nb));
      if (w >= nb) continue;
      for (let s = 0; s < nb; s++) {
        const nin = cc[s + w] - cc[s];
        if (nin < 1 || nin > n - 1) continue;
        const sin = cs[s + w] - cs[s];
        const depth = -sin / nin * (n / (n - nin)); // mean_out - mean_in, relative to the global mean
        const snr = depth / Math.sqrt(1 / nin + 1 / (n - nin));
        if (snr > bp) { bp = snr; bd = d; bs = s; bdepth = depth; }
      }
    }
    power[pi] = bp; bestDur[pi] = bd; bestStart[pi] = bs; bestNb[pi] = nb; bestDepth[pi] = bdepth;
  }
  return { power, bestDur, bestStart, bestNb, bestDepth };
}

export function blsCombine(t0Ref, periods, p) {
  // SDE over the whole periodogram, and the best candidate (mid-transit time t0).
  let m = 0, v = 0, cnt = 0, ib = 0;
  for (let i = 0; i < p.power.length; i++) if (Number.isFinite(p.power[i])) { m += p.power[i]; cnt++; if (!(p.power[ib] >= p.power[i])) ib = i; }
  m /= cnt;
  for (let i = 0; i < p.power.length; i++) if (Number.isFinite(p.power[i])) v += (p.power[i] - m) ** 2;
  const sd = Math.sqrt(v / cnt);
  const sde = Float64Array.from(p.power, (x) => (x - m) / sd);
  const P = periods[ib];
  const candidate = { period: P, t0: t0Ref + (p.bestStart[ib] / p.bestNb[ib]) * P + p.bestDur[ib] / 2,
                      duration: p.bestDur[ib], depth: p.bestDepth[ib], sde: sde[ib] };
  return { candidate, periods, sde };
}

export function bls(time, flux, opts = {}) {
  const s = blsSetup(time, flux, opts);
  return blsCombine(s.t[0], s.periods, blsPower(s.t, s.f, s.periods, s.durationsHr));
}

// ------------------------------------------------------------------ grading

function epochOffset(tFound, tInj, P) { return Math.abs(((((tFound - tInj + P / 2) % P) + P) % P) - P / 2); }

export function match(inj, cand, { threshold, periodTol = 0.01, minTransits = 2 }) {
  if (inj.nTransits < minTransits) return "not_observable";
  if (!cand || cand.sde < threshold) return "below_threshold";
  const ratio = cand.period / inj.period;
  for (const [target, label] of [[1, "recovered"], [0.5, "alias_half"], [2, "alias_double"]]) {
    if (Math.abs(ratio / target - 1) < periodTol) {
      if (epochOffset(cand.t0, inj.t0, Math.min(cand.period, inj.period)) < inj.duration) return label;
    }
  }
  return "wrong_period";
}

// ------------------------------------------------------------------ one trial

export function prepareTrial(lc, star, p) {
  // Plant the planet, clean, and set up the search. lc: {time, flux, noise (white-noise
  // stand-in around 1), sigma}; p: planet + search settings.
  const rng = makeRng((p.seed || 1) + 7919);
  let inj = null, model = new Float64Array(lc.time.length).fill(1);
  if (p.planet && p.planet.enabled) {
    const t0 = lc.time[0] + rng.uniform(0, p.planet.periodD);
    inj = makeInjection({ time: lc.time, noise: lc.noise, star, periodD: p.planet.periodD, b: p.planet.b, t0,
                          snr: p.planet.sizeMode === "radius" ? null : p.planet.snr,
                          radiusEarth: p.planet.sizeMode === "radius" ? p.planet.radiusEarth : null });
    model = transitModel(lc.time, inj, star.u1, star.u2, CADENCE, 3);
  }
  const raw = Float64Array.from(lc.flux, (v, i) => v * model[i]);
  const [t, f] = clean(lc.time, flatten(lc.time, raw, p.detrend));
  const setup = blsSetup(t, f, { periodRange: [p.periodMin || 1, p.periodMax || 30] });
  return { inj, model, raw, t, f, setup };
}

export function finishTrial(prep, powers, p) {
  const { candidate, periods, sde } = blsCombine(prep.setup.t[0], prep.setup.periods, powers);
  const status = prep.inj ? match(prep.inj, candidate, { threshold: p.threshold })
    : (candidate.sde >= p.threshold ? "false_alarm" : "nothing_found");
  return { status, inj: prep.inj, candidate, raw: prep.raw, model: prep.model, t: prep.t, f: prep.f, periods, sde };
}

export function runTrial(lc, star, p) {
  // Single-threaded trial (tests, calibration). The browser splits blsPower across workers.
  const started = Date.now();
  const prep = prepareTrial(lc, star, p);
  const s = prep.setup;
  const out = finishTrial(prep, blsPower(s.t, s.f, s.periods, s.durationsHr), p);
  return { ...out, seconds: (Date.now() - started) / 1000 };
}

export function concatPowers(parts) {
  // Join blsPower results from consecutive period slices, in order.
  const keys = ["power", "bestDur", "bestStart", "bestNb", "bestDepth"];
  const n = parts.reduce((a, q) => a + q.power.length, 0), out = {};
  for (const k of keys) { out[k] = new Float64Array(n); let o = 0; for (const q of parts) { out[k].set(q[k], o); o += q[k].length; } }
  return out;
}

export const DEFAULT_STAR = { radius: 1.0, mass: 1.0, u1: 0.4, u2: 0.25 };
