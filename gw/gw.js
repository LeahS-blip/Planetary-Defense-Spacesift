// Gravitational Wave Lab engine: inspiral chirps, simulated detector noise, and a matched-filter
// template-bank search. Pure functions, no DOM, so the page and the Node tests share them.
//
// Physics (leading post-Newtonian order, quadrupole formula; e.g. Maggiore, "Gravitational Waves" vol. 1, ch. 4):
//   time to merger tau = t_c - t,  T = G*Mc/c^3
//   f_gw(tau)   = (1/pi) * (5 / (256 tau))^(3/8) * T^(-5/8)
//   phase(tau)  = -2 * (tau / (5 T))^(5/8)            (gravitational-wave phase)
//   amplitude  ~ f^(2/3)
// The template stops at the innermost stable circular orbit, f_isco = c^3 / (6^(3/2) pi G M),
// with M the total mass of an equal-mass binary (M = 2^(6/5) Mc). Merger and ringdown are not modelled.
//
// Data are treated as already *whitened*: unit-variance white Gaussian noise per sample.
// For whitened data the optimal signal-to-noise ratio is rho_opt = ||h|| (noise sigma 1).

export const FS = 1024;            // samples per second
export const N = 4096;             // 4 s of data
export const F_LOW = 25;           // Hz: where the template starts
export const TSUN = 4.925490947e-6; // G*Msun/c^3 in seconds
export const MC_MIN = 15, MC_MAX = 60; // chirp-mass range of the search (solar masses)

export const totalMass = (mc) => mc * Math.pow(2, 6 / 5);              // equal masses
export const fIsco = (mc) => 1 / (Math.pow(6, 1.5) * Math.PI * totalMass(mc) * TSUN);
export const tauOfF = (f, mc) => (5 / 256) * Math.pow(Math.PI * f, -8 / 3) * Math.pow(mc * TSUN, -5 / 3);
export const fOfTau = (tau, mc) => Math.pow(5 / (256 * tau), 3 / 8) * Math.pow(mc * TSUN, -5 / 8) / Math.PI;

// --- random numbers: seeded so every run can be replayed
export function rng(seed) {
  let s = (seed >>> 0) || 1;
  const u = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
  let spare = null;
  u.normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const a = u(), b = u(), r = Math.sqrt(-2 * Math.log(a));
    spare = r * Math.sin(2 * Math.PI * b); return r * Math.cos(2 * Math.PI * b);
  };
  return u;
}

// --- the chirp. Returns {c, s, len, fEnd, duration}: cosine and sine (90 degrees shifted) quadratures,
// starting at sample 0 when the frequency passes F_LOW and ending at ISCO. Unnormalised (peak amplitude ~1).
export function chirp(mc) {
  const T = mc * TSUN, fEnd = fIsco(mc);
  const tau0 = tauOfF(F_LOW, mc), tauEnd = tauOfF(fEnd, mc);
  const len = Math.min(N, Math.floor((tau0 - tauEnd) * FS));
  const c = new Float64Array(len), s = new Float64Array(len);
  const taper = Math.round(0.1 * FS);                 // 0.1 s Hann ramp in, so the sudden start doesn't ring
  const tail = Math.max(2, Math.round(FS / fEnd));    // about one cycle ramp out
  for (let i = 0; i < len; i++) {
    const tau = tau0 - i / FS;
    const f = Math.pow(5 / (256 * tau), 3 / 8) * Math.pow(T, -5 / 8) / Math.PI;
    const phi = -2 * Math.pow(tau / (5 * T), 5 / 8);
    let a = Math.pow(f / F_LOW, 2 / 3);
    if (i < taper) a *= 0.5 - 0.5 * Math.cos(Math.PI * i / taper);
    if (i > len - tail) a *= 0.5 - 0.5 * Math.cos(Math.PI * (len - i) / tail);
    c[i] = a * Math.cos(phi); s[i] = a * Math.sin(phi);
  }
  return { c, s, len, fEnd, duration: len / FS };
}

export const norm = (x) => Math.sqrt(x.reduce((a, v) => a + v * v, 0));

// --- simulated data: whitened noise, plus (optionally) a chirp scaled to a chosen optimal SNR.
// Returns {data, truth}. truth = null when no signal was added.
export function simulate({ seed = 1, mc = null, snr = 0, start = null, phase = null } = {}) {
  const r = rng(seed), data = new Float64Array(N);
  for (let i = 0; i < N; i++) data[i] = r.normal();
  if (!mc || !(snr > 0)) return { data, truth: null };
  const h = chirp(mc);
  const ph = phase ?? r() * 2 * Math.PI;
  const t0 = start ?? Math.floor(r() * Math.max(1, N - h.len - FS * 0.25));
  const wave = new Float64Array(h.len);
  for (let i = 0; i < h.len; i++) wave[i] = Math.cos(ph) * h.c[i] + Math.sin(ph) * h.s[i];
  const k = snr / norm(wave);
  for (let i = 0; i < h.len && t0 + i < N; i++) data[t0 + i] += k * wave[i];
  return { data, truth: { mc, snr, start: t0, merger: (t0 + h.len) / FS, phase: ph, amp: k, wave } };
}

// --- FFT (in place, radix 2). re and im are Float64Array of length a power of two.
export function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

// --- template bank: chirp masses spaced evenly in Mc^(-5/3), which sets how fast the chirp sweeps,
// so neighbouring templates overlap about equally.
export function bank(n = 48) {
  const a = Math.pow(MC_MIN, -5 / 3), b = Math.pow(MC_MAX, -5 / 3);
  return Array.from({ length: n }, (_, i) => Math.pow(a + (b - a) * i / (n - 1), -3 / 5));
}

// Precompute a template's spectrum: complex template z = (c + i s) / ||c||, FFT'd and conjugated.
export function prepare(mc) {
  const h = chirp(mc), nc = norm(h.c);
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < h.len; i++) { re[i] = h.c[i] / nc; im[i] = h.s[i] / nc; }
  fft(re, im);
  return { mc, len: h.len, duration: h.duration, re, im };
}

export function dataSpectrum(data) {
  const re = Float64Array.from(data), im = new Float64Array(N);
  fft(re, im);
  return { re, im };
}

// Matched filter: SNR time series rho(t0) for a template starting at sample t0.
// rho = |sum_t d(t) z*(t - t0)|, using both quadratures, so it doesn't depend on the unknown phase.
// Starts where the template would run off the end of the data are excluded (no wrap-around).
export function filter(D, tpl) {
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let k = 0; k < N; k++) { // D(k) * conj(Z(k)) correlates data with the template
    re[k] = D.re[k] * tpl.re[k] + D.im[k] * tpl.im[k];
    im[k] = D.im[k] * tpl.re[k] - D.re[k] * tpl.im[k];
  }
  fft(re, im, true);
  const valid = N - tpl.len, rho = new Float64Array(valid + 1);
  for (let t = 0; t <= valid; t++) rho[t] = Math.hypot(re[t], im[t]);
  return rho;
}

// Full search: every template, every start time. Returns the loudest trigger and the per-template peaks.
export function search(data, templates) {
  const D = dataSpectrum(data);
  let best = { snr: -1 };
  const peaks = templates.map((tpl) => {
    const rho = filter(D, tpl);
    let m = 0, at = 0;
    for (let t = 0; t < rho.length; t++) if (rho[t] > m) { m = rho[t]; at = t; }
    if (m > best.snr) best = { snr: m, mc: tpl.mc, start: at, merger: (at + tpl.len) / FS, tpl };
    return { mc: tpl.mc, snr: m, start: at };
  });
  return { best, peaks };
}

// Best-fit waveform for display: the template at the trigger, with phase and amplitude from the filter output.
export function bestFitWave(data, mc, start) {
  const h = chirp(mc), nc = norm(h.c);
  let a = 0, b = 0;
  for (let i = 0; i < h.len && start + i < N; i++) { a += data[start + i] * h.c[i]; b += data[start + i] * h.s[i]; }
  a /= nc * nc; b /= nc * nc;
  const w = new Float64Array(N);
  for (let i = 0; i < h.len && start + i < N; i++) w[start + i] = a * h.c[i] + b * h.s[i];
  return w;
}

// Smooth with a running mean, so a reader can see a strong chirp poke out of the noise.
export function smooth(x, w) {
  const out = new Float64Array(x.length), h = Math.floor(w / 2);
  let acc = 0;
  for (let i = 0; i < x.length + h; i++) {
    if (i < x.length) acc += x[i];
    if (i - w >= 0) acc -= x[i - w];
    if (i - h >= 0 && i - h < x.length) out[i - h] = acc / Math.min(w, i + 1);
  }
  return out;
}

// --- statistics helpers
export function quantile(a, p) { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
export function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
