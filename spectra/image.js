// SpaceSift Spectrum Lab: read a spectrum from a picture of a rainbow strip with dark lines.
//
// 1. Find the strip: the widest band of colorful rows (labels and axes are grey or white).
// 2. Measure each column's brightness (median over the strip's rows, so text overlays don't count).
// 3. First guess at the wavelength scale from the colors: hue runs violet -> red along the strip.
// 4. Refine it by sliding and stretching the scale until the dark lines line up with well-known lines
//    (calcium H and K, hydrogen, magnesium, sodium, Earth's oxygen bands...).
// Pure functions on ImageData-like objects {width, height, data}; decoding happens in the page.

export function rgbToHsv(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = 60 * (((g - b) / d) % 6);
    else if (mx === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  return [h, mx ? d / mx : 0, mx];
}

/** Rough wavelength (nm) of a rainbow color from its hue; NaN for hues a rainbow can't pin down (deep red, pink). */
export function hueToNm(h) {
  if (h >= 240 && h <= 300) return 440 - (h - 240);
  if (h >= 180) return h > 300 ? NaN : 490 - ((h - 180) / 60) * 50;
  if (h >= 120) return 510 - ((h - 120) / 60) * 20;
  if (h >= 60) return 580 - ((h - 60) / 60) * 70;
  if (h >= 8) return 645 - (h / 60) * 65;
  return NaN;
}

function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
}

const px = (img, x, y) => {
  const p = 4 * (y * img.width + x), d = img.data;
  return [d[p] / 255, d[p + 1] / 255, d[p + 2] / 255, d[p + 3] / 255];
};

/** Rows and columns of the rainbow strip, or null if the picture has no colorful band. */
export function findStrip(img) {
  const { width: W, height: H } = img;
  const step = Math.max(1, Math.floor(W / 400));
  const colorful = [];
  for (let y = 0; y < H; y++) {
    let n = 0, k = 0;
    for (let x = 0; x < W; x += step) {
      const [r, g, b, a] = px(img, x, y);
      const [, s, v] = rgbToHsv(r, g, b);
      if (a > 0.5 && s > 0.35 && v > 0.12) n++;
      k++;
    }
    colorful.push(n / k > 0.3);
  }
  // Longest run of colorful rows, tolerating short gaps.
  let best = null, start = -1, gap = 0;
  for (let y = 0; y <= H; y++) {
    if (y < H && colorful[y]) { if (start < 0) start = y; gap = 0; }
    else if (start >= 0 && (++gap > 3 || y === H)) {
      const end = y - gap;
      if (!best || end - start > best[1] - best[0]) best = [start, end];
      start = -1; gap = 0;
    }
  }
  if (!best || best[1] - best[0] < 3) return null;
  const pad = Math.floor((best[1] - best[0]) * 0.12);
  const y0 = best[0] + pad, y1 = best[1] - pad;
  // Columns: drop blank margins (white, transparent) at either side.
  const blank = (x) => {
    let n = 0;
    for (let y = y0; y <= y1; y++) { const [r, g, b, a] = px(img, x, y); if (a < 0.5 || (Math.min(r, g, b) > 0.92)) n++; }
    return n > 0.6 * (y1 - y0 + 1);
  };
  let x0 = 0, x1 = W - 1;
  while (x0 < x1 && blank(x0)) x0++;
  while (x1 > x0 && blank(x1)) x1--;
  if (x1 - x0 < 50) return null;
  return { x0, x1, y0, y1 };
}

/** Per-column brightness and hue across the strip. */
export function stripProfile(img, s) {
  const V = [], hue = [], sat = [];
  for (let x = s.x0; x <= s.x1; x++) {
    const vs = [], rs = [], gs = [], bs = [];
    for (let y = s.y0; y <= s.y1; y++) { const [r, g, b] = px(img, x, y); vs.push(Math.max(r, g, b)); rs.push(r); gs.push(g); bs.push(b); }
    const [h, sa] = rgbToHsv(median(rs), median(gs), median(bs));
    V.push(median(vs)); hue.push(h); sat.push(sa);
  }
  return { V, hue, sat };
}

function envelope(y, half) {
  const out = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const w = [];
    for (let j = Math.max(0, i - half); j <= Math.min(y.length - 1, i + half); j++) w.push(y[j]);
    w.sort((a, b) => a - b);
    out[i] = w[Math.floor(0.9 * (w.length - 1))];
  }
  return out;
}

// Lines that stand out in most star spectra and spectrum diagrams [nm, weight].
export const KEY_LINES = [
  [393.4, 1], [396.8, 1], [410.2, 0.5], [430.8, 0.6], [434.0, 0.6], [438.4, 0.3], [447.1, 0.2], [486.1, 0.8],
  [516.7, 0.4], [517.3, 0.4], [518.4, 0.4], [527.0, 0.4], [589.0, 0.7], [589.6, 0.7], [656.3, 1], [686.7, 0.6],
  [759.4, 0.7], [854.2, 0.4], [866.2, 0.4],
];

function corr(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

/**
 * Wavelength scale for the strip: {left, right} in nm at its first and last column. Starts from the colors,
 * then searches nearby scales for the one that best lines up the dark lines with KEY_LINES.
 */
export function calibrate(profile) {
  const n = profile.V.length;
  // 1. Colors: robust straight-line fit of hue-wavelength against column.
  let pts = [];
  for (let i = 0; i < n; i++) {
    const l = hueToNm(profile.hue[i]);
    if (Number.isFinite(l) && profile.sat[i] > 0.35 && profile.V[i] > 0.25) pts.push([i, l]);
  }
  let a = 380, b = 370 / (n - 1), fromColors = false;
  for (let it = 0; it < 3 && pts.length > 20; it++) {
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    let sxy = 0, sxx = 0;
    for (const [x, y] of pts) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }
    b = sxy / sxx; a = my - b * mx; fromColors = true;
    const res = pts.map(([x, y]) => Math.abs(y - (a + b * x))), cut = 2.5 * median(res) + 5;
    pts = pts.filter((p, i) => res[i] < cut);
  }
  let reversed = false;
  if (fromColors && b < 0) { reversed = true; a = a + b * (n - 1); b = -b; } // red on the left
  const V = reversed ? [...profile.V].reverse() : profile.V;
  const L0 = a, S0 = b * (n - 1);
  if (!fromColors || S0 < 50 || S0 > 2000) return { left: 380, right: 750, reversed, method: "default", score: 0 };

  // 2. Lines: dark-line profile versus a comb of known lines.
  const env = envelope(V, Math.max(4, Math.round(n / 60)));
  const d = V.map((v, i) => Math.max(0, 1 - v / (env[i] || 1)));
  if (Math.max(...d) < 0.15) return { left: L0, right: L0 + S0, reversed, method: "colors", score: 0 };
  const FINE0 = 250, FS = 0.1, comb = new Float64Array(10000);
  const sigma = Math.max(1.2, 1.5 * (S0 / n));
  for (const [c, w] of KEY_LINES) for (let k = 0; k < comb.length; k++) comb[k] += w * Math.exp(-0.5 * ((FINE0 + k * FS - c) / sigma) ** 2);
  const sample = new Float64Array(n);
  let best = { score: -Infinity };
  for (let f = 0.7; f <= 1.4001; f += 0.01) {
    const S = S0 * f;
    for (let L = L0 - 90; L <= L0 + 90; L += 1) {
      for (let i = 0; i < n; i++) {
        const k = Math.round((L + (S * i) / (n - 1) - FINE0) / FS);
        sample[i] = k >= 0 && k < comb.length ? comb[k] : 0;
      }
      // Correlation, with a gentle pull toward the color-based guess.
      const score = corr(d, sample) - 0.03 * (((L - L0) / 40) ** 2 + ((f - 1) / 0.25) ** 2);
      if (score > best.score) best = { score, L, S };
    }
  }
  if (best.score < 0.12) return { left: L0, right: L0 + S0, reversed, method: "colors", score: best.score };
  return { left: best.L, right: best.L + best.S, reversed, method: "lines", score: best.score };
}

/** Spectrum {wl, flux} from the strip profile and a wavelength scale. */
export function profileToSpectrum(profile, cal) {
  const V = cal.reversed ? [...profile.V].reverse() : profile.V, n = V.length;
  return { wl: V.map((_, i) => cal.left + ((cal.right - cal.left) * i) / (n - 1)), flux: V.map((v) => Math.max(v, 1e-3)), unit: "a picture" };
}

/** Full pipeline on decoded pixels. */
export function readImageSpectrum(img) {
  const strip = findStrip(img);
  if (!strip) throw new Error("Couldn't find a rainbow strip in this picture. The lab can read pictures of a spectrum (a horizontal band of colors with dark lines), but not graphs or photos of stars; for those, upload the data file.");
  const profile = stripProfile(img, strip);
  const cal = calibrate(profile);
  return { strip, profile, cal, ...profileToSpectrum(profile, cal) };
}
