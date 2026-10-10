// SpaceSift Spectrum Lab: simplified physics for reading what stars and asteroids are made of.
//
// Stars: a Planck (blackbody) continuum multiplied by absorption lines whose strengths follow the
// temperature pattern of the Harvard sequence (O B A F G K M). Real line strengths come from detailed
// model atmospheres; this model keeps the pattern astronomers use to classify a star by eye.
// Asteroids: schematic reflectance shapes for the Bus-DeMeo taxonomy classes (DeMeo et al. 2009),
// built from a red slope, a blue drop-off and Gaussian absorption bands. Teaching models, not fits.

const HP = 6.62607015e-34, CL = 2.99792458e8, KB = 1.380649e-23;

export function grid(lo, hi, step) {
  const n = Math.round((hi - lo) / step) + 1;
  return Array.from({ length: n }, (_, i) => +(lo + i * step).toFixed(6));
}

export function makeRng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.normal = () => {
    let u = 0;
    while (!u) u = next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return next;
}

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  }
  return ys[ys.length - 1];
}

const gauss = (x, c, s) => Math.exp(-0.5 * ((x - c) / s) ** 2);

// ---------------------------------------------------------------- stars

/** Planck spectral radiance B_lambda (W sr^-1 m^-3) at wavelength lamNm (nm), temperature T (K). */
export function planck(lamNm, T) {
  const l = lamNm * 1e-9;
  return (2 * HP * CL * CL) / l ** 5 / Math.expm1((HP * CL) / (l * KB * T));
}

/** Wien's law: wavelength (nm) where a blackbody at T is brightest. */
export const wienPeakNm = (T) => 2.897771955e6 / T;

export const STAR_WL = grid(380, 900, 0.25);

// Each species: where its lines sit (nm), the temperature where they are strongest, and how fast they
// fade away from it (width in log10 T). `metal` lines scale with the star's heavy-element content.
export const SPECIES = [
  { id: "HeII", short: "He II", label: "Ionized helium (He II)", peakT: 42000, w: 0.08, depth: 0.3, fwhm: 1.2,
    nm: [454.1, 468.6, 541.2],
    note: "Ionized helium only forms in extreme heat: the mark of an O star, hotter than about 30,000 K." },
  { id: "HeI", short: "He I", label: "Helium (He I)", peakT: 20000, w: 0.12, depth: 0.3, fwhm: 1.0,
    nm: [447.1, 492.2, 501.6, 587.6, 667.8],
    note: "Neutral helium lines peak in B stars (10,000 to 30,000 K). Cooler stars contain just as much helium, but it is too cold to absorb visible light." },
  { id: "H", short: "H", label: "Hydrogen (Balmer series)", peakT: 9500, w: 0.16, depth: 0.65, fwhm: 2.4,
    nm: [397.0, 410.2, 434.0, 486.1, 656.3],
    note: "Hydrogen lines are strongest in A stars near 10,000 K, where the most hydrogen atoms sit in the energy level that absorbs visible light. Hotter, the atoms lose their electron; cooler, they sit too low to absorb." },
  { id: "CaII", short: "Ca II", label: "Ionized calcium (Ca II)", metal: true, peakT: 5000, w: 0.15, depth: 0.85, fwhm: 1.4,
    nm: [393.4, 396.8, 849.8, 854.2, 866.2],
    note: "The calcium H and K lines (393, 397 nm) are the deepest lines in Sun-like stars. Calcium is rare, but these lines absorb so efficiently that a little goes a long way." },
  { id: "CH", short: "CH", label: "CH molecule (G band)", metal: true, peakT: 5000, w: 0.1, depth: 0.4, fwhm: 2.0,
    nm: [430.5],
    note: "The G band comes from carbon-hydrogen molecules, which only survive in stars cool enough (below about 6,000 K) not to break them apart." },
  { id: "Fe", short: "Fe", label: "Iron (Fe I)", metal: true, peakT: 4500, w: 0.13, depth: 0.4, fwhm: 0.5,
    nm: [382.0, 404.6, 438.4, 495.8, 527.0, 532.8],
    note: "Iron makes thousands of thin lines in G and K stars. Their depth is the standard measure of a star's metal content, written [Fe/H]." },
  { id: "Mg", short: "Mg", label: "Magnesium (Mg I b)", metal: true, peakT: 4500, w: 0.12, depth: 0.5, fwhm: 0.5,
    nm: [516.7, 517.3, 518.4],
    note: "The magnesium b triplet near 517 nm is strong in cool stars; its depth also depends on surface gravity, which separates giants from dwarfs." },
  { id: "Na", short: "Na", label: "Sodium (Na I D)", metal: true, peakT: 3800, w: 0.1, depth: 0.65, fwhm: 0.6,
    nm: [589.0, 589.6],
    note: "The sodium D pair (589 nm, the color of streetlights) grows in cool K and M stars, where sodium keeps its outer electron." },
  { id: "TiO", short: "TiO", label: "Titanium oxide (TiO bands)", metal: true, band: true, depth: 0.55,
    nm: [476.1, 495.4, 516.7, 544.8, 615.9, 705.4, 843.2],
    note: "Titanium oxide molecules carve broad, saw-tooth bands into M stars below about 4,000 K. They are the defining feature of the coolest stars." },
];

/** Central depth (0..1) of a species' lines at temperature T and metallicity feh ([Fe/H], dex). */
export function lineDepth(sp, T, feh = 0) {
  let d0;
  if (sp.band) d0 = sp.depth * Math.min(1, Math.max(0, (4100 - T) / 1100));
  else d0 = sp.depth * gauss(Math.log10(T), Math.log10(sp.peakT), sp.w);
  d0 = Math.min(d0, 0.97);
  // Optical depth scales with abundance, so lines saturate instead of going below zero flux.
  const tau = -Math.log(1 - d0) * (sp.metal ? 10 ** feh : 1);
  return 1 - Math.exp(-tau);
}

/** Every species' line depth at (T, feh), strongest first. */
export function features(T, feh = 0) {
  return SPECIES.map((sp) => ({ ...sp, strength: lineDepth(sp, T, feh) })).sort((a, b) => b.strength - a.strength);
}

/** First index with wl[i] >= x (wl sorted ascending). */
function lowerBound(wl, x) {
  let lo = 0, hi = wl.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (wl[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

/**
 * Model spectrum: {wl, flux, continuum, transmission}, normalized so the brightest point of the continuum
 * is 1. vKms shifts the lines by a radial velocity (km/s), as the star's motion does in real spectra.
 */
export function starSpectrum(T, feh = 0, wl = STAR_WL, vKms = 0) {
  const cont = wl.map((l) => planck(l, T));
  const tau = new Float64Array(wl.length);
  const z = 1 + vKms / 299792.458;
  for (const sp of SPECIES) {
    const d = lineDepth(sp, T, feh);
    if (d < 1e-4) continue;
    const tau0 = -Math.log(1 - d);
    for (const c0 of sp.nm) {
      const c = c0 * z;
      if (sp.band) {
        // Band head: sharp edge on the blue side, absorption fading toward the red.
        for (let i = lowerBound(wl, c - 1); i < wl.length && wl[i] < c + 120; i++) {
          const x = wl[i] - c;
          tau[i] += tau0 * (x < 0 ? 1 + x : Math.exp(-x / 18));
        }
      } else {
        const s = sp.fwhm / 2.3548;
        for (let i = lowerBound(wl, c - 6 * s); i < wl.length && wl[i] < c + 6 * s; i++) {
          tau[i] += tau0 * Math.exp(-0.5 * ((wl[i] - c) / s) ** 2);
        }
      }
    }
  }
  const peak = Math.max(...cont);
  return {
    wl,
    flux: cont.map((v, i) => (v * Math.exp(-tau[i])) / peak),
    continuum: cont.map((v) => v / peak),
    transmission: Array.from(tau, (t) => Math.exp(-t)),
  };
}

export const CLASS_BOUNDS = [
  ["O", 30000, 50000], ["B", 10000, 30000], ["A", 7500, 10000], ["F", 6000, 7500],
  ["G", 5200, 6000], ["K", 3700, 5200], ["M", 2400, 3700],
];

export const CLASS_INFO = {
  O: { color: "blue", text: "Very hot and blue. Ionized helium lines; hydrogen fairly weak. Rare, massive and short-lived." },
  B: { color: "blue-white", text: "Hot and blue-white. Neutral helium lines, with hydrogen growing stronger." },
  A: { color: "white", text: "White. Hydrogen lines are at their strongest; metal lines are faint." },
  F: { color: "yellow-white", text: "Yellow-white. Hydrogen is fading while calcium and metal lines grow." },
  G: { color: "yellow", text: "Yellow, like the Sun. Deep ionized-calcium lines and many metal lines." },
  K: { color: "orange", text: "Orange. Metal lines dominate and molecules start to appear." },
  M: { color: "red", text: "Red and cool. Titanium oxide bands carve up the spectrum. Most stars in the galaxy are M dwarfs." },
};

/** Harvard class with subclass (0 = hot end, 9 = cool end), e.g. "G2" for 5,772 K. */
export function spectralClass(T) {
  const t = Math.min(Math.max(T, 2400), 50000);
  for (const [cls, lo, hi] of CLASS_BOUNDS) {
    if (t >= lo) {
      const sub = Math.floor((10 * (Math.log10(hi) - Math.log10(t))) / (Math.log10(hi) - Math.log10(lo)));
      return { letter: cls, sub: Math.min(9, Math.max(0, sub)), name: cls + Math.min(9, Math.max(0, sub)) };
    }
  }
  return { letter: "M", sub: 9, name: "M9" };
}

/** Temperature from B-V color (Ballesteros 2012, valid for roughly 3,000 to 20,000 K). */
export const tFromBv = (bv) => 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));

/** B-V color from temperature, by inverting tFromBv. Clamped at the edges of its valid range. */
export function bvFromT(T) {
  let lo = -0.4, hi = 2.5;
  if (T >= tFromBv(lo)) return lo;
  if (T <= tFromBv(hi)) return hi;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (tFromBv(mid) > T) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// CIE 1931 color-matching functions, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013).
const g2 = (x, mu, s1, s2) => Math.exp(-0.5 * ((x - mu) / (x < mu ? s1 : s2)) ** 2);
export function cmf(l) {
  return [
    1.056 * g2(l, 599.8, 37.9, 31.0) + 0.362 * g2(l, 442.0, 16.0, 26.7) - 0.065 * g2(l, 501.1, 20.4, 26.2),
    0.821 * g2(l, 568.8, 46.9, 40.5) + 0.286 * g2(l, 530.9, 16.3, 31.1),
    1.217 * g2(l, 437.0, 11.8, 36.0) + 0.681 * g2(l, 459.0, 26.0, 13.8),
  ];
}

const toHex = (rgb) => "#" + rgb.map((v) => Math.round(255 * v).toString(16).padStart(2, "0")).join("");
const gamma = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** The color a blackbody at T appears to the eye, scaled to full brightness. Returns [r,g,b] in 0..1. */
export function starRGB(T) {
  let X = 0, Y = 0, Z = 0;
  for (let l = 380; l <= 780; l += 5) {
    const b = planck(l, T), [x, y, z] = cmf(l);
    X += b * x; Y += b * y; Z += b * z;
  }
  const lin = [
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.204 * Y + 1.057 * Z,
  ].map((v) => Math.max(v, 0));
  const m = Math.max(...lin);
  return lin.map((v) => gamma(v / m));
}
export const starHex = (T) => toHex(starRGB(T));

/** Approximate color of a single wavelength (nm), for drawing a rainbow strip (Bruton's approximation). */
export function wavelengthRGB(l) {
  let r = 0, g = 0, b = 0;
  if (l >= 380 && l < 440) { r = (440 - l) / 60; b = 1; }
  else if (l < 490) { g = (l - 440) / 50; b = 1; }
  else if (l < 510) { g = 1; b = (510 - l) / 20; }
  else if (l < 580) { r = (l - 510) / 70; g = 1; }
  else if (l < 645) { r = 1; g = (645 - l) / 65; }
  else if (l <= 750) { r = 1; }
  let f = 0;
  if (l >= 380 && l < 420) f = 0.3 + (0.7 * (l - 380)) / 40;
  else if (l >= 420 && l <= 700) f = 1;
  else if (l > 700 && l <= 750) f = 0.3 + (0.7 * (750 - l)) / 50;
  return [r, g, b].map((c) => (c * f) ** 0.8);
}

export function mysteryStar(rng) {
  const [cls, lo, hi] = CLASS_BOUNDS[Math.floor(rng() * CLASS_BOUNDS.length)];
  const top = cls === "O" ? 42000 : hi;
  const T = 10 ** (Math.log10(lo) + rng() * (Math.log10(top) - Math.log10(lo)));
  return { T: Math.round(T), feh: Math.max(-1.5, Math.min(0.5, 0.25 * rng.normal())) };
}

export function addNoise(arr, rng, sigma) {
  return arr.map((v) => v + sigma * rng.normal());
}

export const REAL_STARS = [
  { name: "Sun", T: 5772, feh: 0, note: "Our star: a G2 dwarf." },
  { name: "Sirius A", T: 9940, feh: 0.4, note: "Brightest star in the night sky; an A star with unusually strong metal lines." },
  { name: "Vega", T: 9600, feh: -0.5, note: "The original zero point of the magnitude scale." },
  { name: "Rigel", T: 12100, feh: -0.1, note: "Blue supergiant in Orion." },
  { name: "Spica", T: 22400, feh: 0, note: "Hot B star in Virgo." },
  { name: "ζ Puppis", T: 40000, feh: 0, note: "One of the hottest stars visible to the naked eye (O4)." },
  { name: "Procyon A", T: 6530, feh: 0, note: "F star 11 light-years away." },
  { name: "Arcturus", T: 4290, feh: -0.5, note: "Orange giant; an old star with a third of the Sun's metals." },
  { name: "Betelgeuse", T: 3600, feh: 0.05, note: "Red supergiant in Orion." },
  { name: "TRAPPIST-1", T: 2570, feh: 0.04, note: "Tiny M8 dwarf with seven Earth-sized planets." },
];

// ---------------------------------------------------------------- asteroids

export const AST_WL = grid(0.45, 2.45, 0.01);

// slope: reflectance gained per micron; uv: fractional drop at 0.45 um; bands: [center um, sigma um, depth].
export const ASTEROID_CLASSES = {
  S: { slope: 0.35, uv: 0.2, bands: [[0.95, 0.09, 0.15], [1.95, 0.25, 0.07]], name: "S-type (stony)",
       composition: "Silicate rock: the minerals olivine and pyroxene, mixed with some iron-nickel metal.",
       analog: "Ordinary chondrite meteorites, confirmed by Hayabusa's sample of Itokawa.", albedo: "Moderate (about 0.25)",
       density: "Solid rock about 3.3 g/cm³; as rubble piles, closer to 2.", color: "#fbbf24" },
  Q: { slope: 0.18, uv: 0.22, bands: [[0.98, 0.1, 0.22], [1.95, 0.25, 0.08]], name: "Q-type (fresh stony)",
       composition: "The same rock as S-types, but a fresh surface not yet reddened by space weathering.",
       analog: "Ordinary chondrites, the most common meteorites to fall on Earth.", albedo: "Moderate (about 0.3)",
       density: "As S-types.", color: "#fde68a" },
  C: { slope: 0.03, uv: 0.1, bands: [], name: "C-type (carbonaceous)",
       composition: "Dark, carbon-rich material: clay minerals that formed in water, plus organic compounds.",
       analog: "Carbonaceous chondrites; Hayabusa2 returned a sample of Ryugu.", albedo: "Very dark (about 0.06)",
       density: "Low, about 1.2 to 1.9 g/cm³: porous, often rubble piles.", color: "#94a3b8" },
  B: { slope: -0.08, uv: 0.03, bands: [], name: "B-type (blue carbonaceous)",
       composition: "Dark carbon-rich material like C-types, with a slightly blue spectrum.",
       analog: "Carbonaceous chondrites; OSIRIS-REx returned a sample of Bennu.", albedo: "Very dark (about 0.05)",
       density: "Low, about 1.2 g/cm³ for Bennu: a loosely held rubble pile.", color: "#7aa2ff" },
  X: { slope: 0.18, uv: 0.05, bands: [], name: "X-type (featureless)",
       composition: "A featureless spectrum that can mean metal, enstatite or primitive dark material. Brightness (albedo) is needed to tell them apart.",
       analog: "Iron meteorites (M), enstatite achondrites (E) or primitive carbon-rich material (P).", albedo: "Anything: dark P, medium M, bright E",
       density: "From about 1.5 g/cm³ (P) to 4 g/cm³ or more (metal-rich M).", color: "#c4b5fd" },
  D: { slope: 0.75, uv: 0.03, bands: [], name: "D-type (very red)",
       composition: "Very dark and red: organic-rich material and ice-bearing silicates from the outer solar system.",
       analog: "Possibly the Tagish Lake meteorite.", albedo: "Very dark (about 0.05)",
       density: "Probably low and icy.", color: "#f87171" },
  V: { slope: 0.25, uv: 0.3, bands: [[0.92, 0.07, 0.45], [1.92, 0.2, 0.3]], name: "V-type (basaltic)",
       composition: "Basalt: pyroxene-rich volcanic rock, chipped off the crust of the large asteroid Vesta.",
       analog: "HED meteorites (howardites, eucrites, diogenites), studied up close by the Dawn spacecraft.", albedo: "Bright (about 0.35)",
       density: "Solid basalt about 3 g/cm³.", color: "#4ade80" },
  A: { slope: 0.6, uv: 0.25, bands: [[1.05, 0.17, 0.35]], name: "A-type (olivine)",
       composition: "Almost pure olivine, possibly the mantle of a body that melted and later shattered.",
       analog: "Brachinite and pallasite meteorites.", albedo: "Moderate (about 0.2)",
       density: "About 3.3 g/cm³.", color: "#fb923c" },
};

/** Reflectance spectrum for a parameter set, normalized to 1 at 0.55 um. */
export function reflectance(p, wl = AST_WL) {
  const raw = (l) => {
    let r = (1 + p.slope * (l - 0.55)) * (1 - p.uv * Math.exp(-(l - 0.45) / 0.08));
    for (const [c, s, d] of p.bands) r *= 1 - d * gauss(l, c, s);
    return r;
  };
  const n = raw(0.55);
  return wl.map((l) => raw(l) / n);
}

/** Classes ranked by RMS difference from a reflectance spectrum (on AST_WL). */
export function classifySpectrum(R, wl = AST_WL) {
  return Object.keys(ASTEROID_CLASSES)
    .map((cls) => {
      const t = reflectance(ASTEROID_CLASSES[cls], wl);
      let s = 0;
      for (let i = 0; i < R.length; i++) s += (R[i] - t[i]) ** 2;
      return { cls, rms: Math.sqrt(s / R.length) };
    })
    .sort((a, b) => a.rms - b.rms);
}

// Approximate solar colors in the SDSS system; an asteroid's color is the Sun's plus its reflectance.
const SUN = { gr: 0.44, ri: 0.11, iz: 0.03 };
const FILTERS = { g: 0.477, r: 0.623, i: 0.763, z: 0.913 };

/** SDSS-like colors an asteroid with reflectance R would show (filter centers only: a rough estimate). */
export function sdssColors(R, wl = AST_WL) {
  const f = Object.fromEntries(Object.entries(FILTERS).map(([k, l]) => [k, interp(l, wl, R)]));
  const c = {
    gr: SUN.gr - 2.5 * Math.log10(f.g / f.r),
    ri: SUN.ri - 2.5 * Math.log10(f.r / f.i),
    iz: SUN.iz - 2.5 * Math.log10(f.i / f.z),
  };
  return { ...c, ...colorRule(c) };
}

/**
 * Rough composition group from SDSS colors (Ivezic et al. 2001): the a* color separates carbon-rich
 * (a* < 0) from stony (a* > 0) asteroids, a very negative i-z marks the deep 0.9 um band of basalt, and
 * a red asteroid with no dip at all (i-z > 0) is D-like. Boundaries are approximate.
 */
export function colorRule({ gr, ri, iz }) {
  const astar = 0.89 * gr + 0.45 * ri - 0.57;
  let group;
  if (astar < 0) group = "C";
  else if (iz < -0.2) group = "V";
  else if (iz > 0.05) group = "D";
  else group = "S";
  return { astar, group };
}

export const COLOR_GROUPS = {
  C: "Carbon-rich or featureless (C, B, X family): the color is flat, so the surface is likely dark carbonaceous material.",
  S: "Stony (S, Q family): reddish with a mild dip near 0.9 µm, the signature of silicate rock.",
  D: "Very red and featureless (D family): red like stony asteroids but with no silicate dip, pointing to dark organic-rich material.",
  V: "Basaltic (V-type): a very deep dip at 0.9 µm makes i-z strongly negative, the signature of pyroxene from Vesta's crust.",
};

export function mysteryAsteroid(rng) {
  const keys = Object.keys(ASTEROID_CLASSES);
  const cls = keys[Math.floor(rng() * keys.length)];
  const p = ASTEROID_CLASSES[cls];
  const q = {
    slope: p.slope + 0.06 * rng.normal(),
    uv: Math.max(0, p.uv * (1 + 0.2 * rng.normal())),
    bands: p.bands.map(([c, s, d]) => [c + 0.01 * rng.normal(), s, Math.max(0, d * (1 + 0.15 * rng.normal()))]),
  };
  return { cls, R: addNoise(reflectance(q), rng, 0.012) };
}

export const REAL_ASTEROIDS = [
  { name: "Didymos & Dimorphos", cls: "S", note: "Targets of NASA's DART impact in 2022, which shortened Dimorphos's orbit by about 32 minutes." },
  { name: "Itokawa", cls: "S", note: "Hayabusa's sample proved S-types are the parents of ordinary chondrite meteorites." },
  { name: "Eros", cls: "S", note: "First asteroid ever orbited (2000) and landed on (2001), by NEAR Shoemaker." },
  { name: "Apophis", cls: "Q", note: "Classed Sq, between S and Q. It passes within about 32,000 km of Earth's surface in April 2029." },
  { name: "Bennu", cls: "B", note: "OSIRIS-REx brought back 122 g in 2023; one of the most hazardous known asteroids." },
  { name: "Ryugu", cls: "C", note: "Classed Cb; Hayabusa2 returned 5.4 g of it in 2020." },
  { name: "Vesta", cls: "V", note: "Second-largest asteroid; source of the HED meteorites." },
  { name: "Psyche", cls: "X", note: "Classed M (metal-rich X-type); NASA's Psyche spacecraft arrives in 2029." },
];

// ---------------------------------------------------------------- plain-language helpers

/** Everyday names and one-line meanings for each absorber, plus the line used to label it. */
export const PLAIN = {
  HeII: { name: "Helium (super-hot)", labelNm: 468.6, why: "Only appears above about 30,000 K, in the hottest stars." },
  HeI: { name: "Helium", labelNm: 587.6, why: "Visible only in hot blue stars; cooler stars have helium but it hides." },
  H: { name: "Hydrogen", labelNm: 656.3, why: "The most common element. Its lines are darkest in white stars near 10,000 K." },
  CaII: { name: "Calcium", labelNm: 393.4, why: "Makes the darkest lines in Sun-like stars, at the violet end." },
  CH: { name: "Carbon (CH)", labelNm: 430.5, why: "Carbon bonded to hydrogen; these molecules survive only in cooler stars." },
  Fe: { name: "Iron", labelNm: 527.0, why: "Astronomers measure a star's metal content from its iron lines." },
  Mg: { name: "Magnesium", labelNm: 517.3, why: "A close trio of green lines, strong in cooler stars." },
  Na: { name: "Sodium", labelNm: 589.3, why: "The yellow streetlight color. Strong in orange and red stars." },
  TiO: { name: "Titanium oxide", labelNm: 705.4, why: "Molecules that carve broad bands into the coolest, reddest stars." },
};

export const ASTEROID_PLAIN = {
  S: "stony rock", Q: "fresh stony rock", C: "dark carbon-rich rock", B: "dark carbon-rich rock",
  X: "metal or dark rock (needs more data)", D: "very dark, organic-rich material", V: "volcanic basalt", A: "olivine crystal rock",
};

/**
 * The three clues astronomers read off a reflectance spectrum: overall color (slope from 0.55 to 1.6 um)
 * and the depth of the 1 um and 2 um absorption dips relative to a straight line across each dip.
 */
export function clues(R, wl = AST_WL) {
  const at = (l) => interp(l, wl, R);
  const dip = (a, b, l0, l1) => {
    const lo = at(l0), hi = at(l1);
    let d = 0;
    for (let i = 0; i < wl.length; i++) {
      if (wl[i] < a || wl[i] > b) continue;
      const line = lo + ((hi - lo) * (wl[i] - l0)) / (l1 - l0);
      d = Math.max(d, 1 - R[i] / line);
    }
    return d;
  };
  return {
    slope: (at(1.6) - 1) / 1.05,
    dip1: dip(0.8, 1.2, 0.72, 1.5),
    dip2: dip(1.7, 2.2, 1.5, 2.4),
  };
}
