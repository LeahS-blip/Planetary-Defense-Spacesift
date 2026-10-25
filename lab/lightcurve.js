// Read a visitor's light curve (brightness over time) from a file: a CSV/text table, or a Kepler / TESS / K2
// FITS light-curve file, optionally gzipped. Returns time in days (starting at 0) and flux relative to its median.

const MAX_POINTS = 150000;

function ascii(u8, a, n) { let s = ""; for (let i = a; i < a + n && i < u8.length; i++) s += String.fromCharCode(u8[i]); return s; }
function cardValue(s) {
  const str = /^\s*'((?:[^']|'')*)'/.exec(s);
  if (str) return str[1].replace(/''/g, "'").trim();
  const v = s.split("/")[0].trim(), n = Number(v.replace(/D/i, "E"));
  return v === "T" ? true : v === "F" ? false : Number.isNaN(n) ? v : n;
}
const TSIZE = { L: 1, B: 1, I: 2, J: 4, K: 8, A: 1, E: 4, D: 8, C: 8, M: 16, P: 8, Q: 16 };

/** Light curve from a FITS file: the first binary table with a time column and a flux column. */
export function parseLightCurveFITS(buf) {
  const u8 = new Uint8Array(buf), dv = new DataView(buf), hdus = [];
  let off = 0;
  while (off + 2880 <= u8.length) {
    const h = {}; let done = false;
    while (!done && off + 2880 <= u8.length) {
      for (let i = 0; i < 36; i++) { const card = ascii(u8, off + 80 * i, 80), key = card.slice(0, 8).trim(); if (key === "END") { done = true; break; } if (card.slice(8, 10) === "= ") h[key] = cardValue(card.slice(10)); }
      off += 2880;
    }
    if (!done) break;
    let size = 0;
    if (h.NAXIS > 0) { size = Math.abs(h.BITPIX) / 8; for (let i = 1; i <= h.NAXIS; i++) size *= h["NAXIS" + i]; }
    size = (size + (h.PCOUNT || 0)) * (h.GCOUNT || 1);
    hdus.push({ h, off }); off += Math.ceil(size / 2880) * 2880;
  }
  if (!hdus.length || !("SIMPLE" in hdus[0].h)) throw new Error("This doesn't look like a FITS file.");
  const object = String(hdus[0].h.OBJECT || hdus[0].h.TICID || hdus[0].h.KEPLERID || "").trim();
  for (const { h, off: o } of hdus) {
    if (h.XTENSION !== "BINTABLE") continue;
    const cols = []; let pos = 0;
    for (let i = 1; i <= h.TFIELDS; i++) {
      const m = /^(\d*)([LXBIJKAEDCMPQ])/.exec(String(h["TFORM" + i]).trim()); if (!m) { pos = -1; break; }
      const rep = m[1] === "" ? 1 : +m[1], t = m[2];
      cols.push({ name: String(h["TTYPE" + i] || "").trim().toLowerCase(), t, rep, pos, scale: h["TSCAL" + i] ?? 1, zero: h["TZERO" + i] ?? 0 });
      pos += t === "X" ? Math.ceil(rep / 8) : t === "P" || t === "Q" ? TSIZE[t] : rep * TSIZE[t];
    }
    if (pos < 0) continue;
    const ok = (c) => c && "BIJKED".includes(c.t) && c.rep === 1;
    const tc = cols.find((c) => c.name === "time") || cols.find((c) => /^(bjd|btjd|hjd|jd|mjd|t)$/.test(c.name));
    const fc = ["pdcsap_flux", "sap_flux", "kspsap_flux", "flux", "det_flux", "fcor"].map((n) => cols.find((c) => c.name === n)).find(ok);
    const qc = cols.find((c) => c.name === "quality" || c.name === "sap_quality");
    if (!ok(tc) || !fc) continue;
    const read = (c, row) => { const p = o + row * h.NAXIS1 + c.pos;
      const v = { E: () => dv.getFloat32(p), D: () => dv.getFloat64(p), J: () => dv.getInt32(p), I: () => dv.getInt16(p), K: () => Number(dv.getBigInt64(p)), B: () => dv.getUint8(p) }[c.t]();
      return v * c.scale + c.zero; };
    const time = [], flux = []; let flagged = 0;
    for (let r = 0; r < h.NAXIS2; r++) {
      if (qc && ok(qc) && read(qc, r) !== 0) { flagged++; continue; } // the mission's own quality flags
      time.push(read(tc, r)); flux.push(read(fc, r));
    }
    return { time, flux, flagged, object, source: `FITS columns "${tc.name}" and "${fc.name}"` };
  }
  throw new Error("Found no light curve in this FITS file (it needs a table with TIME and a flux column such as PDCSAP_FLUX).");
}

/** Light curve from text: columns of numbers, separated by commas, tabs, semicolons or spaces, with an optional header. */
export function parseLightCurveText(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !/^[#%!]/.test(l) && !/^\\/.test(l));
  if (!lines.length) throw new Error("The file is empty.");
  const split = (l) => l.split(/[,\t;]|\s+/).map((s) => s.trim()).filter((s) => s !== "");
  const isNum = (s) => /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s);
  let header = null, start = 0;
  if (!split(lines[0]).every(isNum)) { header = split(lines[0]).map((s) => s.toLowerCase().replace(/["']/g, "")); start = 1; }
  let ti = 0, fi = 1;
  if (header) {
    const t = header.findIndex((n) => /^(time|bjd|btjd|hjd|jd|mjd|t|days?|date)/.test(n));
    const f = ["pdcsap_flux", "sap_flux", "flux", "rel_flux", "relative_flux", "norm_flux", "brightness", "mag", "magnitude"].map((n) => header.indexOf(n)).find((i) => i >= 0)
      ?? header.findIndex((n, i) => i !== t && /flux|bright|mag/.test(n));
    if (t >= 0) ti = t;
    if (f !== undefined && f >= 0) fi = f; else fi = ti === 0 ? 1 : 0;
  }
  const mag = header && /mag/.test(header[fi] || "");
  const time = [], flux = [];
  for (let i = start; i < lines.length; i++) {
    const c = split(lines[i]); if (c.length < 2) continue;
    const t = Number(c[ti]), f = Number(c[fi]);
    if (Number.isFinite(t) && Number.isFinite(f)) { time.push(t); flux.push(mag ? Math.pow(10, -0.4 * f) : f); }
  }
  if (time.length < 2) throw new Error("Couldn't find two columns of numbers (time and brightness) in this file.");
  return { time, flux, flagged: 0, object: "", source: header ? `columns "${header[ti]}" and "${header[fi]}"${mag ? " (magnitudes, converted to flux)" : ""}` : "the first two columns" };
}

/** Clean up and normalise: drop bad points, sort, start time at 0, flux relative to its median, bin very long files. */
export function normalize(raw) {
  let pts = raw.time.map((t, i) => [t, raw.flux[i]]).filter(([t, f]) => Number.isFinite(t) && Number.isFinite(f) && f !== 0);
  if (pts.length < 200) throw new Error(`Only ${pts.length} usable points. A transit search needs a few hundred or more.`);
  pts.sort((a, b) => a[0] - b[0]);
  const t0 = pts[0][0];
  // Time units: big numbers are Julian dates (days); a span of millions suggests seconds.
  let scale = 1, note = "";
  const span0 = pts[pts.length - 1][0] - t0;
  if (span0 > 1e5) { scale = 1 / 86400; note = "Times looked like seconds, so they were converted to days."; }
  let time = pts.map((p) => (p[0] - t0) * scale), flux = pts.map((p) => p[1]);
  const med = [...flux].sort((a, b) => a - b)[flux.length >> 1];
  if (med < 0) throw new Error("The brightness column is negative. Use flux (or magnitudes with a 'mag' header), not differences.");
  flux = flux.map((f) => f / med);
  let binned = false;
  if (time.length > MAX_POINTS) { // keep the browser responsive: average into equal time bins
    const span = time[time.length - 1], nb = MAX_POINTS, bt = [], bf = [];
    let j = 0;
    for (let b = 0; b < nb && j < time.length; b++) { const hi = ((b + 1) / nb) * span; let s = 0, n = 0, st = 0; while (j < time.length && time[j] <= hi) { s += flux[j]; st += time[j]; n++; j++; } if (n) { bt.push(st / n); bf.push(s / n); } }
    time = bt; flux = bf; binned = true;
  }
  const span = time[time.length - 1];
  const steps = []; for (let i = 1; i < Math.min(time.length, 4000); i++) steps.push(time[i] - time[i - 1]);
  steps.sort((a, b) => a - b);
  const cadenceMin = steps[steps.length >> 1] * 1440;
  if (span < 3) throw new Error(`The data covers only ${span.toFixed(1)} days. A transit search needs at least two dips, so a few days or more.`);
  return { time, flux, spanD: span, n: time.length, cadenceMin, binned, note };
}

async function gunzip(buf) {
  if (typeof DecompressionStream === "undefined") throw new Error("This browser can't open .gz files; unzip it first.");
  return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

/** Read a File: returns { time, flux, spanD, n, cadenceMin, label, source, flagged, binned, note }. */
export async function readLightCurve(file) {
  let buf = await file.arrayBuffer(), u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b) { buf = await gunzip(buf); u8 = new Uint8Array(buf); }
  const head = ascii(u8, 0, 512);
  if (head.startsWith("%PDF") || head.startsWith("PK\u0003\u0004")) throw new Error("Please upload a CSV/text table or a FITS light-curve file, not a PDF or spreadsheet.");
  if (u8[0] === 0x89 || (u8[0] === 0xff && u8[1] === 0xd8)) throw new Error("This is a picture. The Transit Lab needs the numbers: a table of time and brightness, or a FITS light curve.");
  const raw = head.startsWith("SIMPLE") ? parseLightCurveFITS(buf) : parseLightCurveText(new TextDecoder().decode(u8));
  const lc = normalize(raw);
  return { ...lc, label: raw.object || file.name.replace(/\.(gz|fits?|fts|csv|txt|dat|tsv)+$/i, ""), source: raw.source, flagged: raw.flagged };
}
