// Runs one trial off the main thread, so the page stays responsive during a search.
import * as E from "./engine.js";

const starCache = new Map();

async function loadReal(kic) {
  if (!starCache.has(kic)) {
    const r = await fetch(new URL(`data/stars/${kic}.json`, self.location.href));
    if (!r.ok) throw new Error(`could not load KIC ${kic} (${r.status})`);
    starCache.set(kic, await r.json());
  }
  return starCache.get(kic);
}

function thin(x, y, n) {
  // Keep the max of y in n chunks, so periodogram peaks survive.
  if (x.length <= n) return [Array.from(x), Array.from(y)];
  const ox = [], oy = [], step = x.length / n;
  for (let c = 0; c < n; c++) {
    let best = Math.floor(c * step);
    for (let i = Math.floor(c * step); i < Math.min(x.length, Math.floor((c + 1) * step)); i++) if (y[i] > y[best]) best = i;
    ox.push(x[best]); oy.push(y[best]);
  }
  return [ox, oy];
}

function binned(x, y, n) {
  if (!x.length) return [[], []];
  let lo = Infinity, hi = -Infinity;
  for (const v of x) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const bins = Array.from({ length: n }, () => []);
  x.forEach((v, i) => bins[Math.min(n - 1, Math.floor(((v - lo) / (hi - lo || 1)) * n))].push(y[i]));
  const bx = [], by = [];
  bins.forEach((b, i) => { if (b.length) { bx.push(lo + ((i + 0.5) * (hi - lo)) / n); by.push(E.median(b)); } });
  return [bx, by];
}

const ppm = (a) => Array.from(a, (v) => Math.round((v - 1) * 1e7) / 10);

// Protocol (the page runs a pool of these workers):
//   prepare {req}           -> plant, clean, set up the search; returns the binned data + period grid
//   power   {t, f, periods} -> BLS power for one slice of the period grid (any worker)
//   finish  {req, powers}   -> combine, grade, and build the plot data (the preparing worker)
const prepared = new Map();

self.onmessage = async (ev) => {
  const { id, type } = ev.data;
  try {
    if (type === "power") {
      const { t, f, periods, durationsHr } = ev.data;
      const p = E.blsPower(t, f, periods, durationsHr);
      return self.postMessage({ id, ok: true, result: p }, [p.power.buffer, p.bestDur.buffer, p.bestStart.buffer, p.bestNb.buffer, p.bestDepth.buffer]);
    }
    if (type === "prepare") {
      const out = await prepare(ev.data.req);
      prepared.set(id, out);
      const s = out.prep.setup;
      return self.postMessage({ id, ok: true, result: { t: s.t, f: s.f, periods: s.periods, durationsHr: s.durationsHr } });
    }
    if (type === "finish") {
      const { prep, lc, label, started } = prepared.get(ev.data.prepId);
      prepared.delete(ev.data.prepId);
      const r = E.finishTrial(prep, ev.data.powers, ev.data.req);
      return self.postMessage({ id, ok: true, result: payload(r, lc, label, (Date.now() - started) / 1000) });
    }
    throw new Error(`unknown message type ${type}`);
  } catch (e) {
    self.postMessage({ id, ok: false, error: String((e && e.stack) || e) });
  }
};

async function prepare(req) {
  const started = Date.now();
  let lc, star = { ...E.DEFAULT_STAR }, label;
  if (req.source === "kepler") {
    const s = await loadReal(req.kic);
    const time = Float64Array.from(s.time), flux = Float64Array.from(s.flux_ppm, (v) => 1 + v * 1e-6);
    const rng = E.makeRng(req.seed + 1), sigma = s.sigma_ppm * 1e-6;
    lc = { time, flux, noise: Float64Array.from(time, () => 1 + sigma * rng.normal()), sigma };
    star = { ...star, radius: s.radius, mass: s.mass };
    label = `KIC ${s.kic} · ${s.label}`;
  } else {
    const s = E.syntheticStar({ baseline: req.baselineD, noisePpm: req.noisePpm, kind: req.varKind,
                                periodD: req.varPeriodD, ampPpm: req.varAmpPpm, seed: req.seed });
    lc = { time: s.time, flux: s.flux, noise: s.noiseOnly, sigma: s.sigma };
    label = "Simulated Sun-like star";
  }
  return { prep: E.prepareTrial(lc, star, req), lc, label, started };
}

function payload(r, lc, label, seconds) {
  const c = r.candidate;
  const phaseH = [], foldF = [];
  const win = Math.max(3 * c.duration * 24, 6);
  for (let i = 0; i < r.t.length; i++) {
    const h = (((((r.t[i] - c.t0 + c.period / 2) % c.period) + c.period) % c.period) - c.period / 2) * 24;
    if (Math.abs(h) < win) { phaseH.push(h); foldF.push(Math.round((r.f[i] - 1) * 1e7) / 10); }
  }
  const [bh, bf] = binned(phaseH, foldF, 80);
  const [pp, ps] = thin(r.periods, r.sde, 3000);
  return {
    label, status: r.status, seconds, sigmaPpm: Math.round(lc.sigma * 1e7) / 10,
    candidate: c, injection: r.inj,
    lc: { t: Array.from(lc.time), raw: ppm(r.raw), model: ppm(r.model), flatT: Array.from(r.t), flat: ppm(r.f) },
    periodogram: { period: pp, sde: ps },
    fold: { phaseH: phaseH.slice(0, 8000), flux: foldF.slice(0, 8000), binH: bh, binFlux: bf },
  };
}
