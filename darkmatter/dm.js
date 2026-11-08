// Dark Matter Detective engine: rotation-curve models fitted to SPARC galaxies. Pure functions, no DOM.
//
// Data: SPARC (Lelli, McGaugh & Schombert 2016, AJ 152, 157). Each point gives the measured circular speed
// Vobs and the speeds the visible components alone would produce: gas, stellar disk and bulge. The stellar
// ones are for a mass-to-light ratio Upsilon = 1 at 3.6 microns, so they scale as sqrt(Upsilon).
//
// Visible-matter prediction:  Vbar^2 = Vgas|Vgas| + Yd Vdisk|Vdisk| + Yb Vbul|Vbul|     (Yb = 1.4 Yd)
// Dark halo (pseudo-isothermal sphere, a cored profile; e.g. Begeman, Broeils & Sanders 1991):
//   rho(r) = rho0 / (1 + (r/rc)^2),  Vhalo^2 = 4 pi G rho0 rc^2 [1 - (rc/r) arctan(r/rc)]
// Total: V^2 = Vbar^2 + Vhalo^2.

export const G = 4.30091e-6;          // kpc (km/s)^2 / Msun
export const Y_DISK = 0.5, Y_BULGE = 0.7; // stellar-population values at 3.6 um adopted by SPARC (Lelli et al. 2016)
export const Y_MAX_PLAUSIBLE = 1.0;   // well above what stellar populations allow at 3.6 um (~0.2-0.8)
export const ACC = 1e6 / 3.0857e19;   // (km/s)^2/kpc -> m/s^2

const sq = (v) => v * Math.abs(v);
export const vbar2 = (row, yd, yb = 1.4 * yd) => sq(row[3]) + yd * sq(row[4]) + yb * sq(row[5]);
export const vhalo2 = (r, rho0, rc) => 4 * Math.PI * G * rho0 * rc * rc * (1 - (rc / r) * Math.atan(r / rc));
export const vbar = (row, yd) => Math.sqrt(Math.max(0, vbar2(row, yd)));
export const vtot = (row, yd, rho0, rc) => Math.sqrt(Math.max(0, vbar2(row, yd) + vhalo2(row[0], rho0, rc)));
export const enclosedMass = (r, v) => v * v * r / G;  // Msun inside radius r for circular speed v

export const quality = (g) => g.Q < 3 && g.inc >= 30;  // the standard SPARC cut (McGaugh, Lelli & Schombert 2016)

function chi2(g, model) {
  let c = 0;
  for (const row of g.rows) { const d = (row[1] - model(row)) / row[2]; c += d * d; }
  return c;
}
const logspace = (a, b, n) => Array.from({ length: n }, (_, i) => Math.pow(10, Math.log10(a) + (Math.log10(b) - Math.log10(a)) * i / (n - 1)));

// Visible matter only, with the stellar mass-to-light ratio free: the most generous case for "no dark matter".
export function fitVisible(g) {
  let best = { chi2: Infinity };
  for (const yd of logspace(0.02, 10, 400)) {
    const c = chi2(g, (row) => vbar(row, yd));
    if (c < best.chi2) best = { yd, chi2: c };
  }
  const dof = Math.max(1, g.rows.length - 1);
  return { ...best, dof, rchi2: best.chi2 / dof };
}

// Visible matter at the stellar-population mass-to-light ratio, plus a dark halo with free rho0 and rc.
export function fitHalo(g, yd = Y_DISK) {
  let best = { chi2: Infinity };
  const scan = (rhos, rcs) => {
    for (const rho0 of rhos) for (const rc of rcs) {
      const c = chi2(g, (row) => vtot(row, yd, rho0, rc));
      if (c < best.chi2) best = { rho0, rc, chi2: c };
    }
  };
  scan(logspace(1e5, 1e11, 61), logspace(0.05, 200, 61));
  for (let k = 0; k < 2; k++) { // refine around the best grid point
    const f = k ? 1.12 : 1.5;
    scan(logspace(best.rho0 / f, best.rho0 * f, 21), logspace(best.rc / f, best.rc * f, 21));
  }
  const dof = Math.max(1, g.rows.length - 2);
  return { ...best, yd, dof, rchi2: best.chi2 / dof };
}

// Mass budget at the outermost measured point.
export function budget(g, yd = Y_DISK) {
  const last = g.rows.at(-1), r = last[0];
  const total = enclosedMass(r, last[1]), visible = enclosedMass(r, vbar(last, yd));
  return { r, total, visible, ratio: total / Math.max(visible, 1), darkFrac: 1 - visible / total };
}

// Radial acceleration relation points: observed vs visible-matter centripetal acceleration (m/s^2).
export function rar(g, yd = Y_DISK) {
  return g.rows.filter((row) => row[0] > 0 && vbar2(row, yd) > 0).map((row) => ({ gbar: vbar2(row, yd) / row[0] * ACC, gobs: row[1] * row[1] / row[0] * ACC }));
}
// McGaugh, Lelli & Schombert 2016 (PRL 117, 201101) fit: gobs = gbar / (1 - exp(-sqrt(gbar/g†))), g† = 1.20e-10 m/s^2.
export const G_DAGGER = 1.2e-10;
export const rarFit = (gbar) => gbar / (1 - Math.exp(-Math.sqrt(gbar / G_DAGGER)));

// Everything step 4 needs, for every quality galaxy.
export function survey(galaxies) {
  return galaxies.filter(quality).map((g) => {
    const vis = fitVisible(g), halo = fitHalo(g), b = budget(g);
    return { name: g.name, T: g.T, Vflat: g.Vflat, n: g.rows.length, vis, halo, budget: b };
  });
}

export function quantile(a, p) { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; }
