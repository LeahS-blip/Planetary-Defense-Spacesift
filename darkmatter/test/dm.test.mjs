// Checks the Dark Matter Detective engine on the real SPARC data. Run from the repo root:  node darkmatter/test/dm.test.mjs
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import * as D from "../dm.js";

const data = JSON.parse(readFileSync(new URL("../data/sparc.json", import.meta.url)));
const gal = (n) => data.galaxies.find((g) => g.name === n);
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

assert.equal(data.galaxies.length, 175);
assert.equal(data.galaxies.filter(D.quality).length, 153, "153 galaxies pass Q<3 and inc>=30, as in McGaugh et al. 2016");

// Physics: the Milky Way at the Sun (8 kpc, 230 km/s) encloses about 1e11 Msun.
close(D.enclosedMass(8, 230) / 1e11, 0.98, 0.05, "enclosed mass");
// Isothermal halo is flat at large radius: V -> sqrt(4 pi G rho0) rc.
close(Math.sqrt(D.vhalo2(1e4, 1e7, 5)), Math.sqrt(4 * Math.PI * D.G * 1e7) * 5, 0.5, "flat halo asymptote");
// RAR fit goes to gobs = gbar at high acceleration and sqrt(gbar g†) at low.
close(D.rarFit(1e-8) / 1e-8, 1, 0.02, "RAR, Newtonian limit");
close(D.rarFit(1e-13) / Math.sqrt(1e-13 * D.G_DAGGER), 1, 0.05, "RAR, deep limit");

// NGC 3198, the classic flat rotation curve (van Albada et al. 1985): visible matter falls far short in the outer parts.
{
  const g = gal("NGC3198"), b = D.budget(g), vis = D.fitVisible(g), halo = D.fitHalo(g);
  console.log(`NGC3198: total/visible mass at ${b.r} kpc = ${b.ratio.toFixed(1)}; visible-only best Yd ${vis.yd.toFixed(2)} rchi2 ${vis.rchi2.toFixed(1)}; halo rchi2 ${halo.rchi2.toFixed(2)} (rho0 ${halo.rho0.toExponential(2)}, rc ${halo.rc.toFixed(2)} kpc)`);
  assert.ok(b.ratio > 3, "NGC 3198 needs several times its visible mass");
  assert.ok(halo.rchi2 < vis.rchi2 / 3, "halo fits far better than visible matter alone");
}
// DDO 154, a gas-rich dwarf: dark matter dominates almost everywhere.
{
  const b = D.budget(gal("DDO154"));
  console.log(`DDO154: total/visible = ${b.ratio.toFixed(1)}`);
  assert.ok(b.ratio > 5);
}

// The survey runs on every quality galaxy.
const t0 = Date.now(), s = D.survey(data.galaxies);
const ms = Date.now() - t0;
const visBad = s.filter((x) => x.vis.rchi2 > 5).length, haloBad = s.filter((x) => x.halo.rchi2 > 5).length, highY = s.filter((x) => x.vis.yd > D.Y_MAX_PLAUSIBLE).length;
const ratios = s.map((x) => x.budget.ratio);
console.log(`survey of ${s.length} in ${ms} ms: visible-only rchi2>5 for ${visBad}; halo rchi2>5 for ${haloBad}; visible-only wants Yd>${D.Y_MAX_PLAUSIBLE} for ${highY}; median mass ratio ${D.quantile(ratios, 0.5).toFixed(1)}, fraction ratio>2: ${(ratios.filter((r) => r > 2).length / s.length).toFixed(2)}`);
assert.ok(haloBad < visBad, "the halo model fails less often");
console.log("dm engine: all checks passed");
