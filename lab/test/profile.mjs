// Where does a demo trial spend its time?  node lab/test/profile.mjs
import * as E from "../engine.js";

for (const baseline of [120, 368]) {
  const lc = E.syntheticStar({ baseline, noisePpm: 150, kind: "pulsation", periodD: 4 / 24, ampPpm: 300, seed: 1 });
  let t = Date.now();
  const bw = E.flatten(lc.time, lc.flux, "biweight-1.0");
  const tBw = Date.now() - t; t = Date.now();
  E.prewhiten(lc.time, lc.flux);
  const tPw = Date.now() - t; t = Date.now();
  console.log(`  prewhitening removed ${E.prewhitenTermsUsed} sinusoids`);
  const [ct, cf] = E.clean(lc.time, bw);
  const r = E.bls(ct, cf);
  const tBls = Date.now() - t;
  console.log(`${baseline} d (${lc.time.length} pts, ${r.periods.length} periods): biweight ${tBw} ms, prewhiten ${tPw} ms, BLS ${tBls} ms`);
}
