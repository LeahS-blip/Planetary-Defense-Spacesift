// 3D model of a binary black hole merger: two black holes spiralling together on a sheet that stands for
// spacetime, with the gravitational waves spreading out as ripples, then the merger and the ringdown.
//
// The motion follows the same leading-order physics as the lab's templates, slowed down so you can watch it:
//   separation a ∝ τ^(1/4), orbital frequency ω ∝ τ^(-3/8), phase Φ ∝ -τ^(5/8), wave amplitude ∝ ω^(2/3),
// with τ the time left before merger. The ripples leave the orbit at a fixed speed, so each ring of the
// pattern shows the orbit as it was when that ring left (the retarded time). Not to scale: the real sheet
// would stretch by less than a proton's width, and the depth of the wells is only a picture of gravity.
import { THREE, view, glowTexture, starfield } from "/shared/models3d.js";
import * as G from "/gw/gw.js";

const A0 = 5.5, W0 = (2 * Math.PI) / 2.6, WAVE_SPEED = 4.2, RING = 1.4, HOLD = 1.8;

export function mergerScene(canvas, { onStatus = () => {} } = {}) {
  let P = null, t = 0, phaseName = "", lastStatus = 0;
  const V = view(canvas, { dist: 19, pitch: 0.55, yaw: 0.5, minDist: 7, maxDist: 42, onFrame: (dt) => advance(dt) });
  if (!V) return null;
  const scene = V.scene;
  scene.add(starfield(1800, 150));

  // -- the spacetime sheet
  const U = {
    uT: { value: 0 }, uTau0: { value: 7 }, uTauEnd: { value: 0.01 }, uW0: { value: W0 }, uPhic: { value: 0 }, uV: { value: WAVE_SPEED },
    uAmp: { value: 0.07 }, uP1: { value: new THREE.Vector3() }, uP2: { value: new THREE.Vector3() }, uGM: { value: 1 }, uMerged: { value: 0 }, uTm: { value: 6 },
  };
  const sheet = new THREE.Mesh(new THREE.PlaneGeometry(44, 44, 300, 300), new THREE.ShaderMaterial({
    uniforms: U, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      uniform float uT, uTau0, uTauEnd, uW0, uPhic, uV, uAmp, uGM, uMerged, uTm; uniform vec3 uP1, uP2;
      varying vec2 vXZ; varying float vRip; varying float vWell;
      // Orbital phase and wave amplitude at source time s.
      vec2 source(float s) {
        float tau = uTau0 - s;
        if (tau > uTauEnd) {
          float w = uW0 * pow(tau / uTau0, -0.375);
          return vec2(uPhic - 1.6 * uW0 * pow(uTau0, 0.375) * pow(tau, 0.625), uAmp * pow(w / uW0, 0.6667));
        }
        float wEnd = uW0 * pow(uTauEnd / uTau0, -0.375), ds = s - uTm;
        float phiEnd = uPhic - 1.6 * uW0 * pow(uTau0, 0.375) * pow(uTauEnd, 0.625);
        return vec2(phiEnd + 1.15 * wEnd * ds, uAmp * pow(wEnd / uW0, 0.6667) * exp(-ds / 0.35));
      }
      void main() {
        vec3 p = vec3(position.x, 0.0, -position.y);
        float r = length(p.xz), ang = atan(p.z, p.x);
        vec2 src = source(uT - r / uV);
        // A quadrupole wave: two crests per orbit, spiralling out. Fades in away from the centre, falls off as 1/r.
        float rip = src.y * cos(2.0 * (ang - src.x)) * smoothstep(0.6, 3.2, r) * 3.2 / (1.0 + 0.16 * r);
        float d1 = length(p.xz - uP1.xz), d2 = length(p.xz - uP2.xz);
        float well = uMerged > 0.5 ? -1.9 * uGM / sqrt(r * r + 0.5) : -uGM / sqrt(d1 * d1 + 0.45) - uGM / sqrt(d2 * d2 + 0.45);
        p.y = well + rip;
        vXZ = p.xz; vRip = rip; vWell = well;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: /* glsl */ `
      varying vec2 vXZ; varying float vRip; varying float vWell;
      void main() {
        vec2 q = vXZ * 1.0, g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
        float line = 1.0 - min(min(g.x, g.y), 1.0);
        float r = length(vXZ), fade = smoothstep(21.0, 8.0, r);
        float wave = clamp(abs(vRip) * 5.0, 0.0, 1.0);
        vec3 base = mix(vec3(0.25, 0.38, 0.9), vec3(0.98, 0.55, 0.85), wave);
        vec3 col = base * line * (0.5 + 0.9 * wave) + vec3(0.12, 0.16, 0.4) * clamp(-vWell * 0.25, 0.0, 1.0) * 0.35;
        gl_FragColor = vec4(col * fade, 1.0);
      }`,
  }));
  scene.add(sheet);

  // -- the black holes: black spheres with a bright rim, the light bent around them
  const ringTex = glowTexture([[0, "rgba(0,0,0,0)"], [0.36, "rgba(0,0,0,0)"], [0.42, "rgba(255,236,210,1)"], [0.5, "rgba(190,210,255,.55)"], [1, "rgba(120,150,255,0)"]], 256);
  const flashTex = glowTexture([[0, "rgba(255,255,255,1)"], [0.2, "rgba(220,230,255,.8)"], [0.55, "rgba(150,120,255,.25)"], [1, "rgba(120,100,255,0)"]], 256);
  const rimMat = new THREE.ShaderMaterial({
    vertexShader: `varying vec3 vN; varying vec3 vV; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vN; varying vec3 vV; void main() { float f = pow(1.0 - max(dot(vN, vV), 0.0), 4.0); gl_FragColor = vec4(vec3(0.75, 0.82, 1.0) * f * 1.4, 1.0); }`,
  });
  const hole = () => {
    const g = new THREE.Group(), core = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), rimMat);
    const ring = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    ring.scale.setScalar(4.6); g.add(ring, core); g.userData = { core, ring }; scene.add(g); return g;
  };
  const bh1 = hole(), bh2 = hole(), rem = hole();
  const flash = new THREE.Sprite(new THREE.SpriteMaterial({ map: flashTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
  scene.add(flash);

  // -- fading orbit trails
  const TRAIL = 160;
  const trail = (col) => {
    const g = new THREE.BufferGeometry(), pos = new Float32Array(TRAIL * 3), c = new Float32Array(TRAIL * 3);
    for (let i = 0; i < TRAIL; i++) { const k = (1 - i / TRAIL) ** 1.6; c.set([col[0] * k, col[1] * k, col[2] * k], 3 * i); }
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    l.frustumCulled = false; scene.add(l); return { l, pos, n: 0 };
  };
  const tr1 = trail([0.55, 0.7, 1]), tr2 = trail([1, 0.6, 0.9]);
  const push = (T, v) => { T.pos.copyWithin(3, 0, (TRAIL - 1) * 3); T.pos.set([v.x, v.y, v.z], 0); T.n = Math.min(TRAIL, T.n + 1); T.l.geometry.setDrawRange(0, T.n); T.l.geometry.attributes.position.needsUpdate = true; };
  const clearTrail = (T) => { T.n = 0; T.l.geometry.setDrawRange(0, 0); };

  function setMass(mc) {
    const m = mc * 2 ** 0.2, rbh = 0.3 * (m / 34) ** 0.6, aEnd = 2.6 * rbh;
    const tau0 = 7 * (30 / mc) ** 0.8, tauEnd = tau0 * (aEnd / A0) ** 4;
    P = { mc, m, rbh, aEnd, tau0, tauEnd, tm: tau0 - tauEnd, gm: 0.75 * (m / 34) ** 0.5, fIsco: G.fIsco(mc) };
    U.uTau0.value = tau0; U.uTauEnd.value = tauEnd; U.uPhic.value = 1.6 * W0 * tau0; U.uTm.value = P.tm; U.uGM.value = P.gm;
    for (const b of [bh1, bh2]) { b.userData.core.scale.setScalar(rbh); b.userData.ring.scale.setScalar(rbh * 4.6); }
    restart();
  }
  function restart() { t = 0; clearTrail(tr1); clearTrail(tr2); advance(0); V.redraw(); }
  const phiAt = (tau) => U.uPhic.value - 1.6 * W0 * P.tau0 ** 0.375 * tau ** 0.625;

  function advance(dt) {
    if (!P) return;
    t += dt;
    if (t > P.tm + RING + HOLD) { t = 0; clearTrail(tr1); clearTrail(tr2); }
    U.uT.value = t;
    const tau = P.tau0 - t, merged = tau <= P.tauEnd;
    U.uMerged.value = merged ? 1 : 0;
    bh1.visible = bh2.visible = !merged; rem.visible = merged;
    let status;
    if (!merged) {
      const a = A0 * (tau / P.tau0) ** 0.25, ph = phiAt(tau), w = W0 * (tau / P.tau0) ** -0.375;
      bh1.position.set((a / 2) * Math.cos(ph), 0, (a / 2) * Math.sin(ph)); bh2.position.set(-bh1.position.x, 0, -bh1.position.z);
      // ride the sheet: each hole sits in its own well
      const y = -P.gm / Math.sqrt(0.45) - P.gm / Math.sqrt(a * a + 0.45);
      bh1.position.y = bh2.position.y = y + P.rbh * 0.6;
      U.uP1.value.copy(bh1.position); U.uP2.value.copy(bh2.position);
      push(tr1, bh1.position); push(tr2, bh2.position);
      flash.material.opacity = 0;
      const fReal = P.fIsco * (tau / P.tauEnd) ** -0.375, tauReal = G.tauOfF(fReal, P.mc) - G.tauOfF(P.fIsco, P.mc);
      status = { phase: "Inspiral", text: `Inspiral · ${(w / (2 * Math.PI)).toFixed(1)} orbits per second on screen · merger in ${(tau - P.tauEnd).toFixed(1)} s`,
        real: realLine(fReal, tauReal, tau - P.tauEnd) };
    } else {
      const s = t - P.tm, mf = 0.95 * 2 * P.m, rr = 0.3 * (mf / 34) ** 0.6 * 1.05;
      const wob = 1 + 0.08 * Math.exp(-s / 0.35) * Math.sin(s * 30); // the new horizon rings like a struck bell
      rem.position.set(0, -1.9 * P.gm / Math.sqrt(0.5) + rr * 0.6, 0);
      rem.userData.core.scale.set(rr * wob, rr / wob, rr * wob); rem.userData.ring.scale.setScalar(rr * 4.6 * wob);
      flash.position.copy(rem.position); flash.scale.setScalar(2 + 9 * (1 - Math.exp(-s / 0.25)));
      flash.material.opacity = Math.exp(-s / 0.45);
      if (tr1.n) { clearTrail(tr1); clearTrail(tr2); }
      status = s < 0.35 ? { phase: "Merger", text: "Merger: the horizons touch and become one", real: `About 5% of the mass, ${(0.05 * 2 * P.m).toFixed(1)} Suns' worth, leaves as gravitational waves in a fraction of a second.` }
        : s < RING ? { phase: "Ringdown", text: "Ringdown: the new black hole shakes and settles", real: `One black hole of about ${mf.toFixed(0)} solar masses. The last ripples spread outward.` }
        : { phase: "Settled", text: "A single, quiet black hole. Starting again…", real: `Your chirp mass ${P.mc.toFixed(0)} M☉ means two ${P.m.toFixed(0)} M☉ black holes (equal masses assumed).` };
    }
    if (status.phase !== phaseName || performance.now() - lastStatus > 150) { phaseName = status.phase; lastStatus = performance.now(); onStatus(status); }
  }

  return { setMass, restart, view: V };
}

// The real system at the same point in its inspiral: wave pitch, time left, and how the animation's clock compares.
function realLine(f, tauReal, tauScreen) {
  const left = tauReal < 1 ? `${(1000 * tauReal).toFixed(0)} ms` : `${tauReal.toFixed(1)} s`;
  const k = tauScreen / Math.max(tauReal, 1e-5);
  const clock = k >= 1.5 ? `shown about ${k.toFixed(0)}× slower` : k <= 0.67 ? `shown about ${(1 / k).toFixed(0)}× faster` : "shown at about real speed";
  const band = f < 20 ? " Still too low for LIGO, which hears from about 20 Hz." : " Inside LIGO's band.";
  return `In reality: waves at ${f < 10 ? f.toFixed(1) : f.toFixed(0)} Hz, ${left} before merger, ${clock}.${band}`;
}
