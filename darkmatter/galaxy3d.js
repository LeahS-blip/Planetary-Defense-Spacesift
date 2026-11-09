// A 3D model of a SPARC galaxy, spinning at its real, measured speeds.
//
// The stars and gas are illustrative (a disk with two spiral arms, sized from the galaxy's own measurements), but
// the motion is not: every particle orbits at the speed the rotation curve gives at its radius. Switch the dark
// matter off and each particle keeps its measured speed but feels only the gravity of the visible matter (gas plus
// stars at Υ = 0.5, treated as spherical). The outer stars are then moving faster than that gravity can hold, and
// they fly outward. A translucent sphere shows the fitted dark halo, as its column density seen from the camera.
import { THREE, view, glowTexture, starfield, mulberry } from "/shared/models3d.js";
import * as D from "/darkmatter/dm.js";

const N = 9000, SIZE = 8, MYR = 977.8; // kpc / (km/s) in millions of years
const SCREEN_ORBIT = 16;               // seconds for one turn at the last measured point

export function galaxyScene(canvas, { onStatus = () => {} } = {}) {
  const V = view(canvas, { dist: 22, pitch: 0.72, yaw: 0.3, minDist: 5, maxDist: 50, onFrame: (dt) => advance(dt) });
  if (!V) return null;
  const scene = V.scene;
  scene.add(starfield(1600, 160, 3));

  // -- particles
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3), size = new Float32Array(N);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3)); geo.setAttribute("size", new THREE.BufferAttribute(size, 1));
  const pts = new THREE.Points(geo, new THREE.ShaderMaterial({
    uniforms: { uPx: { value: Math.min(2, devicePixelRatio || 1) } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float size; attribute vec3 color; uniform float uPx; varying vec3 vC;
      void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vC = color; gl_PointSize = size * uPx * 26.0 / -mv.z; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; void main() { float d = length(gl_PointCoord - 0.5); float a = smoothstep(0.5, 0.0, d); gl_FragColor = vec4(vC * a * a, 1.0); }`,
  }));
  pts.frustumCulled = false; scene.add(pts);
  const core = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture([[0, "rgba(255,236,200,.9)"], [0.3, "rgba(255,200,140,.25)"], [1, "rgba(255,180,120,0)"]]), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(core);

  // -- the dark halo: brightness follows the projected density of a cored isothermal sphere, ∝ rc / sqrt(rc² + b²)
  const HU = { uRc: { value: 1 }, uR: { value: SIZE * 2.3 }, uOn: { value: 1 } };
  const halo = new THREE.Mesh(new THREE.SphereGeometry(SIZE * 2.3, 64, 48), new THREE.ShaderMaterial({
    uniforms: HU, transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
    vertexShader: `varying vec3 vW; void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`,
    fragmentShader: `uniform float uRc, uR, uOn; varying vec3 vW;
      void main() {
        vec3 d = normalize(vW - cameraPosition); float tc = -dot(cameraPosition, d);
        float b = length(cameraPosition + d * tc); // closest approach of this sight line to the centre
        float sigma = sqrt(uRc / sqrt(uRc * uRc + b * b)) * smoothstep(uR, uR * 0.45, b); // square root: a gentler fall-off for the eye
        gl_FragColor = vec4(vec3(0.55, 0.42, 1.0) * sigma * 0.4 * uOn, 1.0);
      }`,
  }));
  scene.add(halo);

  // -- a dashed ring at the last measured point
  const ringGeo = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 181 }, (_, i) => new THREE.Vector3(Math.cos(i / 180 * 6.2832), 0, Math.sin(i / 180 * 6.2832))));
  const edge = new THREE.Line(ringGeo, new THREE.LineDashedMaterial({ color: 0x7aa2ff, dashSize: 0.06, gapSize: 0.05, transparent: true, opacity: 0.45 }));
  edge.computeLineDistances(); edge.scale.setScalar(SIZE); scene.add(edge);

  // -- state
  let g, s, rmax, Vobs, gbar, k, mode = "dm", tSim = 0, lastStatus = 0;
  const P = { w: new Float32Array(N), r: new Float32Array(N), th: new Float32Array(N), z: new Float32Array(N), x: new Float32Array(N), y: new Float32Array(N), vx: new Float32Array(N), vy: new Float32Array(N) };

  function interp(rows, f, r) { // linear in radius; rising linearly from the centre to the first point
    if (r <= rows[0][0]) return f(rows[0]) * r / rows[0][0];
    for (let i = 1; i < rows.length; i++) if (r <= rows[i][0]) { const a = rows[i - 1], b = rows[i], u = (r - a[0]) / (b[0] - a[0]); return f(a) + (f(b) - f(a)) * u; }
    return null;
  }

  function setGalaxy(gal) {
    g = gal; const rows = g.rows; rmax = rows.at(-1)[0]; s = SIZE / rmax;
    const vLast = rows.at(-1)[1], vbLast = D.vbar(rows.at(-1), D.Y_DISK);
    Vobs = (r) => interp(rows, (q) => q[1], r) ?? vLast;
    const vb = (r) => interp(rows, (q) => D.vbar(q, D.Y_DISK), r) ?? vbLast * Math.sqrt(rmax / r);
    // gravity of the visible matter, tabulated: g = Vbar² / r
    const NT = 800, RT = 8 * rmax, tab = new Float32Array(NT + 1);
    for (let i = 0; i <= NT; i++) { const r = Math.max(1e-3, (i / NT) * RT); tab[i] = vb(r) ** 2 / r; }
    gbar = (r) => { if (r >= RT) return vbLast ** 2 * rmax / (r * r); const x = (r / RT) * NT, i = Math.floor(x); return tab[i] + (tab[Math.min(NT, i + 1)] - tab[i]) * (x - i); };
    k = (2 * Math.PI) / (SCREEN_ORBIT * (vLast / rmax));

    // Where the starlight is: scale length from the radius where the disk's own rotation peaks (≈ 2.2 Rd).
    let pk = rows[0]; for (const q of rows) if (q[4] > pk[4]) pk = q;
    const Rd = Math.max(0.06 * rmax, Math.min(0.35 * rmax, pk[4] > 0 ? pk[0] / 2.2 : rmax / 4));
    const bulge = rows.some((q) => q[5] > 0), rnd = mulberry([...g.name].reduce((a, c) => a * 31 + c.charCodeAt(0), 7) | 0);
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(6.2832 * rnd());
    const pitch = Math.tan((14 * Math.PI) / 180), nGas = Math.floor(N * 0.3), nBul = bulge ? Math.floor(N * 0.1) : 0;
    for (let i = 0; i < N; i++) {
      let r, th, z, c, sz;
      if (i < nGas) { // gas reaches out as far as the measurements, often well past the stars
        r = rmax * 1.05 * Math.sqrt(rnd()); th = rnd() * 6.2832; z = 0.015 * rmax * gauss();
        const knot = rnd() < 0.05; c = knot ? [1, 0.45, 0.65] : [0.32, 0.5, 0.95]; sz = knot ? 2.6 : 2.4;
        if (rnd() < 0.5) th = (rnd() < 0.5 ? 0 : Math.PI) + Math.log(Math.max(r, 1e-3) / Rd) / pitch + 0.5 * gauss();
      } else if (i < nGas + nBul) {
        r = Math.abs(0.07 * rmax * gauss()) + 1e-3; th = rnd() * 6.2832; z = 0.05 * rmax * gauss(); c = [1, 0.8, 0.52]; sz = 1.8;
      } else {
        do r = -Rd * Math.log(rnd() * rnd() + 1e-9); while (r > 1.1 * rmax);
        const arm = rnd() < 0.55;
        th = arm ? (rnd() < 0.5 ? 0 : Math.PI) + Math.log(Math.max(r, 1e-3) / Rd) / pitch + 0.28 * gauss() : rnd() * 6.2832;
        z = 0.025 * rmax * gauss() * Math.exp(-r / (3 * Rd));
        const b = 0.55 + 0.45 * rnd();
        c = arm ? [0.62 * b, 0.76 * b, 1.0 * b] : [1.0 * b, 0.88 * b, 0.72 * b]; sz = arm ? 1.5 + rnd() : 1.2 + 0.8 * rnd();
      }
      P.r[i] = r; P.th[i] = th; P.z[i] = z;
      col.set(c, 3 * i); size[i] = sz;
    }
    geo.attributes.color.needsUpdate = true; geo.attributes.size.needsUpdate = true;
    core.scale.setScalar(bulge ? 4.2 : 2.4);
    const h = D.fitHalo(g); HU.uRc.value = Math.max(0.15, h.rc * s);
    for (let i = 0; i < N; i++) { const r = Math.max(P.r[i], 1e-3); P.w[i] = Vobs(r) / r; } // angular speed at each radius
    setMode("dm");
  }

  function setMode(m) {
    mode = m; tSim = 0;
    if (m === "nodm") for (let i = 0; i < N; i++) { // start from where each particle is now, at its measured speed
      const r = P.r[i], th = P.th[i], v = Vobs(r);
      P.x[i] = r * Math.cos(th); P.y[i] = r * Math.sin(th); P.vx[i] = -v * Math.sin(th); P.vy[i] = v * Math.cos(th);
    }
    place(); status(true);
  }

  function advance(dt) {
    if (!g) return;
    const h = dt * k; tSim += h;
    HU.uOn.value += ((mode === "dm" ? 1 : 0) - HU.uOn.value) * Math.min(1, dt * 3);
    if (mode === "dm") {
      for (let i = 0; i < N; i++) P.th[i] += P.w[i] * h;
    } else {
      // Leapfrog in the visible matter's gravity. Substeps keep the fast inner orbits stable.
      const wMax = Vobs(0.15 * rmax) / (0.15 * rmax), n = Math.min(24, Math.max(1, Math.ceil((h * wMax) / 0.08))), e = h / n;
      for (let j = 0; j < n; j++) for (let i = 0; i < N; i++) {
        let x = P.x[i], y = P.y[i], r = Math.hypot(x, y) + 1e-3, a = gbar(r) / r;
        P.vx[i] -= 0.5 * e * a * x; P.vy[i] -= 0.5 * e * a * y;
        x += e * P.vx[i]; y += e * P.vy[i]; P.x[i] = x; P.y[i] = y;
        r = Math.hypot(x, y) + 1e-3; a = gbar(r) / r;
        P.vx[i] -= 0.5 * e * a * x; P.vy[i] -= 0.5 * e * a * y;
      }
    }
    place(); status(false);
  }

  function place() {
    if (!g) return;
    for (let i = 0; i < N; i++) {
      let x, y;
      if (mode === "dm") { x = P.r[i] * Math.cos(P.th[i]); y = P.r[i] * Math.sin(P.th[i]); } else { x = P.x[i]; y = P.y[i]; }
      pos[3 * i] = x * s; pos[3 * i + 1] = P.z[i] * s; pos[3 * i + 2] = y * s;
    }
    geo.attributes.position.needsUpdate = true; V.redraw();
  }

  function status(force) {
    const now = performance.now(); if (!force && now - lastStatus < 200) return; lastStatus = now;
    const vLast = Vobs(rmax), myr = tSim * MYR;
    if (mode === "dm") {
      onStatus({ mode, text: `Spinning at the measured speeds · ${myr < 1000 ? myr.toFixed(0) + " million" : (myr / 1000).toFixed(1) + " billion"} years shown`,
        real: `At the dashed ring, the last measured point (${rmax.toFixed(1)} kpc from the centre), stars and gas orbit at ${vLast.toFixed(0)} km/s: one lap every ${(2 * Math.PI * rmax / vLast * MYR).toFixed(0)} million years. The inner parts lap faster, which winds the arms up.` });
    } else {
      let out = 0; for (let i = 0; i < N; i++) if (Math.hypot(P.x[i], P.y[i]) > 1.3 * rmax) out++;
      onStatus({ mode, text: `Dark matter removed · ${myr < 1000 ? myr.toFixed(0) + " million" : (myr / 1000).toFixed(1) + " billion"} years later · ${(100 * out / N).toFixed(0)}% of the galaxy has flown past 1.3× its edge`,
        real: `Same starting speeds, but only the visible matter's gravity: ${vLast.toFixed(0)} km/s at the edge, where visible matter alone can only hold ${D.vbar(g.rows.at(-1), D.Y_DISK).toFixed(0)} km/s. Without extra mass, this galaxy can't stay together.` });
    }
  }

  return { setGalaxy, setMode, get mode() { return mode; }, reset: () => setMode(mode), view: V };
}
