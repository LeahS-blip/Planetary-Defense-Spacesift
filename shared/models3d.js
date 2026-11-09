// Real 3D models for SpaceSift, drawn with WebGL (three.js).
//  - star() and rock(): the spectra lab's star and asteroid, rendered off-screen and painted into its 2D orb canvas,
//    so all the orb's existing behaviour (lighting up, the answer ring, reduced motion) is unchanged.
//  - view(): a small orbit viewer for the physics labs' scenes: drag or use the arrow keys to turn, +/- to zoom.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
export { THREE };

export const GLSL_NOISE = /* glsl */ `
vec3 hash33(vec3 p) {
  p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));
  return fract(sin(p) * 43758.5453123);
}
float hash13(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1., 0., 0.)), f.x), mix(hash13(i + vec3(0., 1., 0.)), hash13(i + vec3(1., 1., 0.)), f.x), f.y),
             mix(mix(hash13(i + vec3(0., 0., 1.)), hash13(i + vec3(1., 0., 1.)), f.x), mix(hash13(i + vec3(0., 1., 1.)), hash13(i + vec3(1., 1., 1.)), f.x), f.y), f.z);
}
float fbm(vec3 p) { float a = 0.5, s = 0.0; for (int k = 0; k < 5; k++) { s += a * vnoise(p); p = p * 2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; } return s; }
// Distance to the nearest of a set of jittering cell centres: small in a granule's middle, large in the dark lanes.
float worley(vec3 p, float t) {
  vec3 i = floor(p), f = fract(p); float d = 8.0;
  for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++) for (int z = -1; z <= 1; z++) {
    vec3 g = vec3(float(x), float(y), float(z)), o = hash33(i + g);
    o = 0.5 + 0.42 * sin(t + 6.2831 * o);
    vec3 r = g + o - f; d = min(d, dot(r, r));
  }
  return sqrt(d);
}`;

// ------------------------------------------------------------------ off-screen renderer for the spectra orbs
let OFF; // undefined: not tried yet; null: no WebGL
function off() {
  if (OFF !== undefined) return OFF;
  try {
    const canvas = document.createElement("canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1); renderer.setClearColor(0x000000, 0);
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20); cam.position.z = 5;
    OFF = { canvas, renderer, cam, size: 0 };
  } catch { OFF = null; }
  return OFF;
}
export const ok = () => !!off();
function frame(W, R) {
  const o = off();
  if (o.size !== W) { o.renderer.setSize(W, W, false); o.size = W; }
  const h = W / 2 / R; Object.assign(o.cam, { left: -h, right: h, top: h, bottom: -h }); o.cam.updateProjectionMatrix();
  return o;
}

// The star: granulation from a 3D cell pattern on a turning sphere, limb darkening, and a corona with streamers.
let STAR;
function starScene() {
  if (STAR) return STAR;
  const scene = new THREE.Scene(), u = { uColor: { value: new THREE.Vector3(1, 1, 1) }, uTime: { value: 0 }, uGran: { value: 0.3 }, uFlick: { value: 1 } };
  const surf = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), new THREE.ShaderMaterial({
    uniforms: u,
    vertexShader: /* glsl */ `varying vec3 vObj; varying vec3 vN;
      void main() { vObj = position; vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `uniform vec3 uColor; uniform float uTime, uGran, uFlick; varying vec3 vObj; varying vec3 vN;
      ${GLSL_NOISE}
      void main() {
        float mu = clamp(vN.z, 0.0, 1.0), rr = sqrt(1.0 - mu * mu);
        // Convection cells: big and vigorous in cool stars, small and faint in hot ones.
        float across = 24.0 - 18.0 * uGran, speed = 0.12 + 0.55 * uGran, contrast = 0.05 + 0.42 * uGran;
        float w = worley(vObj * across * 0.5, uTime * speed * 2.0);
        float w2 = worley(vObj * across * 1.1 + 7.0, uTime * speed * 3.0);
        float lanes = smoothstep(0.35, 1.0, w), fine = smoothstep(0.4, 1.0, w2);
        float gran = 1.0 - contrast * lanes - 0.35 * contrast * fine;
        gran += 0.25 * contrast * (1.0 - smoothstep(0.0, 0.4, w)); // bright granule centres
        float mott = 0.93 + 0.1 * fbm(vObj * 2.5 + vec3(0.0, uTime * 0.02, 0.0));
        // Limb darkening: near the edge we see cooler, higher gas, so it is dimmer and redder.
        float I = 1.0 - 0.6 * (1.0 - mu);
        vec3 col = min(uColor * uFlick, vec3(1.0)) * gran * mott * vec3(I, pow(I, 1.2), pow(I, 1.45));
        float hot = 0.3 * uFlick * (1.0 - smoothstep(0.0, 0.75, rr)); // white-hot centre
        col = 1.0 - (1.0 - col) * (1.0 - hot);
        gl_FragColor = vec4(col, 1.0);
      }`,
  }));
  surf.rotation.x = 0.35;
  const corona = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 3.4), new THREE.ShaderMaterial({
    uniforms: u, transparent: true, depthWrite: false,
    vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `uniform vec3 uColor; uniform float uTime, uFlick; varying vec2 vUv;
      ${GLSL_NOISE}
      void main() {
        vec2 p = (vUv * 2.0 - 1.0) * 1.7; float r = length(p), a = atan(p.y, p.x);
        float x = clamp((r - 0.92) / 0.68, 0.0, 1.0);
        float glow = x < 0.3 ? mix(0.6, 0.2, x / 0.3) : mix(0.2, 0.0, (x - 0.3) / 0.7);
        float wisp = fbm(vec3(cos(a) * 2.2, sin(a) * 2.2, r * 3.0 - uTime * 0.15));
        float streak = pow(0.5 + 0.5 * sin(6.0 * a + 0.5 * sin(uTime * 0.13) + 2.0 * wisp), 6.0) * 0.22 * smoothstep(1.7, 1.0, r) * smoothstep(0.85, 1.05, r);
        float alpha = (glow * (0.7 + 0.6 * wisp) + streak) * uFlick * smoothstep(1.6, 1.25, r); // fade out before the canvas edge
        gl_FragColor = vec4(min(uColor * 1.15, vec3(1.0)), clamp(alpha, 0.0, 1.0));
      }`,
  }));
  corona.position.z = -2;
  scene.add(corona, surf);
  return (STAR = { scene, u, surf });
}
/** Paint a 3D star into a 2D context. Returns false when WebGL is unavailable, so the caller can fall back. */
export function star(b, W, R, t, rgb, g, fl) {
  if (!off()) return false;
  try {
    const o = frame(W, R), s = starScene();
    s.u.uColor.value.set(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255); s.u.uTime.value = t; s.u.uGran.value = g; s.u.uFlick.value = fl;
    s.surf.rotation.y = t * 0.04;
    o.renderer.render(s.scene, o.cam);
    b.drawImage(o.canvas, 0, 0, W, W);
    return true;
  } catch { OFF = null; return false; }
}

// ------------------------------------------------------------------ the asteroid
function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function hash3(x, y, z, s) {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(z, 1274126177) + Math.imul(s, 1103515245)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16; return (h >>> 0) / 4294967296;
}
function vnoise(x, y, z, s) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), sm = (v) => v * v * (3 - 2 * v);
  const u = sm(x - xi), v = sm(y - yi), w = sm(z - zi), L = (a, b, t) => a + (b - a) * t, H = (i, j, k) => hash3(xi + i, yi + j, zi + k, s);
  return L(L(L(H(0, 0, 0), H(1, 0, 0), u), L(H(0, 1, 0), H(1, 1, 0), u), v), L(L(H(0, 0, 1), H(1, 0, 1), u), L(H(0, 1, 1), H(1, 1, 1), u), v), w) * 2 - 1;
}
function fbm3(x, y, z, s, oct = 5) { let a = 0.5, f = 1, t = 0; for (let k = 0; k < oct; k++) { t += a * vnoise(x * f, y * f, z * f, s + k); a *= 0.5; f *= 2.03; } return t; }

/** A lumpy, cratered rock: a sphere pushed in and out by noise, with bowl-shaped craters and raised rims. */
export function rockGeometry(seed) {
  const rnd = mulberry(seed), geo = new THREE.SphereGeometry(1, 192, 144), pos = geo.attributes.position, n = pos.count;
  const stretch = [1.15 + 0.35 * rnd(), 0.9 + 0.15 * rnd(), 0.72 + 0.15 * rnd()];
  const craters = Array.from({ length: 22 }, (_, i) => {
    const z = rnd() * 2 - 1, a = rnd() * 6.283, s = Math.sqrt(1 - z * z), big = i < 4;
    return { c: [s * Math.cos(a), s * Math.sin(a), z], r: big ? 0.35 + 0.25 * rnd() : 0.07 + 0.16 * rnd(), d: big ? 0.07 + 0.05 * rnd() : 0.03 + 0.05 * rnd() };
  });
  const col = new Float32Array(n * 3), v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    let r = 1 + 0.22 * fbm3(v.x * 1.1, v.y * 1.1, v.z * 1.1, seed, 3) + 0.035 * fbm3(v.x * 6, v.y * 6, v.z * 6, seed + 9, 4);
    let alb = 1 + 0.12 * fbm3(v.x * 3, v.y * 3, v.z * 3, seed + 21, 3);
    for (const k of craters) {
      const s = Math.acos(Math.max(-1, Math.min(1, v.x * k.c[0] + v.y * k.c[1] + v.z * k.c[2]))) / k.r;
      if (s > 1.6) continue;
      r += s < 1 ? k.d * (s * s - 1) : 0;                       // the bowl
      r += 0.35 * k.d * Math.exp(-(((s - 1) / 0.22) ** 2));      // the raised rim
      if (s < 1) alb *= 0.9 + 0.1 * s;                           // darker floors
    }
    pos.setXYZ(i, v.x * r * stretch[0] * 0.82, v.y * r * stretch[1] * 0.82, v.z * r * stretch[2] * 0.82);
    col[3 * i] = col[3 * i + 1] = col[3 * i + 2] = alb;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();
  return geo;
}
let ROCK;
function rockScene() {
  if (ROCK) return ROCK;
  const scene = new THREE.Scene(), mat = new THREE.MeshStandardMaterial({ color: 0x8a8378, roughness: 0.97, metalness: 0, vertexColors: true });
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat), sun = new THREE.DirectionalLight(0xffffff, 3.4);
  sun.position.set(-0.62, 0.46, 0.64); // light from upper left, as in the 2D version
  scene.add(mesh, sun, new THREE.AmbientLight(0x8090c0, 0.18));
  return (ROCK = { scene, mesh, mat, seed: null });
}
/** Paint a 3D asteroid into a 2D context, tinted with the class colour. Returns false without WebGL. */
export function rock(b, W, R, t, rgb, seed) {
  if (!off()) return false;
  try {
    const o = frame(W, R), s = rockScene();
    if (s.seed !== seed) { s.mesh.geometry.dispose(); s.mesh.geometry = rockGeometry(seed); s.seed = seed; s.mesh.rotation.set(0.5 + (seed % 7) * 0.2, (seed % 13) * 0.5, 0.3); }
    s.mat.color.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);
    s.mesh.rotation.y = (seed % 13) * 0.5 + t * 0.14;
    o.renderer.render(s.scene, o.cam);
    b.drawImage(o.canvas, 0, 0, W, W);
    return true;
  } catch { OFF = null; return false; }
}

// ------------------------------------------------------------------ orbit viewer for the lab scenes
/**
 * Set up a scene on a canvas: drag (or arrow keys when focused) to turn, +/- to zoom. Animation pauses when the
 * canvas is off screen or the tab is hidden; with reduced motion it starts paused.
 * onFrame(dt, t) advances the scene; it is only called while playing.
 */
export function view(canvas, { dist = 16, pitch = 0.5, yaw = 0, minDist = 6, maxDist = 40, fov = 45, onFrame = () => {} } = {}) {
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true }); } catch { return null; }
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1)); renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(fov, 1, 0.05, 600);
  const st = { dist, pitch, yaw }, reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let playing = !reduce, visible = false, dirty = true, last = 0, t = 0, drag = null;
  const place = () => {
    camera.position.set(st.dist * Math.cos(st.pitch) * Math.sin(st.yaw), st.dist * Math.sin(st.pitch), st.dist * Math.cos(st.pitch) * Math.cos(st.yaw));
    camera.lookAt(0, 0, 0); dirty = true;
  };
  const turn = (dy, dp) => { st.yaw += dy; st.pitch = Math.max(-1.45, Math.min(1.45, st.pitch + dp)); place(); };
  const zoom = (f) => { st.dist = Math.max(minDist, Math.min(maxDist, st.dist * f)); place(); };
  canvas.style.touchAction = "pan-y";
  canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
  canvas.addEventListener("pointermove", (e) => { if (!drag) return; turn(-(e.clientX - drag[0]) * 0.008, (e.clientY - drag[1]) * 0.006); drag = [e.clientX, e.clientY]; });
  const up = () => { drag = null; canvas.style.cursor = "grab"; };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.style.cursor = "grab";
  canvas.addEventListener("keydown", (e) => {
    const k = { ArrowLeft: () => turn(0.15, 0), ArrowRight: () => turn(-0.15, 0), ArrowUp: () => turn(0, 0.1), ArrowDown: () => turn(0, -0.1),
      "+": () => zoom(0.88), "=": () => zoom(0.88), "-": () => zoom(1.14), " ": () => api.setPlaying(!playing) }[e.key];
    if (k) { k(); e.preventDefault(); }
  });
  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
  };
  new ResizeObserver(size).observe(canvas);
  new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; }).observe(canvas);
  const loop = (now) => {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0; last = now;
    if (visible && !document.hidden) {
      if (playing) { t += dt; onFrame(dt, t); dirty = true; }
      if (dirty) { renderer.render(scene, camera); dirty = false; }
    }
    requestAnimationFrame(loop);
  };
  const api = {
    THREE, scene, camera, renderer,
    get playing() { return playing; },
    setPlaying(p) { playing = p; canvas.dispatchEvent(new CustomEvent("ss-play", { detail: p })); },
    step(dt) { t += dt; onFrame(dt, t); dirty = true; }, // advance once, e.g. for a still frame with reduced motion
    redraw() { dirty = true; }, zoom, turn, reduce,
  };
  place(); size(); requestAnimationFrame(loop);
  return api;
}

/** A soft round sprite texture for glows and stars. */
export function glowTexture(stops = [[0, "rgba(255,255,255,1)"], [0.25, "rgba(255,255,255,.55)"], [1, "rgba(255,255,255,0)"]], N = 128) {
  const c = document.createElement("canvas"); c.width = c.height = N;
  const g = c.getContext("2d"), gr = g.createRadialGradient(N / 2, N / 2, 0, N / 2, N / 2, N / 2);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr; g.fillRect(0, 0, N, N);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; return tex;
}

/** A sky of faint background stars on a big sphere. */
export function starfield(n = 1500, R = 120, seed = 7) {
  const rnd = mulberry(seed), p = new Float32Array(n * 3), c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const z = rnd() * 2 - 1, a = rnd() * 6.283, s = Math.sqrt(1 - z * z), b = 0.35 + 0.65 * rnd() ** 2, warm = rnd();
    p.set([R * s * Math.cos(a), R * z, R * s * Math.sin(a)], 3 * i);
    c.set([b * (0.85 + 0.15 * warm), b * 0.9, b * (1 - 0.15 * warm)], 3 * i);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false }));
}
export { mulberry };
