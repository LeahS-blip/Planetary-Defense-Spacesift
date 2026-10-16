// Tonight's sky, live: the celestial sphere drawn as a globe you can turn, with every candidate on the
// Minor Planet Center's confirmation page at its sky position. Positions and scores are read from the live
// dashboard, which the pipeline rebuilds twice a day.
const cv = document.getElementById("globe");
if (cv) init();

function init() {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const DPR = Math.min(2, devicePixelRatio || 1), tip = document.getElementById("globe-tip");
  const EPS = (23.4393 * Math.PI) / 180;
  const vec = (raH, dec) => { const a = (raH / 24) * 2 * Math.PI, d = (dec * Math.PI) / 180; return [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)]; };
  const ring = (f) => Array.from({ length: 145 }, (_, i) => f((i / 144) * 2 * Math.PI));
  const equator = ring((a) => [Math.cos(a), Math.sin(a), 0]);
  const ecliptic = ring((l) => [Math.cos(l), Math.sin(l) * Math.cos(EPS), Math.sin(l) * Math.sin(EPS)]);
  const meridians = Array.from({ length: 6 }, (_, k) => ring((d) => { const a = (k / 6) * Math.PI; return [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)]; }));
  let seed = 11; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const bg = Array.from({ length: 700 }, () => { const z = rnd() * 2 - 1, a = rnd() * 6.283, s = Math.sqrt(1 - z * z); return { v: [s * Math.cos(a), s * Math.sin(a), z], m: rnd() }; });
  let focus = null, target = null, downAt = null;
  let objs = [], yaw = 0.6, pitch = 0.35, vyaw = reduce ? 0 : 0.0016, drag = null, hover = null, visible = false, mouse = null;

  const rot = ([x, y, z]) => { // spin about the celestial pole, then tilt toward the viewer
    const x1 = x * Math.cos(yaw) - y * Math.sin(yaw), y1 = x * Math.sin(yaw) + y * Math.cos(yaw);
    return [x1, y1 * Math.cos(pitch) - z * Math.sin(pitch), y1 * Math.sin(pitch) + z * Math.cos(pitch)];
  };
  function draw(t) {
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * DPR) || cv.height !== Math.round(h * DPR)) { cv.width = Math.round(w * DPR); cv.height = Math.round(h * DPR); }
    const ctx = cv.getContext("2d"); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, w, h);
    const R = Math.min(w, h) * 0.42, cx = w / 2, cy = h / 2;
    const P = (v) => { const [x, d, z] = rot(v); return [cx + x * R, cy - z * R, -d]; }; // depth > 0 faces the viewer
    // sphere glow
    const gl = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 1.25);
    gl.addColorStop(0, "rgba(60,90,200,.10)"); gl.addColorStop(0.8, "rgba(60,90,200,.05)"); gl.addColorStop(1, "rgba(60,90,200,0)");
    ctx.fillStyle = gl; ctx.beginPath(); ctx.arc(cx, cy, R * 1.25, 0, 6.283); ctx.fill();
    ctx.strokeStyle = "rgba(140,160,220,.22)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 6.283); ctx.stroke();
    const line = (pts, col, back, width = 1, dash) => {
      ctx.lineWidth = width; ctx.setLineDash(dash || []);
      for (const side of [false, true]) {
        ctx.strokeStyle = side ? col : back; ctx.beginPath(); let on = false;
        for (const v of pts) { const [x, y, d] = P(v); if ((d > 0) === side) { on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true; } else on = false; }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    };
    // the far side first
    for (const s of bg) { const [x, y, d] = P(s.v); if (d > 0) continue; ctx.globalAlpha = 0.12 + 0.15 * s.m; ctx.fillStyle = "#c9d0e4"; ctx.fillRect(x, y, 1, 1); }
    ctx.globalAlpha = 1;
    for (const m of meridians) line(m, "rgba(140,160,220,.14)", "rgba(140,160,220,.05)");
    line(equator, "rgba(140,160,220,.35)", "rgba(140,160,220,.08)");
    line(ecliptic, "rgba(249,168,212,.6)", "rgba(249,168,212,.12)", 1.3, [5, 4]);
    const proj = objs.map((o) => { const [x, y, d] = P(o.v); return { o, x, y, d }; }).sort((a, b) => a.d - b.d);
    for (const q of proj) if (q.d <= 0) dot(ctx, q, 0.25);
    // Earth at the centre: we are inside this sphere, looking out
    const er = R * 0.13, eg = ctx.createRadialGradient(cx - er * 0.4, cy - er * 0.4, er * 0.1, cx, cy, er);
    eg.addColorStop(0, "#9fd3ff"); eg.addColorStop(0.5, "#2c6fd1"); eg.addColorStop(1, "#0b2350");
    ctx.fillStyle = "rgba(90,160,255,.18)"; ctx.beginPath(); ctx.arc(cx, cy, er * 1.5, 0, 6.283); ctx.fill();
    ctx.fillStyle = eg; ctx.beginPath(); ctx.arc(cx, cy, er, 0, 6.283); ctx.fill();
    // near side
    for (const s of bg) { const [x, y, d] = P(s.v); if (d <= 0) continue; ctx.globalAlpha = 0.3 + 0.6 * s.m; ctx.fillStyle = "#eef2fc"; ctx.fillRect(x, y, 1.3, 1.3); }
    ctx.globalAlpha = 1;
    hover = null;
    for (const q of proj) if (q.d > 0) { dot(ctx, q, 1, t); if (mouse && Math.hypot(mouse[0] - q.x, mouse[1] - q.y) < 12) hover = q; }
    // labels for the three likeliest
    ctx.font = "600 11px Comfortaa, sans-serif";
    for (const q of proj) if (q.d > 0 && q.o.rank < 3 && q.o !== focus) { ctx.fillStyle = "#eef2fc"; ctx.globalAlpha = 0.85; ctx.fillText(q.o.desig, q.x + 10, q.y + 4); }
    ctx.globalAlpha = 1;
    const fq = focus && proj.find((q) => q.o === focus);
    if (fq && fq.d > 0) {
      const pr = 14 + (reduce ? 0 : 3 * Math.sin(t * 0.005));
      ctx.strokeStyle = "#eef2fc"; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(fq.x, fq.y, pr, 0, 6.283); ctx.stroke();
      for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2 + t * 0.0008; ctx.beginPath(); ctx.moveTo(fq.x + Math.cos(a) * (pr + 4), fq.y + Math.sin(a) * (pr + 4)); ctx.lineTo(fq.x + Math.cos(a) * (pr + 11), fq.y + Math.sin(a) * (pr + 11)); ctx.stroke(); }
      ctx.fillStyle = "#eef2fc"; ctx.font = "700 12px Comfortaa, sans-serif"; ctx.fillText(fq.o.desig, fq.x + pr + 14, fq.y + 4);
    }
    if (hover) { tip.hidden = false; tip.style.left = `${hover.x + 14}px`; tip.style.top = `${hover.y - 10}px`;
      tip.innerHTML = `<b>${hover.o.desig}</b><br>P(real NEO) ${hover.o.p.toFixed(2)} · brightness V ${hover.o.mag}`; cv.style.cursor = "pointer"; }
    else { tip.hidden = true; cv.style.cursor = drag ? "grabbing" : "grab"; }
  }
  function dot(ctx, q, a, t = 0) {
    const p = q.o.p, col = p >= 0.8 ? "74,222,128" : p >= 0.4 ? "251,191,36" : "148,163,184", r = 2 + p * 3.5;
    if (p >= 0.8 && a === 1) { const pulse = reduce ? 0.5 : 0.5 + 0.5 * Math.sin(t * 0.003 + q.o.rank);
      const g = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, r * 5); g.addColorStop(0, `rgba(${col},${0.35 * pulse + 0.15})`); g.addColorStop(1, `rgba(${col},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(q.x, q.y, r * 5, 0, 6.283); ctx.fill(); }
    ctx.fillStyle = `rgba(${col},${a})`; ctx.beginPath(); ctx.arc(q.x, q.y, r, 0, 6.283); ctx.fill();
  }
  function loop(t) {
    if (visible) {
      if (target && !drag) { // swing the sky so the chosen candidate faces us
        let dy = target[0] - yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
        yaw += dy * (reduce ? 1 : 0.08); pitch += (target[1] - pitch) * (reduce ? 1 : 0.08);
        if (Math.abs(dy) < 0.002 && Math.abs(target[1] - pitch) < 0.002) target = null;
      } else if (!drag && !focus) { yaw += vyaw; vyaw += ((reduce ? 0 : 0.0016) - vyaw) * 0.02; }
      draw(t);
    }
    requestAnimationFrame(loop);
  }
  cv.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; downAt = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => {
    const r = cv.getBoundingClientRect(); mouse = [e.clientX - r.left, e.clientY - r.top];
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY];
    yaw += dx * 0.006; pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.006)); vyaw = dx * 0.006;
    if (reduce) draw(0);
  });
  cv.addEventListener("pointerup", (e) => {
    const click = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5; drag = null; downAt = null;
    if (click && hover) focusOn(hover.o); else if (!click) { focus = null; showFocus(null); }
  });
  function focusOn(o) {
    focus = o;
    const [x, y, z] = o.v, rho = Math.hypot(x, y);
    target = [-Math.PI / 2 - Math.atan2(y, x), Math.max(-1.5, Math.min(1.5, Math.atan2(z, rho) - 0.4))]; // parked just above Earth
    vyaw = 0; showFocus(o);
    dispatchEvent(new CustomEvent("ss-sky-focus", { detail: { desig: o.desig } })); // the flat map highlights it too
    dispatchEvent(new CustomEvent("ss-sound", { detail: { type: "focus" } }));
  }
  function showFocus(o) {
    const box = document.getElementById("globe-focus"); if (!box) return; box.hidden = !o; if (!o) return;
    // Ecliptic latitude: how far it sits from the plane where ordinary asteroids crowd.
    const [x, y, z] = o.v, beta = Math.asin(Math.max(-1, Math.min(1, z * Math.cos(EPS) - y * Math.sin(EPS)))) * 180 / Math.PI;
    const ago = o.t ? (Date.now() - o.t) / 3.6e6 : null, word = o.p >= 0.8 ? "likely real" : o.p >= 0.4 ? "uncertain" : "probably not real";
    box.innerHTML = `<div class="gf-k">SELECTED CANDIDATE</div><b class="gf-name">${o.desig}</b>
      <div class="gf-row"><span>P(real NEO)</span><b>${o.p.toFixed(2)} · ${word}</b></div>
      <div class="gf-row"><span>Brightness</span><b>magnitude ${o.mag}</b></div>
      <div class="gf-row"><span>Distance from the ecliptic</span><b>${Math.abs(beta).toFixed(0)}°${Math.abs(beta) > 25 ? " · far from the main belt's crowd" : ""}</b></div>
      ${ago != null ? `<div class="gf-row"><span>Last measured</span><b>${ago < 48 ? Math.round(ago) + " hours" : Math.round(ago / 24) + " days"} ago</b></div>` : ""}
      <div class="gf-row"><span>Rank tonight</span><b>${o.rank + 1} of ${objs.length}</b></div>`;
  }
  document.getElementById("globe-top")?.addEventListener("click", () => objs.length && focusOn(objs[0]));
  addEventListener("ss-sky-pick", (e) => { const o = objs.find((q) => q.desig === e.detail.desig); if (o) focusOn(o); }); // picked on the flat map
  cv.addEventListener("pointerleave", () => { mouse = null; });
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(cv);

  // Data: the dashboard embeds its candidates (window.SS_SKY); the home page reads them from the live dashboard.
  const load = (list, upd) => {
    objs = list.slice().sort((a, b) => b.p - a.p).map((o, i) => ({ desig: o.desig, p: o.p, mag: o.v || "?", v: vec(o.ra, o.dec), t: o.t, rank: i }));
    const hi = objs.filter((o) => o.p >= 0.8).length, cnt = document.getElementById("globe-count");
    if (cnt) cnt.innerHTML = `<b>${objs.length}</b> candidates right now · <b>${hi}</b> likely real${upd ? ` · captured ${new Date(upd).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}` : ""}`;
    if (reduce) draw(0);
  };
  if (window.SS_SKY) load(window.SS_SKY, document.querySelector('meta[name="ss-updated"]')?.content);
  else fetch("/outputs/dashboard.html").then((r) => r.text()).then((html) => {
    const m = html.match(/var objs = (\[[\s\S]*?\]), cfg/);
    if (!m) throw new Error("no sky data");
    load(JSON.parse(m[1]), new DOMParser().parseFromString(html, "text/html").querySelector('meta[name="ss-updated"]')?.content);
  }).catch(() => { const cnt = document.getElementById("globe-count"); if (cnt) cnt.textContent = "Live positions couldn't be loaded right now; the sky grid is still yours to turn."; });
  requestAnimationFrame(loop);
}
