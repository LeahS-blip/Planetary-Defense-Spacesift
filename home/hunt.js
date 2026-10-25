// Your turn: find a hidden planet by folding a star's light curve on a trial orbit, then race the search.
// The star is simulated (white noise, optional stellar rhythm, box-shaped transits). The "signal strength"
// is the same quantity a box search ranks orbits by: the deepest dip in the folded data, in units of its noise.
const root = document.getElementById("hunt");
if (root) init();

function init() {
  const $ = (id) => document.getElementById(id);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const DPR = Math.min(2, devicePixelRatio || 1);
  const PMIN = 1.5, PMAX = 14, STEPS = 4000, THRESH = 7;
  const toP = (v) => PMIN * Math.exp((v / STEPS) * Math.log(PMAX / PMIN));
  const toV = (P) => Math.round((Math.log(P / PMIN) / Math.log(PMAX / PMIN)) * STEPS);
  let star, mode = "quiet", clean = false, t0 = 0, solved = false, seed = (Date.now() % 100000) + 1;
  let last = null; // the latest find, for the share card
  let trace = new Float32Array(STEPS + 1); // signal strength at every orbit the visitor has passed over

  function rng(s) { return () => ((s = (s * 16807) % 2147483647) / 2147483647); }
  function makeStar(hard) {
    const r = rng(seed++), g = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(6.2832 * r());
    const DAYS = 90, N = 2200, P = 2.2 + r() * 9.5, T0 = r() * P, dur = 0.09 * Math.cbrt(P / 2) + 0.05;
    const depth = 13 / Math.sqrt((N * dur) / P); // all the dips together stand ~13 sigma above the noise
    const rh = hard ? [[0.7 + r() * 1.3, 2.4 + r(), r() * 6.28], [2.2 + r() * 2.5, 1.6 + r(), r() * 6.28], [9 + r() * 6, 1.2, r() * 6.28]] : [];
    const t = [], f = [], v = [];
    for (let i = 0; i < N; i++) {
      let tt = (i / N) * DAYS; if (tt > 41 && tt < 44) continue; // a gap in the data, like a real mission
      tt += (r() - 0.5) * (DAYS / N);
      let ph = ((tt - T0) / P) % 1; if (ph < 0) ph += 1;
      const inT = Math.min(ph, 1 - ph) * P < dur / 2;
      t.push(tt); f.push(g() - (inT ? depth : 0)); v.push(rh.reduce((a, [p, amp, o]) => a + amp * Math.sin((6.2832 * tt) / p + o), 0));
    }
    return { t, f, v, P, dur, DAYS, hard };
  }
  const data = () => star.f.map((x, i) => x + (star.hard && !clean ? star.v[i] : 0));

  // Fold on a trial period: bin the phases and find the deepest run of 1-4 bins (a box-shaped dip).
  const NB = 90;
  function fold(y, P) {
    const s = new Float64Array(NB), n = new Uint32Array(NB), t = star.t;
    let mean = 0; for (const v of y) mean += v; mean /= y.length;
    let sd = 0; for (const v of y) sd += (v - mean) ** 2; sd = Math.sqrt(sd / y.length);
    for (let i = 0; i < t.length; i++) { const b = Math.floor(((t[i] / P) % 1) * NB); s[b] += y[i] - mean; n[b]++; }
    let best = 0, at = 0, w = 1;
    for (let k = 1; k <= 4; k++) for (let b = 0; b < NB; b++) {
      let ss = 0, nn = 0; for (let j = 0; j < k; j++) { const q = (b + j) % NB; ss += s[q]; nn += n[q]; }
      if (!nn) continue;
      const snr = (-ss / nn) / (sd / Math.sqrt(nn));
      if (snr > best) { best = snr; at = b; w = k; }
    }
    return { snr: best, at, w, s, n, mean, sd };
  }

  function fit(c) {
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * DPR) || c.height !== Math.round(h * DPR)) { c.width = Math.round(w * DPR); c.height = Math.round(h * DPR); }
    const ctx = c.getContext("2d"); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, w, h); return { ctx, w, h };
  }
  function drawRaw() {
    const { ctx, w, h } = fit($("hunt-raw")), y = data(), sc = star.hard && !clean ? 8 : 4.5;
    ctx.fillStyle = "rgba(201,208,228,.55)";
    for (let i = 0; i < y.length; i++) ctx.fillRect((star.t[i] / star.DAYS) * w - 0.7, h / 2 - (y[i] / sc) * (h / 2) - 0.7, 1.4, 1.4);
    ctx.fillStyle = "#a3abc4"; ctx.font = "11px IBM Plex Sans, sans-serif"; ctx.fillText("brightness over 90 days", 6, 13);
  }
  function drawFold(P) {
    const { ctx, w, h } = fit($("hunt-fold")), y = data(), F = fold(y, P), sc = star.hard && !clean ? 8 : 4;
    const shift = (F.at + F.w / 2) / NB - 0.5; // put the deepest dip in the middle
    const X = (ph) => (((ph - shift) % 1 + 1) % 1) * w;
    ctx.fillStyle = "rgba(201,208,228,.4)";
    for (let i = 0; i < y.length; i++) ctx.fillRect(X((star.t[i] / P) % 1) - 0.7, h / 2 - ((y[i] - F.mean) / sc) * (h / 2) - 0.7, 1.4, 1.4);
    const good = F.snr >= THRESH;
    ctx.strokeStyle = good ? "#4ade80" : "#9fb6ff"; ctx.lineWidth = 2.2; ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = good ? 12 : 6;
    const pts = []; for (let b = 0; b < NB; b++) if (F.n[b]) pts.push([X((b + 0.5) / NB), h / 2 - (F.s[b] / F.n[b] / sc) * (h / 2)]);
    pts.sort((a, b) => a[0] - b[0]); ctx.beginPath(); pts.forEach(([x, yy], i) => (i ? ctx.lineTo(x, yy) : ctx.moveTo(x, yy))); ctx.stroke(); ctx.shadowBlur = 0;
    ctx.fillStyle = "#a3abc4"; ctx.font = "11px IBM Plex Sans, sans-serif"; ctx.fillText(`every point stacked on a ${P.toFixed(3)}-day orbit`, 6, 13);
    // meter
    const k = Math.min(1, F.snr / 14);
    $("hunt-meter").style.setProperty("--k", k); $("hunt-meter").classList.toggle("good", good);
    $("hunt-snr").innerHTML = good ? `<b>${F.snr.toFixed(1)}</b> · strong enough to claim` : `<b>${F.snr.toFixed(1)}</b> · a detection needs ${THRESH}`;
    return F;
  }
  function setP(P) {
    if (!star) return newStar(); // the controls work even before the section has scrolled into view
    const v = toV(P); $("hunt-p").value = v; $("hunt-pv").textContent = `${P.toFixed(3)} days`;
    const F = drawFold(P); trace[v] = Math.max(trace[v], F.snr); drawTrace(v);
    dispatchEvent(new CustomEvent("ss-sound", { detail: { type: "tick", snr: F.snr } }));
  }
  // The trace under the slider: a periodogram the visitor builds by hand as they sweep.
  function drawTrace(v) {
    const { ctx, w, h } = fit($("hunt-trace")), per = Math.max(1, Math.floor(STEPS / w));
    for (let x = 0; x < w; x++) {
      let m = 0; const a = Math.floor((x / w) * STEPS); for (let j = a; j < a + per && j <= STEPS; j++) m = Math.max(m, trace[j]);
      if (!m) continue;
      ctx.fillStyle = m >= THRESH ? "#4ade80" : "rgba(122,162,255,.55)";
      const bh = Math.min(h, (m / 14) * h); ctx.fillRect(x, h - bh, 1, bh);
    }
    ctx.fillStyle = "rgba(248,113,113,.5)"; ctx.fillRect(0, h - (THRESH / 14) * h, w, 1);
    ctx.fillStyle = "#eef2fc"; ctx.fillRect((v / STEPS) * w - 1, 0, 2, h);
  }
  const curP = () => toP(+$("hunt-p").value);

  function newStar() {
    star = makeStar(mode === "flicker"); clean = false; solved = false; trace = new Float32Array(STEPS + 1);
    $("hunt-clean").checked = false; $("hunt-clean-wrap").hidden = mode !== "flicker";
    $("hunt-result").hidden = true; $("hunt-claim").disabled = false;
    setP(PMIN * 2.1 + Math.random() * 1.5); drawRaw(); t0 = performance.now();
  }
  function periodogram() {
    const y = data(), t1 = performance.now(), n = 3000, out = [];
    for (let i = 0; i < n; i++) { const P = PMIN * Math.exp((i / (n - 1)) * Math.log(PMAX / PMIN)); out.push([P, fold(y, P).snr]); }
    return { out, ms: performance.now() - t1, n };
  }
  function drawPgram(pg, guess) {
    const { ctx, w, h } = fit($("hunt-pgram")), max = Math.max(...pg.out.map((q) => q[1]), THRESH + 2);
    const X = (P) => (Math.log(P / PMIN) / Math.log(PMAX / PMIN)) * w, Y = (s) => h - 16 - (s / max) * (h - 30);
    ctx.strokeStyle = "rgba(248,113,113,.6)"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(0, Y(THRESH)); ctx.lineTo(w, Y(THRESH)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#7aa2ff"; ctx.lineWidth = 1.2; ctx.beginPath(); pg.out.forEach(([P, s], i) => (i ? ctx.lineTo(X(P), Y(s)) : ctx.moveTo(X(P), Y(s)))); ctx.stroke();
    const mark = (P, col, label, up) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(P), 14); ctx.lineTo(X(P), h - 16); ctx.stroke();
      ctx.fillStyle = col; ctx.font = "600 11px IBM Plex Sans, sans-serif"; ctx.textAlign = X(P) > w - 90 ? "right" : "left"; ctx.fillText(label, X(P) + (ctx.textAlign === "right" ? -5 : 5), up ? 12 : 26); ctx.textAlign = "left"; };
    mark(star.P, "#ffb45e", "the planet", true); mark(guess, "#eef2fc", "your answer", false);
    ctx.fillStyle = "#a3abc4"; ctx.font = "10.5px IBM Plex Sans, sans-serif"; ctx.fillText(`${PMIN} days`, 2, h - 3); ctx.textAlign = "right"; ctx.fillText(`${PMAX} days`, w - 2, h - 3); ctx.textAlign = "left";
  }
  function claim() {
    const P = curP(), F = fold(data(), P), secs = (performance.now() - t0) / 1000, rel = (x) => Math.abs(P - x) / x;
    const fmtS = (x) => (x < 60 ? Math.round(x) + " s" : Math.round(x / 60) + " min");
    const verdict = F.snr < THRESH ? ["miss", "Not strong enough yet", "The dip at that orbit doesn't stand out from the noise. Keep sliding, or fine-tune with the arrows."]
      : rel(star.P) < 0.006 ? ["win", `Found it in ${secs < 60 ? Math.round(secs) + " seconds" : Math.round(secs / 60) + " min"}!`, `Your orbit: ${P.toFixed(3)} days. The real one: ${star.P.toFixed(3)} days.`]
      : rel(star.P / 2) < 0.006 || rel(star.P * 2) < 0.006 ? ["half", "Close: that's an echo of the planet", `You found ${P < star.P ? "half" : "twice"} the real orbit (${star.P.toFixed(3)} days). Searches fall for this too.`]
      : ["miss", "That's noise, not the planet", star.hard && !clean ? "On a flickering star the star itself makes fake dips. Try removing its rhythm first." : "A random cluster of low points lined up. Real planets repeat exactly."];
    $("hunt-result").hidden = false; $("hunt-result").className = "hunt-result " + verdict[0];
    $("hunt-vh").textContent = verdict[1]; $("hunt-vp").textContent = verdict[2];
    if (verdict[0] === "miss" && F.snr < THRESH) { $("hunt-race").hidden = true; return; }
    const pg = periodogram(); $("hunt-race").hidden = false; drawPgram(pg, P);
    const best = pg.out.reduce((a, b) => (b[1] > a[1] ? b : a));
    $("hunt-cmp").innerHTML = `The computer tried <b>${pg.n.toLocaleString("en-US")} orbits in ${Math.max(1, Math.round(pg.ms))} ms</b> and its top pick was ${best[0].toFixed(3)} days${Math.abs(best[0] - star.P) / star.P < 0.006 ? ", the planet" : Math.abs(best[0] - star.P / 2) / star.P < 0.006 || Math.abs(best[0] - 2 * star.P) / star.P < 0.006 ? ", an echo of the planet" : ": <b>wrong</b>, fooled by the star"}. That's why SpaceSift tests searches on thousands of planets with known answers.`;
    dispatchEvent(new CustomEvent("ss-sound", { detail: { type: verdict[0] === "win" ? "found" : "miss" } }));
    $("hunt-share").hidden = verdict[0] !== "win";
    if (verdict[0] === "win") {
      solved = true; $("hunt-claim").disabled = true; if (!reduce) burst();
      const key = "ss-hunt-best-" + mode; let best = null; try { best = parseFloat(localStorage.getItem(key)); } catch {}
      const record = !(best <= secs); if (record) try { localStorage.setItem(key, String(secs)); } catch {}
      $("hunt-best").textContent = record ? (best ? `New personal best! (was ${fmtS(best)})` : "Your first find. That's your time to beat.") : `Your best on this kind of star: ${fmtS(best)}.`;
      last = { secs, P, mode };
    } else $("hunt-best").textContent = "";
  }
  function burst() { // a little celebration over the folded plot
    const c = $("hunt-fold"), r = c.getBoundingClientRect(), fx = document.createElement("canvas");
    fx.className = "hunt-fx"; fx.width = r.width * DPR; fx.height = r.height * DPR; c.parentElement.appendChild(fx);
    const ctx = fx.getContext("2d"); ctx.scale(DPR, DPR);
    const ps = Array.from({ length: 90 }, () => ({ x: r.width / 2, y: r.height / 2, vx: (Math.random() - 0.5) * 9, vy: (Math.random() - 0.8) * 8, c: ["#4ade80", "#7aa2ff", "#ffb45e", "#eef2fc"][Math.random() * 4 | 0] }));
    let f = 0; (function step() { ctx.clearRect(0, 0, r.width, r.height); ps.forEach((p) => { p.x += p.vx; p.y += p.vy; p.vy += 0.25; ctx.globalAlpha = 1 - f / 60; ctx.fillStyle = p.c; ctx.fillRect(p.x, p.y, 3, 3); });
      if (++f < 60) requestAnimationFrame(step); else fx.remove(); })();
  }

  // Listen: the folded data as sound, dip in the middle, played three times over.
  function listen() {
    if (!star) newStar();
    const tog = document.getElementById("sound-toggle"); if (tog && tog.getAttribute("aria-pressed") !== "true") tog.click();
    const y = data(), F = fold(y, curP()), shift = Math.round(F.at + F.w / 2 - NB / 2), bins = [];
    for (let b = 0; b < NB; b++) { const q = (b + shift + NB) % NB; bins.push(F.n[q] ? F.s[q] / F.n[q] / F.sd * 3 : 0); }
    $("hunt-listen").classList.add("playing");
    dispatchEvent(new CustomEvent("ss-sound", { detail: { type: "sonify", bins, reps: 2, done: () => $("hunt-listen").classList.remove("playing") } }));
  }
  // Share card: the folded dip, the time, the site. Shared as an image where the browser can, saved otherwise.
  async function share() {
    if (!last) return;
    const W = 1200, H = 630, c = document.createElement("canvas"); c.width = W; c.height = H; const g = c.getContext("2d");
    const bg = g.createLinearGradient(0, 0, W, H); bg.addColorStop(0, "#0b1430"); bg.addColorStop(1, "#04060d"); g.fillStyle = bg; g.fillRect(0, 0, W, H);
    for (let i = 0; i < 260; i++) { g.globalAlpha = Math.random() * 0.7; g.fillStyle = "#dfe6ff"; g.fillRect(Math.random() * W, Math.random() * H, 1.5, 1.5); } g.globalAlpha = 1;
    g.fillStyle = "#7aa2ff"; g.font = "700 22px IBM Plex Mono, sans-serif"; g.fillText("S P A C E S I F T", 70, 90);
    g.fillStyle = "#eef2fc"; g.font = "700 64px IBM Plex Sans, sans-serif"; g.fillText(`I found a planet in ${last.secs < 60 ? Math.round(last.secs) + " seconds" : Math.round(last.secs / 60) + " minutes"}.`, 70, 190);
    g.fillStyle = "#c9d0e4"; g.font = "28px IBM Plex Sans, sans-serif";
    g.fillText(`Hidden in ${last.mode === "flicker" ? "a flickering star" : "a star"}'s light, orbiting every ${last.P.toFixed(3)} days.`, 70, 245);
    const y = data(), F = fold(y, last.P), x0 = 70, x1 = W - 70, mid = 430, sc = 22, shift = (F.at + F.w / 2) / NB - 0.5;
    g.fillStyle = "rgba(201,208,228,.45)";
    for (let i = 0; i < y.length; i++) { const ph = ((((star.t[i] / last.P) % 1) - shift) % 1 + 1) % 1; g.fillRect(x0 + ph * (x1 - x0), mid - (y[i] - F.mean) * sc, 2.4, 2.4); }
    g.strokeStyle = "#4ade80"; g.lineWidth = 5; g.shadowColor = "#4ade80"; g.shadowBlur = 18; g.beginPath();
    const pts = []; for (let b = 0; b < NB; b++) if (F.n[b]) pts.push([x0 + ((((b + 0.5) / NB - shift) % 1 + 1) % 1) * (x1 - x0), mid - (F.s[b] / F.n[b]) * sc]);
    pts.sort((a, b) => a[0] - b[0]).forEach(([x, yy], i) => (i ? g.lineTo(x, yy) : g.moveTo(x, yy))); g.stroke(); g.shadowBlur = 0;
    g.fillStyle = "#a3abc4"; g.font = "24px IBM Plex Sans, sans-serif"; g.fillText("Can you beat me? spacesift.vercel.app/#hunt", 70, H - 50);
    const blob = await new Promise((r) => c.toBlob(r, "image/png")), file = new File([blob], "spacesift-planet.png", { type: "image/png" });
    if (navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ files: [file], text: "I found a hidden planet on SpaceSift. Can you beat my time?", url: location.origin + "/#hunt" }); return; } catch {} }
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "spacesift-planet.png"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  $("hunt-listen").onclick = listen; $("hunt-share").onclick = share;
  $("hunt-p").max = STEPS;
  $("hunt-p").addEventListener("input", () => setP(curP()));
  $("hunt-minus").onclick = () => setP(curP() * 0.9995); $("hunt-plus").onclick = () => setP(curP() * 1.0005);
  $("hunt-claim").onclick = () => (star ? claim() : newStar()); $("hunt-new").onclick = newStar;
  $("hunt-clean").onchange = (e) => { clean = e.target.checked; trace = new Float32Array(STEPS + 1); drawRaw(); setP(curP()); };
  $("hunt-modes").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; mode = b.dataset.m;
    for (const x of $("hunt-modes").children) x.setAttribute("aria-pressed", String(x === b)); newStar(); };
  addEventListener("resize", () => { if (star) { drawRaw(); setP(curP()); } });
  addEventListener("keydown", (e) => { if (document.activeElement === $("hunt-p") && e.shiftKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) { e.preventDefault(); setP(curP() * (e.key === "ArrowLeft" ? 0.9995 : 1.0005)); } });
  // Build the first star when the section comes near, so it costs nothing on load.
  new IntersectionObserver((es, o) => { if (es[0].isIntersecting) { o.disconnect(); if (!star) newStar(); } }, { rootMargin: "300px" }).observe(root);
}
