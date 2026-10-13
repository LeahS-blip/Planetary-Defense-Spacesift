// Opt-in sound for the landing page, synthesized with Web Audio (no files): a quiet ambient drone, a detector
// that ticks higher as the planet hunt's signal strengthens, a chime for a find, and light-curve sonification.
// Other scripts ask for sounds with: dispatchEvent(new CustomEvent("ss-sound", { detail: { type, ... } })).
const btn = document.getElementById("sound-toggle");
let ac = null, master = null, drone = null, on = false, lastTick = 0;
try { on = localStorage.getItem("ss-sound") === "on"; } catch {}

function ctx() {
  if (!ac) {
    ac = new (window.AudioContext || window.webkitAudioContext)();
    master = ac.createGain(); master.gain.value = 0; master.connect(ac.destination);
  }
  if (ac.state === "suspended") ac.resume();
  return ac;
}
function startDrone() {
  if (drone) return;
  const a = ctx(), lp = a.createBiquadFilter(), g = a.createGain(), lfo = a.createOscillator(), lg = a.createGain();
  lp.type = "lowpass"; lp.frequency.value = 520; g.gain.value = 0.05;
  lfo.frequency.value = 0.06; lg.gain.value = 180; lfo.connect(lg).connect(lp.frequency); lfo.start();
  const oscs = [55, 82.4, 110.2, 164.8].map((f, i) => { const o = a.createOscillator(); o.type = i % 2 ? "triangle" : "sine"; o.frequency.value = f; o.detune.value = (i - 1.5) * 6; o.connect(lp); o.start(); return o; });
  lp.connect(g).connect(master);
  drone = { oscs, lfo, g };
}
function setOn(v) {
  on = v;
  try { localStorage.setItem("ss-sound", v ? "on" : "off"); } catch {}
  btn.setAttribute("aria-pressed", String(v)); btn.querySelector("span:last-child").textContent = v ? "Sound on" : "Sound off";
  if (v) { const a = ctx(); startDrone(); master.gain.cancelScheduledValues(a.currentTime); master.gain.linearRampToValueAtTime(0.9, a.currentTime + 1.2); blip(660, 0.08, 0.12); }
  else if (ac) { master.gain.cancelScheduledValues(ac.currentTime); master.gain.linearRampToValueAtTime(0, ac.currentTime + 0.4); }
}
function blip(freq, len = 0.06, vol = 0.08, type = "sine", when = 0) {
  if (!on || !ac) return;
  const t = ac.currentTime + when, o = ac.createOscillator(), g = ac.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, t);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vol, t + 0.005); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  o.connect(g).connect(master); o.start(t); o.stop(t + len + 0.02);
}
// A light curve as sound: brightness sets the pitch, so a stacked transit is heard as a repeated dip.
function sonify(bins, reps = 3) {
  if (!on || !ac) return 0;
  const seg = 0.028, t0 = ac.currentTime + 0.05, o = ac.createOscillator(), g = ac.createGain();
  o.type = "triangle"; g.gain.value = 0; o.connect(g).connect(master);
  let t = t0;
  for (let r = 0; r < reps; r++) for (const b of bins) { o.frequency.setValueAtTime(Math.max(90, 440 + b * 120), t); t += seg; }
  g.gain.linearRampToValueAtTime(0.12, t0 + 0.05); g.gain.setValueAtTime(0.12, t - 0.05); g.gain.linearRampToValueAtTime(0, t);
  o.start(t0); o.stop(t + 0.05);
  return t - t0;
}
addEventListener("ss-sound", (e) => {
  if (!on) return;
  const d = e.detail || {};
  if (d.type === "tick") { // the detector: ticks rise in pitch with the signal, and sparkle past the threshold
    const now = performance.now(); if (now - lastTick < 45) return; lastTick = now;
    const k = Math.min(1, d.snr / 14);
    blip(180 + k * k * 1400, 0.035, 0.03 + 0.05 * k, d.snr >= 7 ? "sine" : "square");
  } else if (d.type === "found") {
    [523.25, 659.25, 783.99, 1046.5, 1318.5].forEach((f, i) => blip(f, 0.5, 0.09, "sine", i * 0.09));
  } else if (d.type === "miss") { blip(220, 0.18, 0.06, "triangle"); blip(185, 0.25, 0.05, "triangle", 0.12); }
  else if (d.type === "focus") { blip(880, 0.12, 0.05); blip(1320, 0.2, 0.04, "sine", 0.06); }
  else if (d.type === "sonify") { const dur = sonify(d.bins, d.reps); d.done && setTimeout(d.done, dur * 1000); }
});
btn.hidden = false;
btn.addEventListener("click", () => setOn(!on));
btn.setAttribute("aria-pressed", String(on)); btn.querySelector("span:last-child").textContent = on ? "Sound on" : "Sound off";
// Browsers only allow audio after a gesture: if sound was left on, start it at the first click or key.
if (on) { const go = (e) => { if (btn.contains(e.target)) { on = false; } else setOn(true); removeEventListener("pointerdown", go); removeEventListener("keydown", go); }; addEventListener("pointerdown", go); addEventListener("keydown", go); }
export const soundOn = () => on;
