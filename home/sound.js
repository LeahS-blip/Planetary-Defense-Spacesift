// Opt-in sound for the landing page, synthesized with Web Audio (no files). Everything is tuned to one key
// (A major pentatonic) and runs through a generated reverb and a gentle limiter, so it stays soft and musical:
// a breathing chord pad, glassy detector notes that climb as the hunt's signal strengthens, a bell for a find,
// and a gliding tone that turns a folded light curve into sound.
// Other scripts ask for sounds with: dispatchEvent(new CustomEvent("ss-sound", { detail: { type, ... } })).
const btn = document.getElementById("sound-toggle");
let ac = null, out = null, wet = null, dry = null, pad = null, on = false, lastNote = -1, lastAt = 0;
try { on = localStorage.getItem("ss-sound") === "on"; } catch {}
const A = 440, PENTA = [0, 2, 4, 7, 9]; // A major pentatonic, in semitones
const note = (i, base = A) => base * Math.pow(2, (Math.floor(i / 5) * 12 + PENTA[((i % 5) + 5) % 5]) / 12);

function build() {
  ac = new (window.AudioContext || window.webkitAudioContext)();
  const lim = ac.createDynamicsCompressor(); lim.threshold.value = -18; lim.knee.value = 12; lim.ratio.value = 6; lim.attack.value = 0.003; lim.release.value = 0.25;
  out = ac.createGain(); out.gain.value = 0; out.connect(lim).connect(ac.destination);
  // Reverb: a stereo impulse of decaying, slightly darkened noise (about 3 s).
  const len = Math.floor(ac.sampleRate * 3.2), ir = ac.createBuffer(2, len, ac.sampleRate);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); let lp = 0; for (let i = 0; i < len; i++) { lp = lp * 0.6 + (Math.random() * 2 - 1) * 0.4; d[i] = lp * Math.pow(1 - i / len, 2.6); } }
  const verb = ac.createConvolver(); verb.buffer = ir;
  wet = ac.createGain(); wet.gain.value = 0.55; dry = ac.createGain(); dry.gain.value = 0.8;
  wet.connect(verb).connect(out); dry.connect(out);
}
function ctx() { if (!ac) build(); if (ac.state === "suspended") ac.resume(); return ac; }
const send = (node, w = 0.5) => { const d = ac.createGain(), s = ac.createGain(); d.gain.value = 1 - w * 0.5; s.gain.value = w; node.connect(d).connect(dry); node.connect(s).connect(wet); };

// The pad: an airy A-major-ninth chord of soft sines, each voice swelling on its own slow cycle.
function startPad() {
  if (pad) return;
  const a = ctx(), bus = a.createGain(), lp = a.createBiquadFilter();
  bus.gain.value = 0; lp.type = "lowpass"; lp.frequency.value = 1400; lp.Q.value = 0.3;
  bus.connect(lp); send(lp, 0.85);
  const voices = [220, 277.18, 329.63, 493.88, 554.37].map((f, i) => {
    const o = a.createOscillator(), o2 = a.createOscillator(), g = a.createGain(), lfo = a.createOscillator(), lg = a.createGain();
    o.frequency.value = f; o2.frequency.value = f * 1.003; // a slow chorus between two near-unison sines
    g.gain.value = 0.5; lfo.frequency.value = 0.035 + i * 0.013; lg.gain.value = 0.45; lfo.connect(lg).connect(g.gain);
    o.connect(g); o2.connect(g); g.connect(bus); o.start(); o2.start(); lfo.start();
    return [o, o2, lfo];
  });
  bus.gain.linearRampToValueAtTime(0.009, a.currentTime + 4); // well under the interaction sounds
  pad = { bus, voices };
}
function stopPad() { if (!pad) return; const p = pad; pad = null; p.bus.gain.setTargetAtTime(0, ac.currentTime, 0.5); setTimeout(() => p.voices.flat().forEach((o) => o.stop()), 3000); }

// A glassy plink: a sine with a quiet octave partial and a fast, smooth decay.
function plink(f, vol = 0.05, len = 0.5, w = 0.45, when = 0) {
  if (!on || !ac) return;
  const t = ac.currentTime + when, g = ac.createGain();
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vol, t + 0.004); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  for (const [m, a] of [[1, 1], [2, 0.18], [3.01, 0.05]]) { const o = ac.createOscillator(), og = ac.createGain(); o.frequency.value = f * m; og.gain.value = a; o.connect(og).connect(g); o.start(t); o.stop(t + len + 0.05); }
  send(g, w);
}
// A bell: frequency modulation at an inharmonic ratio whose brightness fades as the note rings.
function bell(f, vol = 0.07, len = 2.2, when = 0) {
  if (!on || !ac) return;
  const t = ac.currentTime + when, car = ac.createOscillator(), mod = ac.createOscillator(), mg = ac.createGain(), g = ac.createGain();
  car.frequency.value = f; mod.frequency.value = f * 3.5;
  mg.gain.setValueAtTime(f * 2.2, t); mg.gain.exponentialRampToValueAtTime(f * 0.05, t + len * 0.6);
  mod.connect(mg).connect(car.frequency); car.connect(g);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(vol, t + 0.006); g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  send(g, 0.6); car.start(t); mod.start(t); car.stop(t + len + 0.1); mod.stop(t + len + 0.1);
}
// A light curve as sound: a smooth tone whose pitch follows the brightness, so a stacked transit is heard as a
// repeated, gliding dip. Bins are lightly smoothed first so the noise doesn't chatter.
function sonify(bins, reps = 2) {
  if (!on || !ac) return 0;
  const sm = bins.map((_, i) => (bins[(i - 1 + bins.length) % bins.length] + 2 * bins[i] + bins[(i + 1) % bins.length]) / 4);
  const seg = 0.026, t0 = ac.currentTime + 0.08, o = ac.createOscillator(), lp = ac.createBiquadFilter(), g = ac.createGain();
  o.type = "sine"; lp.type = "lowpass"; lp.frequency.value = 2200;
  o.connect(lp).connect(g); send(g, 0.3);
  let t = t0; o.frequency.setValueAtTime(523.25, t0);
  for (let r = 0; r < reps; r++) for (const b of sm) { o.frequency.setTargetAtTime(523.25 * Math.pow(2, Math.max(-1, Math.min(0.5, b * 0.16))), t, 0.012); t += seg; }
  g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(0.065, t0 + 0.12); g.gain.setValueAtTime(0.065, t - 0.15); g.gain.linearRampToValueAtTime(0, t);
  o.start(t0); o.stop(t + 0.1);
  return t - t0;
}

function setOn(v) {
  on = v;
  try { localStorage.setItem("ss-sound", v ? "on" : "off"); } catch {}
  btn.setAttribute("aria-pressed", String(v)); btn.querySelector("span:last-child").textContent = v ? "Sound: on" : "Sound: off";
  if (v) { const a = ctx(); out.gain.cancelScheduledValues(a.currentTime); out.gain.setTargetAtTime(1, a.currentTime, 0.3); startPad(); plink(note(7), 0.05, 0.9); plink(note(9), 0.04, 1.1, 0.5, 0.09); }
  else if (ac) { out.gain.cancelScheduledValues(ac.currentTime); out.gain.setTargetAtTime(0, ac.currentTime, 0.15); stopPad(); }
}
addEventListener("ss-sound", (e) => {
  if (!on || !ac) return;
  const d = e.detail || {};
  if (d.type === "tick") {
    // The detector: the signal picks a note on the scale, so sweeping the slider plays a soft rising melody.
    const k = Math.max(0, Math.min(1, d.snr / 12)), i = Math.round(k * 11), now = performance.now();
    if (i === lastNote && now - lastAt < 260) return;
    if (now - lastAt < 70) return;
    lastNote = i; lastAt = now;
    plink(note(i), 0.025 + 0.03 * k, 0.35 + 0.4 * k, 0.4);
    if (d.snr >= 7) plink(note(i + 2, A), 0.02, 0.8, 0.7, 0.03); // past the threshold: a shimmering note above
  } else if (d.type === "found") {
    [0, 2, 4, 5, 7].forEach((n, j) => bell(note(n + 5), 0.06, 2.4, j * 0.11));
  } else if (d.type === "miss") {
    plink(note(4, 220), 0.05, 0.6, 0.5); plink(note(2, 220), 0.045, 0.9, 0.55, 0.16);
  } else if (d.type === "focus") {
    bell(note(7), 0.045, 1.6);
  } else if (d.type === "sonify") {
    const dur = sonify(d.bins, d.reps); d.done && setTimeout(d.done, dur * 1000);
  }
});
btn.hidden = false;
btn.addEventListener("click", () => setOn(!on));
btn.setAttribute("aria-pressed", String(on)); btn.querySelector("span:last-child").textContent = on ? "Sound: on" : "Sound: off";
// Browsers only allow audio after a gesture: if sound was left on, start it at the first click or key.
if (on) { const go = (e) => { if (btn.contains(e.target)) { on = false; } else setOn(true); removeEventListener("pointerdown", go); removeEventListener("keydown", go); }; addEventListener("pointerdown", go); addEventListener("keydown", go); }
export const soundOn = () => on;
