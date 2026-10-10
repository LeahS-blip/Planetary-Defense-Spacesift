// Runs the spectrum fit off the main thread so the page stays responsive.
import { fitStar } from "./upload.js";

self.onmessage = (e) => {
  try {
    self.postMessage({ ok: true, fit: fitStar(e.data) });
  } catch (err) {
    self.postMessage({ ok: false, error: err.message });
  }
};
