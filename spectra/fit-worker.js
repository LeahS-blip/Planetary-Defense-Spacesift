// Runs the spectrum match off the main thread so the page stays responsive.
import { analyze } from "./analyze.js";

self.onmessage = async (e) => {
  try {
    self.postMessage({ ok: true, ...(await analyze(e.data)) });
  } catch (err) {
    self.postMessage({ ok: false, error: err.message });
  }
};
