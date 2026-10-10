// Runs the spectrum match off the main thread so the page stays responsive.
import { fitTemplates, getLibrary } from "./upload.js";

self.onmessage = async (e) => {
  try {
    self.postMessage({ ok: true, fit: fitTemplates(e.data, await getLibrary()) });
  } catch (err) {
    self.postMessage({ ok: false, error: err.message });
  }
};
