// One entry point for the worker (and the main-thread fallback): data files and pictures.
import { fitTemplates, getLibrary } from "./upload.js";
import { readImageSpectrum, profileToSpectrum } from "./image.js";

/**
 * m.kind: "data" ({obs: {wl, flux}}), "image" ({img: ImageData-like}) or "recal" ({profile, cal}: a picture
 * re-read with a wavelength scale the user typed in).
 */
export async function analyze(m) {
  const lib = await getLibrary();
  if (m.kind === "image" || m.kind === "recal") {
    const r = m.kind === "image" ? readImageSpectrum(m.img) : { profile: m.profile, cal: m.cal, ...profileToSpectrum(m.profile, m.cal) };
    // Pictures show where lines are but not their true strength or the star's real colors, so match the
    // line pattern only.
    return { fit: fitTemplates(r, lib, { normalized: true }), image: { profile: r.profile, cal: r.cal } };
  }
  return { fit: fitTemplates(m.obs, lib) };
}
