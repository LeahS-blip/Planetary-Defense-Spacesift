// Shared helpers for the physics labs: accessible charts, live announcements, motion preferences.

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
export const scrollToEl = (el) => el.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });

// Give a Plotly chart a name and a plain-language description. The chart itself is hidden from
// screen readers (its SVG is a maze of unlabelled paths); the description sits in the page as text.
export function describe(id, title, text) {
  const el = document.getElementById(id);
  el.setAttribute("role", "img");
  el.setAttribute("aria-label", `${title}. ${text}`);
  let d = el.parentElement.querySelector(`.chart-desc[data-for="${id}"]`);
  if (!d) {
    d = document.createElement("p");
    d.className = "chart-desc"; d.dataset.for = id;
    el.after(d);
  }
  d.textContent = text;
}

// A collapsible data table under a chart, so every number is reachable without a mouse.
export function table(id, caption, head, rows) {
  const el = document.getElementById(id);
  let t = el.parentElement.querySelector(`details.table[data-for="${id}"]`);
  if (!t) {
    t = document.createElement("details");
    t.className = "table"; t.dataset.for = id;
    (el.parentElement.querySelector(`.chart-desc[data-for="${id}"]`) || el).after(t);
  }
  const open = t.open;
  t.innerHTML = `<summary>Show the data as a table</summary><div class="wrap" tabindex="0" role="region" aria-label="${caption}"><table><caption class="sr-only">${caption}</caption>` +
    `<thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
  t.open = open;
}

// One polite live region per page, for results that appear after an action.
let live;
export function announce(msg) {
  if (!live) {
    live = document.createElement("div");
    live.className = "sr-only"; live.setAttribute("aria-live", "polite"); live.setAttribute("role", "status");
    document.body.append(live);
  }
  live.textContent = "";
  setTimeout(() => { live.textContent = msg; }, 50);
}

// Toggle-button groups: keep aria-pressed in step with the visual state.
export function pressed(buttons, active) {
  buttons.forEach((b) => { const on = b === active; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on ? "true" : "false"); });
}
