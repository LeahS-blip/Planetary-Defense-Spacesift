// SpaceSift platform strip + mode banner: one navigation for every page, and a clear label for what kind of page it is.
// Usage: <div id="ss-nav" data-active="home|transit|spectra|asteroids|evidence"></div> + <script src="/shared/nav.js"></script>
(function () {
  const host = document.getElementById("ss-nav");
  if (!host) return;
  const active = host.dataset.active || "";
  const items = [
    ["transit", "/lab/", "01", "Hidden planets", ""],
    ["spectra", "/spectra/", "02", "Starlight", ""],
    ["asteroids", "/outputs/dashboard.html", "03", "New asteroids", '<span class="ss-live-dot" title="Live data, updated twice a day"></span>'],
  ];
  // What each page is for. Experiments test methods; the dashboard shows live data; the evidence page reports results.
  const MODES = {
    transit: ["exp", "EXPERIMENT", "<b>Test a planet search yourself.</b> Each run is one example; the conclusions come from thousands of runs.", "/evidence/#transit", "What the experiments found →"],
    spectra: ["exp", "EXPERIMENT", "<b>See how light reveals what stars and asteroids are made of</b>, or analyze your own spectrum.", "/evidence/#spectra", "How accurate is it? →"],
    asteroids: ["live", "LIVE DATA", "<b>Inspect the candidates awaiting confirmation right now.</b> Updated twice a day. A ranking of how likely each is real, not an impact forecast.", "/evidence/#asteroids", "How good is the model? →"],
    evidence: ["report", "REPORT", "<b>What the investigations found</b>, in plain words first, with full methods, errors and limits below.", "", ""],
  };
  const logo = '<svg width="20" height="20" viewBox="0 0 100 100" aria-hidden="true"><g stroke-linecap="round"><line x1="10" y1="82" x2="44" y2="66" stroke="#3d4a75" stroke-width="8"/><line x1="18" y1="66" x2="52" y2="50" stroke="#5b6b9e" stroke-width="8"/><line x1="30" y1="52" x2="58" y2="39" stroke="#8b9cc7" stroke-width="8"/><circle cx="70" cy="38" r="20" fill="#eef2fc"/></g></svg>';
  const m = MODES[active];
  const home = active === "home";
  if (!home) document.body.classList.add("ss-page");
  host.outerHTML =
    '<div class="ss-strip' + (home ? " over" : "") + '" role="navigation" aria-label="SpaceSift">' +
    '<a class="ss-brand" href="' + (home ? "#top" : "/") + '">' + logo + "<b>SPACESIFT</b><span>the science of detection</span></a>" +
    "<nav>" +
    items.map(([k, href, n, t, extra]) => '<a href="' + href + '"' + (k === active ? ' class="on" aria-current="page"' : "") + ' title="Investigation ' + n + ": " + t + '"><i>' + n + '</i><span class="t">' + t + "</span>" + extra + "</a>").join("") +
    '<span class="ss-sep"></span><a class="ev' + (active === "evidence" ? " on" : "") + '" href="/evidence/"' + (active === "evidence" ? ' aria-current="page"' : "") + ">Evidence</a>" +
    "</nav></div>" +
    (m ? '<div class="ss-mode ' + (m[0] === "exp" ? "" : m[0]) + '"><span class="ss-pill ' + m[0] + '">' + m[1] + "</span><span>" + m[2] + "</span>" + (m[3] ? '<a href="' + m[3] + '">' + m[4] + "</a>" : "") + "</div>" : "");
  if (home) {
    const strip = document.querySelector(".ss-strip.over");
    const onScroll = () => strip.classList.toggle("scrolled", scrollY > innerHeight * 0.6);
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
  }
})();
