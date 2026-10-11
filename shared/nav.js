// SpaceSift platform strip: one navigation for every page, so the three investigations read as one project.
// Usage: <div id="ss-nav" data-active="transit|spectra|asteroids|evidence"></div> + <script src="/shared/nav.js"></script>
(function () {
  const host = document.getElementById("ss-nav");
  if (!host) return;
  const active = host.dataset.active || "";
  const items = [
    ["transit", "/lab/", "01", "Hidden planets"],
    ["spectra", "/spectra/", "02", "Starlight"],
    ["asteroids", "/outputs/dashboard.html", "03", "New asteroids"],
  ];
  const logo = '<svg width="20" height="20" viewBox="0 0 100 100" aria-hidden="true"><g stroke-linecap="round"><line x1="10" y1="82" x2="44" y2="66" stroke="#3d4a75" stroke-width="8"/><line x1="18" y1="66" x2="52" y2="50" stroke="#5b6b9e" stroke-width="8"/><line x1="30" y1="52" x2="58" y2="39" stroke="#8b9cc7" stroke-width="8"/><circle cx="70" cy="38" r="20" fill="#eef2fc"/></g></svg>';
  host.outerHTML =
    '<div class="ss-strip" role="navigation" aria-label="SpaceSift">' +
    '<a class="ss-brand" href="/">' + logo + "<b>SPACESIFT</b><span>the science of detection</span></a>" +
    "<nav>" +
    items.map(([k, href, n, t]) => '<a href="' + href + '"' + (k === active ? ' class="on" aria-current="page"' : "") + ' title="Investigation ' + n + ": " + t + '"><i>' + n + '</i><span class="t">' + t + "</span></a>").join("") +
    '<a class="ev' + (active === "evidence" ? " on" : "") + '" href="/evidence/"' + (active === "evidence" ? ' aria-current="page"' : "") + ">Evidence</a>" +
    "</nav></div>";
})();
