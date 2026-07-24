#!/usr/bin/env python3
"""Generate outputs/dashboard.html — self-contained NEOCP vetting dashboard.

Embeds current data (no server needed; open the file in any browser):
  * ranked vetting list from outputs/daily_vetting.csv
  * dataset growth from data/processed/labeled.csv + backfill_labeled.csv
  * model metrics from outputs/training_results.json
Re-run after each capture/train cycle:  python3 src/make_dashboard.py
"""
import csv, json, html
from datetime import datetime, timezone
from pathlib import Path
from followup import classify_row, STATUS, STALE_DAYS

ROOT = Path(__file__).resolve().parent.parent
BRAND = "SPACESIFT"

def comet_svg(size=54):
    """The comet mark: white nucleus + three motion-streak tails, no text."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
            '<g stroke-linecap="round">'
            '<line x1="10" y1="82" x2="44" y2="66" stroke="#3d4a75" stroke-width="7"/>'
            '<line x1="18" y1="66" x2="52" y2="50" stroke="#5b6b9e" stroke-width="7"/>'
            '<line x1="30" y1="52" x2="58" y2="39" stroke="#8b9cc7" stroke-width="7"/>'
            '<circle cx="70" cy="38" r="20" fill="#eef2fc"/>'
            '</g></svg>')

def size_m(h):
    """Rough diameter in meters from absolute magnitude H (albedo 0.14)."""
    try:
        return 1329000 / (0.14 ** 0.5) * 10 ** (-float(h) / 5)
    except (TypeError, ValueError):
        return None

def fmt_size(m):
    if m is None:
        return ""
    if m >= 1000:
        return f"{m/1000:.1f} km"
    return f"{m:.0f} m"

def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))

def sky_map(objects, replay_label=""):
    """Inline SVG: equatorial sky (RA hours, Dec deg), ecliptic overlaid."""
    import math
    W, H, L, T = 860, 400, 46, 18   # canvas, left/top margins
    PW, PH = W - L - 14, H - T - 44
    def xy(ra_h, dec):
        x = L + PW * (1 - ra_h / 24.0)          # RA increases leftward (astro convention)
        y = T + PH * (1 - (dec + 90) / 180.0)
        return x, y
    parts = [f'<svg id="skymap" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="sky map">']
    parts.append(f'<rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="#0e1430" rx="8"/>')
    for hh in range(0, 25, 4):                   # RA grid
        x, _ = xy(hh, 0)
        parts.append(f'<line x1="{x:.0f}" y1="{T}" x2="{x:.0f}" y2="{T+PH}" stroke="#1d2547" stroke-width="1"/>')
        parts.append(f'<text x="{x:.0f}" y="{T+PH+16}" fill="#8b93ad" font-size="10" text-anchor="middle">{hh}h</text>')
    for dd in range(-60, 61, 30):                # Dec grid
        _, y = xy(0, dd)
        parts.append(f'<line x1="{L}" y1="{y:.0f}" x2="{L+PW}" y2="{y:.0f}" stroke="#1d2547" stroke-width="1"/>')
        parts.append(f'<text x="{L-6}" y="{y:.0f}" fill="#8b93ad" font-size="10" text-anchor="end" dominant-baseline="middle">{dd:+d}&#176;</text>')
    # ecliptic: equatorial coords of ecliptic latitude 0
    eps = math.radians(23.4393)
    pts = []
    for i in range(121):
        lam = math.radians(i * 3)
        ra = math.atan2(math.sin(lam) * math.cos(eps), math.cos(lam)) % (2 * math.pi)
        dec = math.degrees(math.asin(math.sin(eps) * math.sin(lam)))
        x, y = xy(ra * 12 / math.pi, dec)
        pts.append((x, y))
    # split polyline where RA wraps
    seg = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if abs(b[0] - a[0]) > PW / 2:
            parts.append('<polyline points="' + " ".join(f"{x:.0f},{y:.0f}" for x, y in seg)
                         + '" fill="none" stroke="#f9a8d4" stroke-width="1.2" stroke-dasharray="5 4" opacity=".55"/>')
            seg = []
        seg.append(b)
    parts.append('<polyline points="' + " ".join(f"{x:.0f},{y:.0f}" for x, y in seg)
                 + '" fill="none" stroke="#f9a8d4" stroke-width="1.2" stroke-dasharray="5 4" opacity=".55"/>')
    parts.append(f'<text x="{L+PW-8}" y="{T+14}" fill="#f9a8d4" font-size="10" text-anchor="end" opacity=".8">ecliptic</text>')
    parts.append(f'<text id="liveclock" x="{L+8}" y="{T+16}" fill="#4ade80" font-size="11" opacity=".9">LIVE</text>')
    colors = {"hi": "#4ade80", "mid": "#fbbf24", "lo": "#64748b"}
    payload = []
    for o in sorted(objects, key=lambda o: o["p"]):
        c = colors["hi" if o["p"] >= 0.8 else ("mid" if o["p"] >= 0.4 else "lo")]
        r = 4 + 3 * o["p"]
        # static measured-history line
        trail = o.get("trail") or []
        pts = [xy(th % 24, max(-89, min(89, td))) for _, th, td in trail]
        ok_trail = len(pts) >= 2 and all(abs(a[0] - b[0]) < PW / 2 for a, b in zip(pts, pts[1:]))
        if ok_trail:
            d = "M " + " L ".join(f"{px:.0f} {py:.0f}" for px, py in pts)
            parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.3" opacity=".4"/>')
        sid = f'obj_{o["desig"]}'
        payload.append({"id": sid, "desig": o["desig"], "ra": o["ra_h"], "dec": o["dec"],
                        "t": o.get("last_ms"), "vra": o.get("dra_h_day", 0.0),
                        "vdec": o.get("ddec_day", 0.0), "p": o["p"], "v": str(o["v"]),
                        "r": round(r, 1), "c": c, "label": True})
    import json as _json
    parts.append(f'<g id="livedots"></g>')
    parts.append('</svg>')
    cfg = {"L": L, "T": T, "PW": PW, "PH": PH}
    script = """
<script>
(function() {
  var objs = __OBJS__, cfg = __CFG__;
  function xy(raH, dec) {
    raH = ((raH % 24) + 24) % 24; dec = Math.max(-89, Math.min(89, dec));
    return [cfg.L + cfg.PW * (1 - raH / 24), cfg.T + cfg.PH * (1 - (dec + 90) / 180)];
  }
  var g = document.getElementById('livedots');
  var NS = 'http://www.w3.org/2000/svg';
  var nodes = {};
  objs.forEach(function(o) {
    var seg = document.createElementNS(NS, 'line');
    seg.setAttribute('stroke', o.c); seg.setAttribute('stroke-width', '1.2');
    seg.setAttribute('stroke-dasharray', '3 3'); seg.setAttribute('opacity', '.6');
    g.appendChild(seg);
    var dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('r', o.r); dot.setAttribute('fill', o.c);
    dot.setAttribute('stroke', '#0b1020'); dot.setAttribute('stroke-width', '1');
    var ti = document.createElementNS(NS, 'title'); dot.appendChild(ti);
    g.appendChild(dot);
    var txt = document.createElementNS(NS, 'text');
    txt.setAttribute('fill', '#e8ecf8'); txt.setAttribute('font-size', '10');
    txt.setAttribute('paint-order', 'stroke'); txt.setAttribute('stroke', '#0b1020');
    txt.setAttribute('stroke-width', '3'); txt.style.display = 'none';
    txt.style.pointerEvents = 'none'; txt.textContent = o.desig;
    g.appendChild(txt);
    dot.addEventListener('mouseenter', function() { txt.style.display = ''; dot.setAttribute('stroke', '#e8ecf8'); });
    dot.addEventListener('mouseleave', function() { txt.style.display = 'none'; dot.setAttribute('stroke', '#0b1020'); });
    nodes[o.id] = {seg: seg, dot: dot, ti: ti, txt: txt};
  });
  function update() {
    var now = Date.now();
    objs.forEach(function(o) {
      var n = nodes[o.id];
      var days = o.t ? (now - o.t) / 86400000 : 0;
      var stale = days > 7;
      var dDays = stale ? 7 : days;
      var ra = o.ra + o.vra * dDays, dec = o.dec + o.vdec * dDays;
      var lp = xy(o.ra, o.dec), np_ = xy(ra, dec);
      n.seg.setAttribute('x1', lp[0]); n.seg.setAttribute('y1', lp[1]);
      n.seg.setAttribute('x2', np_[0]); n.seg.setAttribute('y2', np_[1]);
      n.dot.setAttribute('cx', np_[0]); n.dot.setAttribute('cy', np_[1]);
      n.dot.setAttribute('fill-opacity', stale ? '.35' : '.9');
      n.ti.textContent = o.desig + '  P(real NEO)=' + o.p.toFixed(2) + '  V=' + o.v +
        ' | position extrapolated ' + days.toFixed(1) + ' days past last measurement' +
        (stale ? ' (STALE - capped at 7d, treat as lost-in-space uncertain)' : '');
      if (n.txt) { n.txt.setAttribute('x', np_[0] + o.r + 3); n.txt.setAttribute('y', np_[1] + 3); }
    });
    var lbl = document.getElementById('liveclock');
    if (lbl) lbl.textContent = 'LIVE ' + new Date().toISOString().slice(0, 16).replace('T', ' ') +
      ' UTC - dots extrapolated from last measurement';
  }
  update(); setInterval(update, 30000);
})();
</script>"""
    script = script.replace("__OBJS__", _json.dumps(payload)).replace("__CFG__", _json.dumps(cfg))
    return "".join(parts) + script

def main():
    vet = read_csv(ROOT / "outputs" / "daily_vetting.csv")
    lab = read_csv(ROOT / "data" / "processed" / "labeled.csv")
    bf = read_csv(ROOT / "data" / "processed" / "backfill_labeled.csv")
    outcomes = read_csv(ROOT / "data" / "capture" / "outcomes.csv")
    snaps = sorted((ROOT / "data" / "capture" / "snapshots").glob("neocp_*.json"))
    res = {}
    rj = ROOT / "outputs" / "training_results.json"
    if rj.exists():
        res = json.loads(rj.read_text())

    all_lab = lab + [b for b in bf if b["trksub"] not in {r["trksub"] for r in lab}]
    n_pos = sum(1 for r in all_lab if r.get("y") == "1")
    n_neg = len(all_lab) - n_pos
    latest_snap = snaps[-1].stem.replace("neocp_", "") if snaps else "-"

    # show only objects on the CURRENT page (latest snapshot)
    if snaps:
        cur = {r["Temp_Desig"] for r in json.loads(snaps[-1].read_text())}
        vet = [r for r in vet if r["trksub"] in cur]

    map_objs = []
    if snaps:
        def snap_ms(sp):
            d = sp.stem.replace("neocp_", "")
            return int(datetime(int(d[:4]), int(d[4:6]), int(d[6:8]),
                                int(d[9:11]), int(d[11:13]), tzinfo=timezone.utc).timestamp() * 1000)
        history = {}
        for sp in snaps:  # chronological (t_ms, ra_h, dec) per object
            ms = snap_ms(sp)
            for r in json.loads(sp.read_text()):
                try:
                    pt = (ms, float(r["R.A."]), float(r["Decl."]))
                except (KeyError, ValueError, TypeError):
                    continue
                hist = history.setdefault(r["Temp_Desig"], [])
                if not hist or hist[-1][1:] != pt[1:]:
                    hist.append(pt)
        # tracklet-derived motion fallback for single-snapshot objects
        feats = {r["trksub"]: r for r in read_csv(ROOT / "data" / "processed" / "features.csv")}
        import math
        snap_pos = {r["Temp_Desig"]: r for r in json.loads(snaps[-1].read_text())}
        pmap = {r["trksub"]: float(r.get("p_real_neo") or 0) for r in vet}
        last_ms = snap_ms(snaps[-1])
        for d, s in snap_pos.items():
            try:
                ra_h, dec = float(s["R.A."]), float(s["Decl."])
            except (KeyError, ValueError, TypeError):
                continue
            hist = history.get(d, [])
            dra_h_day = ddec_day = 0.0
            if len(hist) >= 2:
                (t0, r0, d0), (t1, r1, d1) = hist[-2], hist[-1]
                dt = max((t1 - t0) / 86400000.0, 1e-6)
                dra = r1 - r0
                if dra > 12: dra -= 24
                if dra < -12: dra += 24
                dra_h_day, ddec_day = dra / dt, (d1 - d0) / dt
            else:
                f = feats.get(d, {})
                try:
                    rate = float(f.get("sky_rate_deg_day") or 0)
                    pa = math.radians(float(f.get("motion_pa_deg") or 0))
                    ddec_day = rate * math.cos(pa)
                    dra_h_day = rate * math.sin(pa) / max(math.cos(math.radians(dec)), .05) / 15.0
                except (ValueError, TypeError):
                    pass
            map_objs.append({"desig": d, "ra_h": ra_h, "dec": dec,
                             "v": s.get("V", ""), "p": pmap.get(d, 0.0),
                             "trail": hist, "last_ms": last_ms,
                             "dra_h_day": round(dra_h_day, 5), "ddec_day": round(ddec_day, 5)})
    sky_html = sky_map(map_objs) if map_objs else ""

    test = res.get("test") or res.get("cv") or {}
    op = (test.get("at_99pct_neo_recall") or {})
    base = (res.get("baseline_on_test") or res.get("baseline_digest2_score") or {})
    op_b = (base.get("at_99pct_neo_recall") or {})

    def stat(label, value, sub=""):
        return (f'<div class="stat"><div class="v">{value}</div>'
                f'<div class="l">{label}</div><div class="s">{sub}</div></div>')

    stats = "".join([
        stat("objects on NEOCP", len(vet), f"snapshot {latest_snap} UTC"),
        stat("labeled training rows", len(all_lab), f"{n_pos} NEO / {n_neg} non-NEO"),
        stat("outcomes archived", len(outcomes), "since 2026-05-16"),
        stat("model ROC-AUC",
             f'{res["cv_auc"]:.2f}' if res.get("cv_auc") else (f'{test.get("roc_auc", 0):.2f}' if test else "—"),
             (f'±{res["cv_auc_std"]:.2f} · 30-fold CV · DART' if res.get("cv_auc")
              else res.get("model", ""))),
        stat("junk excluded @ 100% NEO recall",
             f'{100*op.get("non_neo_excluded_frac", 0):.0f}%' if op else "—",
             f'baseline {100*op_b.get("non_neo_excluded_frac", 0):.0f}% · benchmark 80%'),
    ])
    _c = (res.get("conformal") or {}).get("0.05")
    if _c:
        stats += stat("junk excluded @ 95% NEO guarantee",
                      f'{100*_c["junk_excluded_mean"]:.0f}%',
                      "conformal — statistical promise, not estimate")

    # Queue table header with hoverable "?" help on every column.
    QUEUE_COLS = [
        ("#", "Rank — candidates sorted by the model's P(real NEO), highest first."),
        ("desig", "MPC temporary designation (trksub) for this unconfirmed candidate."),
        ("P(real NEO)", "Machine-learned probability (0–1) that this is a genuine near-Earth "
                        "object rather than a false positive. Use as a ranking, not calibrated risk."),
        ("follow-up", "Observation status from the feed. Needs follow-up: single-night, no one has "
                      "tracked it yet. Being tracked: multi-night arc, seen recently. Going stale: "
                      "not observed in over 3 days — at risk of being lost."),
        ("~size", "Rough diameter from absolute magnitude H (assumes 0.14 albedo). ▲ marks "
                  "≥140 m — regional-devastation class if real."),
        ("digest2", "The feed's own digest2 score (0–100): higher = more NEO-like orbit. "
                    "Near-chance on the hard cases, which is why the model adds value."),
        ("V", "Apparent magnitude — how bright it looks now. Larger number = fainter."),
        ("H", "Absolute magnitude — intrinsic brightness, a stand-in for size."),
        ("obs", "Number of observations reported so far."),
        ("arc d", "Arc length in days: span between first and last observation. Under 1 = single night."),
        ("unseen d", "Days since the object was last observed."),
    ]
    queue_header = "<tr>" + "".join(
        f'<th>{html.escape(lbl)}'
        f'<span class="{"hint r" if i >= 6 else "hint"}" data-tip="{html.escape(tip)}">?</span></th>'
        for i, (lbl, tip) in enumerate(QUEUE_COLS)) + "</tr>"

    rows_html = []
    for i, r in enumerate(vet, 1):
        try:
            p = float(r.get("p_real_neo") or 0)
        except ValueError:
            p = 0.0
        cls = "hi" if p >= 0.8 else ("mid" if p >= 0.4 else "lo")
        bar = f'<div class="bar"><div class="fill {cls}" style="width:{p*100:.0f}%"></div></div>'
        sm = size_m(r.get("H"))
        big = sm is not None and sm >= 140
        size_cell = (f'<td>{fmt_size(sm)}'
                     + (' <span class="big" title="&#8805;140 m — regional-devastation class if real; catalog only ~40% complete at this size">&#9650;</span>' if big else '')
                     + '</td>')
        st = r.get("followup_status") or classify_row(r)
        meta = STATUS.get(st, STATUS["unknown"])
        badge = (f'<span class="fu" style="--c:{meta["color"]}" '
                 f'title="{meta["label"]} — from arc length, obs count and days-since-seen">'
                 f'{meta["label"]}</span>')
        rows_html.append(
            f'<tr{" class=bigrow" if big else ""}><td>{i}</td><td class="mono">{html.escape(r["trksub"])}</td>'
            f'<td class="pcell">{bar}<span>{p:.2f}</span></td>'
            f'<td>{badge}</td>'
            + size_cell +
            f'<td>{r.get("digest2_feed_score","")}</td><td>{r.get("V","")}</td>'
            f'<td>{r.get("H","")}</td><td>{r.get("nobs","")}</td>'
            f'<td>{r.get("arc_days","")}</td><td>{r.get("not_seen_dys","")}</td></tr>')

    sentry = {}
    sj = ROOT / "data" / "capture" / "sentry.json"
    if sj.exists():
        sentry = json.loads(sj.read_text())
    sentry_rows = []
    for s in (sentry.get("top") or [])[:8]:
        d_m = (s.get("diameter_km") or 0) * 1000
        ip = s.get("ip_cum") or 0
        iptxt = f"1 in {round(1/ip):,}" if ip > 0 else "&lt;1 in 10&#8310;"
        url = f'https://cneos.jpl.nasa.gov/sentry/details.html#?des={s.get("des","").replace(" ", "%20")}'
        sentry_rows.append(
            f'<tr><td><a class="mono" href="{url}" target="_blank">{html.escape(s.get("fullname") or s.get("des",""))}</a></td>'
            f'<td>{fmt_size(d_m) if d_m else "?"}</td><td>{iptxt}</td>'
            f'<td>{html.escape(str(s.get("range","")))}</td><td>{s.get("ps_cum","")}</td></tr>')
    sentry_html = ""
    if sentry_rows:
        sentry_html = f"""
<h2>Impact watch — NASA Sentry</h2>
<table><tr><th>object</th><th>size</th><th>impact odds</th><th>when</th><th>palermo</th></tr>
{''.join(sentry_rows)}</table>
<p class="note">The highest-ranked <em>known</em> impact risks, from
<a href="https://cneos.jpl.nasa.gov/sentry/" target="_blank" style="color:var(--acc)">JPL's Sentry monitor</a>
(as of {html.escape(str(sentry.get('fetched_utc','')))}). Palermo scale is log&#8321;&#8320; risk vs.
background: &#8722;2 &#8776; 1% of background risk; 0 would equal it — nothing known is close.
None of these merit worry today; they merit <em>tracking</em>, which is the point of this whole
pipeline: the dangerous one is the one nobody has found yet.</p>"""

    from collections import Counter
    oc = Counter(o["outcome_class"] for o in outcomes)
    oc_rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>'
                      for k, v in oc.most_common())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logo = comet_svg(54)
    brand = BRAND
    from urllib.parse import quote
    favicon = quote(comet_svg(64))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEOCP Vetting — Planetary Defense</title>
<link rel="icon" href="data:image/svg+xml,{favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root {{ --bg:#0b1020; --card:#141a30; --ink:#e8ecf8; --dim:#8b93ad;
        --hi:#4ade80; --mid:#fbbf24; --lo:#64748b; --acc:#7aa2ff; }}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--bg); color:var(--ink);
       font:14px/1.55 "Space Mono",ui-monospace,Consolas,monospace; padding:28px; }}
svg text {{ font-family:"Space Mono",ui-monospace,Consolas,monospace; }}
h1 {{ font-size:32px; font-weight:700; letter-spacing:.18em; }}
.topbar {{ display:flex; justify-content:flex-start; align-items:center; gap:18px; }}
.topbar svg {{ flex:0 0 auto; }}
.subtitle {{ color:var(--dim); font-size:13px; margin-top:2px; }}
.brand {{ text-align:center; flex:0 0 auto; }}
.brand svg {{ display:block; margin:0 auto; }}
.brandname {{ font-size:12px; font-weight:700; letter-spacing:.38em; color:var(--ink);
             margin-top:2px; padding-left:.38em; }}
h1 small {{ color:var(--dim); font-weight:400; font-size:13px; margin-left:10px; }}
h2 {{ font-size:15px; color:var(--acc); margin:28px 0 10px; text-transform:uppercase;
     letter-spacing:.12em; }}
.intro {{ background:var(--card); border-left:3px solid var(--acc); border-radius:10px;
         padding:14px 18px; margin-top:16px; max-width:900px; }}
.intro p {{ color:var(--dim); font-size:13.5px; }}
.intro p + p {{ margin-top:8px; }}
.intro strong {{ color:var(--ink); }} .intro em {{ color:var(--acc); font-style:normal; }}
.intro a {{ color:var(--acc); text-decoration:underline; text-underline-offset:3px; }}
.statgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
            gap:12px; margin-top:18px; }}
.stat {{ background:var(--card); border-radius:12px; padding:16px 18px; }}
.stat .v {{ font-size:26px; font-weight:650; }}
.stat .l {{ color:var(--dim); font-size:12.5px; margin-top:2px; }}
.stat .s {{ color:var(--dim); font-size:11px; margin-top:6px; opacity:.8; }}
table {{ width:100%; border-collapse:collapse; background:var(--card);
        border-radius:12px; overflow:hidden; font-size:13.5px; }}
th {{ text-align:left; color:var(--dim); font-weight:500; font-size:11.5px;
     text-transform:uppercase; letter-spacing:.08em; padding:10px 12px;
     border-bottom:1px solid #232b4a; }}
td {{ padding:8px 12px; border-bottom:1px solid #1b2340; }}
tr:last-child td {{ border-bottom:none; }}
.mono {{ color:var(--acc); }}
.pcell {{ display:flex; align-items:center; gap:10px; min-width:170px; }}
.bar {{ flex:1; height:7px; background:#20294a; border-radius:4px; overflow:hidden; }}
.fill {{ height:100%; border-radius:4px; }}
.fill.hi {{ background:var(--hi); }} .fill.mid {{ background:var(--mid); }}
.fill.lo {{ background:var(--lo); }}
.skycard {{ background:var(--card); border-radius:12px; padding:14px; }}
.skycard svg {{ width:100%; height:auto; display:block; }}
.fu {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px;
       font-weight:600; white-space:nowrap; color:var(--c);
       border:1px solid var(--c); background:rgba(255,255,255,.04); }}
.big {{ color:#f87171; font-size:10px; }}
.bigrow td {{ background:rgba(248,113,113,.05); }}
th {{ position:relative; }}
.hint {{ display:inline-block; margin-left:4px; width:13px; height:13px; border-radius:50%;
         border:1px solid var(--dim); color:var(--dim); font-size:9px; line-height:12px;
         text-align:center; font-weight:700; cursor:help; vertical-align:middle; }}
.hint:hover {{ color:var(--acc); border-color:var(--acc); }}
.hint:hover::after {{ content:attr(data-tip); position:absolute; left:0; top:130%; z-index:20;
         width:210px; padding:8px 10px; background:#0e1430; color:#cdd5ee;
         border:1px solid #2a3358; border-radius:7px; font-size:11px; font-weight:400;
         line-height:1.45; text-align:left; white-space:normal; letter-spacing:normal;
         text-transform:none; font-family:ui-sans-serif,system-ui,"Segoe UI",Arial,sans-serif;
         box-shadow:0 6px 18px rgba(0,0,0,.45); }}
/* right-half columns: grow the tooltip leftward so the table's overflow:hidden can't clip it */
.hint.r:hover::after {{ left:auto; right:0; }}
.cols {{ display:grid; grid-template-columns:2fr 1fr; gap:20px; align-items:start; }}
.note {{ color:var(--dim); font-size:12px; margin-top:10px; }}
footer {{ color:var(--dim); font-size:11.5px; margin-top:30px; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
</style></head><body>
<div class="topbar">
{logo}
<div>
<h1>{brand}</h1>
<div class="subtitle">NEOCP vetting — real-NEO probability for objects awaiting confirmation · generated {now}</div>
</div>
</div>
<div class="intro">
<p><strong>What is this?</strong> Every night, survey telescopes flag possible new near-Earth
asteroids. Candidates land on the Minor Planet Center's <a href="https://www.minorplanetcenter.net/iau/NEO/toconfirm_tabular.html" target="_blank"><em>NEO (Near-Earth Object) Confirmation Page</em></a> — the source of every object listed below — where
volunteer astronomers must re-observe them within days — before their orbits become too
uncertain and they are lost. But only about half of the candidates turn out to be real NEOs;
the rest are camera artifacts, satellites, stars, or ordinary main-belt asteroids that waste
scarce telescope time.</p>
<p>This dashboard ranks the current candidates by a machine-learned <strong>probability that
each one is a real NEO</strong>, so follow-up effort goes to the objects that matter. Green
bars: observe first. Gray bars: probably junk. The model learns from every past candidate
whose fate is now known, and it improves as the archive below it grows.</p>
</div>
<div class="statgrid">{stats}</div>
<h2>Where they are right now</h2>
<div class="skycard">{sky_html}
<p class="note">Each dot is a candidate awaiting confirmation, plotted at its current sky position
(right ascension &amp; declination — celestial longitude/latitude). Bigger + greener = more likely a
real NEO; labels mark the top targets. The dashed line is the <span style="color:#f9a8d4">ecliptic</span>,
the plane of the solar system — ordinary main-belt asteroids hug it, so candidates far off it are
more interesting. Hover any dot for details. The solid faint line is each object's
<strong>measured history</strong> across our snapshots; the dot sits at its <strong>estimated position
right now</strong>, extrapolated live (updated every 30s) from its last measured motion — the dashed
segment covers the stretch since its last actual measurement. The estimate assumes straight-line motion,
so the longer since measurement, the less certain it is; dots fade when >7 days stale. That growing
uncertainty is precisely why unconfirmed objects get lost. Hover a dot to see its designation
and details.</p>
</div>
<div class="cols">
<div>
<h2>Current confirmation queue — ranked</h2>
<table>
{queue_header}
{''.join(rows_html)}
</table>
<p class="note">Model: {html.escape(str(res.get('model','feed-score fallback')))} trained on
{res.get('n','?')} labeled objects. Probabilities are early-stage — treat as ranking,
not calibrated risk. Green ≥0.8, amber ≥0.4.<br>
Follow-up status: <b style="color:{STATUS['needs_followup']['color']}">Needs follow-up</b>
= fresh single-night candidate, no one's tracked it yet ·
<b style="color:{STATUS['tracked']['color']}">Being tracked</b> = multi-night arc, seen recently ·
<b style="color:{STATUS['going_stale']['color']}">Going stale</b> = not seen in
&gt;{int(STALE_DAYS)} days, at risk of loss.</p>
</div>
<div>
{sentry_html}
<h2>Outcome archive</h2>
<table><tr><th>class</th><th>n</th></tr>{oc_rows}</table>
<p class="note">"neo" = confirmed via discovery MPEC. Negatives accrue from live capture;
designated_other are mostly main-belt identifications.</p>
</div>
</div>
<footer>planetary-defense-neocp-vetting · data: MPC NEOCP / Previous-NEOCP / MPECs ·
regenerate with <span class="mono">python3 src/make_dashboard.py</span></footer>
</body></html>"""
    out = ROOT / "outputs" / "dashboard.html"
    out.write_text(doc, encoding="utf-8")
    print(f"dashboard: {len(vet)} queue rows, {len(all_lab)} labeled -> {out}")

if __name__ == "__main__":
    main()
