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
    parts.append(f'<rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="rgba(11,16,32,.65)" stroke="rgba(140,160,220,.16)" rx="10"/>')
    for hh in range(0, 25, 4):                   # RA grid
        x, _ = xy(hh, 0)
        parts.append(f'<line x1="{x:.0f}" y1="{T}" x2="{x:.0f}" y2="{T+PH}" stroke="rgba(140,160,220,.10)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.0f}" y="{T+PH+16}" fill="#7f88a6" font-size="10" text-anchor="middle">{hh}h</text>')
    for dd in range(-60, 61, 30):                # Dec grid
        _, y = xy(0, dd)
        parts.append(f'<line x1="{L}" y1="{y:.0f}" x2="{L+PW}" y2="{y:.0f}" stroke="rgba(140,160,220,.10)" stroke-width="1"/>')
        parts.append(f'<text x="{L-6}" y="{y:.0f}" fill="#7f88a6" font-size="10" text-anchor="end" dominant-baseline="middle">{dd:+d}&#176;</text>')
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
    dot.setAttribute('stroke', '#04060d'); dot.setAttribute('stroke-width', '1');
    var ti = document.createElementNS(NS, 'title'); dot.appendChild(ti);
    g.appendChild(dot);
    var txt = document.createElementNS(NS, 'text');
    txt.setAttribute('fill', '#eef2fc'); txt.setAttribute('font-size', '10');
    txt.setAttribute('paint-order', 'stroke'); txt.setAttribute('stroke', '#04060d');
    txt.setAttribute('stroke-width', '3'); txt.style.display = 'none';
    txt.style.pointerEvents = 'none'; txt.textContent = o.desig;
    g.appendChild(txt);
    dot.addEventListener('mouseenter', function() { txt.style.display = ''; dot.setAttribute('stroke', '#eef2fc'); });
    dot.addEventListener('mouseleave', function() { txt.style.display = 'none'; dot.setAttribute('stroke', '#04060d'); });
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
    if (lbl) lbl.textContent = 'Positions estimated for ' + new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) + ' your time';
  }
  update(); document.addEventListener('DOMContentLoaded', update); setInterval(update, 30000);
})();
</script>"""
    script = script.replace("__OBJS__", _json.dumps(payload)).replace("__CFG__", _json.dumps(cfg))
    return "".join(parts) + script

def main():
    vet = read_csv(ROOT / "outputs" / "daily_vetting.csv")
    vet_all = list(vet)   # full list before the current-snapshot filter, for the tracked archive
    lab = read_csv(ROOT / "data" / "processed" / "labeled.csv")
    bf = read_csv(ROOT / "data" / "processed" / "backfill_labeled.csv")
    outcomes = read_csv(ROOT / "data" / "capture" / "outcomes.csv")
    snaps = sorted((ROOT / "data" / "capture" / "snapshots").glob("neocp_*.json"))
    res = {}
    rj = ROOT / "outputs" / "training_results.json"
    if rj.exists():
        res = json.loads(rj.read_text())
    # Results written before the leakage fix carry no "inputs" field: keep ranking, but withdraw their scores.
    withdrawn = bool(res) and not res.get("inputs")
    if withdrawn:
        res = {k: v for k, v in res.items() if k in ("n", "model", "trained_utc", "features")}

    all_lab = lab + [b for b in bf if b["trksub"] not in {r["trksub"] for r in lab}]
    n_pos = sum(1 for r in all_lab if r.get("y") == "1")
    n_neg = len(all_lab) - n_pos
    latest_snap = snaps[-1].stem.replace("neocp_", "") if snaps else "-"
    def nice(stamp):  # 20261011T2148 -> 11 Oct 2026, 21:48 UTC
        try:
            return datetime.strptime(stamp[:13], "%Y%m%dT%H%M").strftime("%d %b %Y, %H:%M UTC").lstrip("0")
        except ValueError:
            return stamp
    snap_iso = (datetime.strptime(latest_snap[:13], "%Y%m%dT%H%M").strftime("%Y-%m-%dT%H:%MZ")
                if snaps and latest_snap[:8].isdigit() else "")

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

    ho = res.get("holdout") or {}
    auc_m = (ho.get("roc_auc") or {}).get("model", test.get("roc_auc"))
    auc_b = (ho.get("roc_auc") or {}).get("baseline", base.get("roc_auc"))
    trained = res.get("trained_utc", "")[:10] or "2026-08-05"
    try:
        trained = datetime.strptime(trained, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    except ValueError:
        pass
    sp_ = ho.get("split") or {}
    # Every page quotes the same evaluation: trained on earlier candidates, tested once on later ones.
    split_txt = (f"tested on {sp_['test_n']} candidates posted after {sp_['test_dates'][0][:10]}, "
                 f"trained on {sp_['train_n']} earlier ones" if sp_ else "tested on the newest 25% of candidates")
    stats = "".join([
        stat("candidates on the page now", len(vet), f"captured {nice(latest_snap)} · rescored twice a day"),
        stat("ROC-AUC on later candidates",
             f"{auc_m:.2f}" if auc_m is not None else "—",
             (f"digest2 alone {auc_b:.2f} · " if auc_b is not None else "") + split_txt),
        stat("junk excluded at ≥99% NEO recall",
             f'{100*op.get("non_neo_excluded_frac", 0):.0f}%' if op else "—",
             f'same test · digest2 alone {100*op_b.get("non_neo_excluded_frac", 0):.0f}% · benchmark 80%'),
        stat("model trained on", res.get("n", "?"),
             f"labeled candidates · last retrained {trained} · retrained every Monday"),
        stat("outcomes archived", len(outcomes), "since 2026-05-16"),
    ])

    # Honest evaluation block: confusion matrices on candidates held out by date.
    def _pct(x):
        return "—" if x is None else f"{100 * x:.0f}%"
    if withdrawn:
        eval_html = """<div class="sk">EVALUATION · ON CANDIDATES IT NEVER SAW</div><h2 id="eval">Is the model any good?</h2>
<div class="evalcard"><p><b>Being re-evaluated.</b> The previous evaluation was withdrawn: some inputs described how each candidate changed
on the page until it left, which leaks the outcome. The model is being retrained on first-posting information only, and the corrected
evaluation appears here when that finishes. Until then, treat the ranking with extra caution.</p></div>"""
    elif ho.get("operating_points"):
        sp = ho["split"]
        op95 = ho["operating_points"]["95pct_recall"]
        ci = ho.get("auc_bootstrap_95ci") or {}
        cal = ho.get("calibration") or {}
        def row(name, c):
            return (f"<tr><td>{name}</td><td>{c['tp']}</td><td>{c['fn']}</td><td>{c['fp']}</td><td>{c['tn']}</td>"
                    f"<td>{_pct(c['neo_recall'])}</td><td>{_pct(c['junk_excluded'])}</td></tr>")
        rel = "".join(f"<tr><td>{b['bin'][0]:.1f}–{b['bin'][1]:.1f}</td><td>{b['n']}</td><td>{b['mean_predicted']:.2f}</td>"
                      f"<td>{b['fraction_neo']:.2f}</td></tr>" for b in cal.get("reliability_model", []))
        eval_html = f"""<div class="sk">EVALUATION · ON CANDIDATES IT NEVER SAW</div><h2 id="eval">Is the model any good?</h2>
<p class="lede">{"Yes, at ranking: " if (auc_m or 0) > (auc_b or 0) else "Not yet: "}on candidates it had never seen, posted after it was trained, it ranked real near-Earth objects
above junk better than the digest2 score alone (ROC-AUC {ho['roc_auc']['model']:.2f} vs {ho['roc_auc']['baseline']:.2f}).
At a cut-off that keeps 95% of real ones, it threw out {_pct(op95['model']['junk_excluded'])} of the junk; digest2 alone threw out {_pct(op95['baseline']['junk_excluded'])}.</p>
<details class="deep"><summary>Full held-out evaluation: confusion matrices, confidence intervals, calibration</summary>
<div class="evalcard">
<p>Trained on candidates discovered {sp['train_dates'][0]}–{sp['train_dates'][1]} ({sp['train_n']}), tested once on the
<strong>{sp['test_n']} later candidates</strong> ({sp['test_dates'][0]}–{sp['test_dates'][1]}, {sp['test_neo']} real NEOs).
The cut-off score was chosen on the training period to keep 95% of real NEOs, then applied unchanged.</p>
<table><tr><th></th><th>real NEOs kept</th><th>real NEOs missed</th><th>junk kept</th><th>junk excluded</th><th>NEO recall</th><th>junk excluded</th></tr>
{row("Model", op95["model"])}{row("digest2 score alone", op95["baseline"])}</table>
<p class="note">ROC-AUC {ho['roc_auc']['model']:.2f} (95% CI {ci.get('model', [0, 0])[0]:.2f}–{ci.get('model', [0, 0])[1]:.2f}) vs
{ho['roc_auc']['baseline']:.2f} for digest2 alone (difference CI {ci.get('difference', [0, 0])[0]:+.2f} to {ci.get('difference', [0, 0])[1]:+.2f}).
Calibration: Brier score {cal.get('brier_model', 0):.3f} (lower is better; always guessing the base rate scores
{cal.get('brier_always_base_rate', 0):.3f}).</p>
<details><summary>Does “0.8” mean 80%? (reliability table)</summary><table><tr><th>predicted</th><th>n</th><th>mean predicted</th><th>actually NEO</th></tr>{rel}</table></details>
<p class="note">Inputs are only what is known when a candidate is first posted. An earlier version also used how each candidate
evolved on the page until it left, which leaks the outcome and inflated the score; see the
<a href="/evidence/#asteroids" style="color:var(--acc)">evidence page</a>.</p>
</div></details>"""
    else:
        eval_html = f"""<div class="sk">EVALUATION · ON CANDIDATES IT NEVER SAW</div><h2 id="eval">Is the model any good?</h2>
<div class="evalcard"><p>On the newest 25% of candidates, held out by discovery date (n={test.get('n', '?')}), the model's ROC-AUC is
<strong>{test.get('roc_auc', 0):.2f}</strong> versus <strong>{base.get('roc_auc', 0):.2f}</strong> for the digest2 score alone, so it ranks
candidates better. But at the strict setting that keeps ≥99% of real NEOs it excludes only {_pct(op.get('non_neo_excluded_frac'))} of junk,
versus {_pct(op_b.get('non_neo_excluded_frac'))} for digest2 alone, far from the 80% benchmark. Confusion matrices and a calibration check
appear here after the next Monday retrain.</p></div>"""

    # Limitations shown where people act on the ranking, not buried in documentation.
    lim = [f"It ranks better than the digest2 score alone (ROC-AUC {auc_m:.2f} vs {auc_b:.2f}, {split_txt}), "
           f"but at the strict setting that keeps ≥99% of real NEOs it drops only {_pct(op.get('non_neo_excluded_frac'))} of junk "
           f"(digest2 alone: {_pct(op_b.get('non_neo_excluded_frac'))})." if auc_m is not None and auc_b is not None else
           "Its advantage over the digest2 score alone is not yet established.",
           "Probabilities are <b>not calibrated</b>: use them to order follow-up, not as odds.",
           "Labels are noisy: some candidates marked “not confirmed” were real NEOs that were simply lost.",
           f"Trained on {res.get('n', '?')} candidates (last retrained {trained}; retrained every Monday); unusual candidates unlike past ones may be scored poorly.",
           "<b>Not an impact forecast:</b> a high score means “probably a real near-Earth object”, nothing about hitting Earth."]
    limits_html = ('<div class="ss-limits" style="margin:0 0 14px"><h4>BEFORE YOU ACT ON THIS RANKING</h4><ul>'
                   + "".join(f"<li>{x}</li>" for x in lim)
                   + '</ul><a href="/evidence/#asteroids" style="display:inline-block;margin-top:8px;color:var(--acc);font-weight:700;text-decoration:none">Full evaluation →</a></div>')

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
<div class="sk">CONTEXT · KNOWN IMPACT RISKS</div><h2>Impact watch: NASA Sentry</h2>
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

    # Tracked-objects archive: every candidate ever captured while "being tracked",
    # kept so they aren't lost when they roll off the live NEOCP page (and out of the
    # current-snapshot queue above). Built from vet_all — the unfiltered vetting CSV.
    TRACKED_COLS = [
        ("#", "Position in this archive, highest model probability first."),
        ("desig", "MPC temporary designation (trksub)."),
        ("P(real NEO)", "Model probability (0-1) this is a genuine near-Earth object. "
                        "Ranking, not calibrated risk."),
        ("~size", "Rough diameter from absolute magnitude H (assumes 0.14 albedo). ▲ marks ≥140 m."),
        ("digest2", "The feed's own digest2 score (0-100): higher = more NEO-like orbit."),
        ("V", "Apparent magnitude — how bright it looks now. Larger = fainter."),
        ("H", "Absolute magnitude — intrinsic brightness, a stand-in for size."),
        ("obs", "Number of observations reported so far."),
        ("arc d", "Arc length in days between first and last observation."),
        ("unseen d", "Days since the object was last observed — how close it is to being lost."),
    ]
    def _is_tracked(r):
        return (r.get("followup_status") or classify_row(r)) == "tracked"
    tracked = sorted((r for r in vet_all if _is_tracked(r)),
                     key=lambda r: float(r.get("p_real_neo") or 0), reverse=True)
    tracked_html = ""
    if tracked:
        thead = "<tr>" + "".join(
            f'<th>{html.escape(lbl)}'
            f'<span class="{"hint r" if i >= 5 else "hint"}" data-tip="{html.escape(tip)}">?</span></th>'
            for i, (lbl, tip) in enumerate(TRACKED_COLS)) + "</tr>"
        trows = []
        for i, r in enumerate(tracked, 1):
            try:
                p = float(r.get("p_real_neo") or 0)
            except ValueError:
                p = 0.0
            cls = "hi" if p >= 0.8 else ("mid" if p >= 0.4 else "lo")
            bar = f'<div class="bar"><div class="fill {cls}" style="width:{p*100:.0f}%"></div></div>'
            sm = size_m(r.get("H"))
            big = sm is not None and sm >= 140
            size_cell = (f'<td>{fmt_size(sm)}'
                         + (' <span class="big" title="&#8805;140 m — regional-devastation class if real">&#9650;</span>' if big else '')
                         + '</td>')
            trows.append(
                f'<tr{" class=bigrow" if big else ""}><td>{i}</td>'
                f'<td class="mono">{html.escape(r["trksub"])}</td>'
                f'<td class="pcell">{bar}<span>{p:.2f}</span></td>'
                + size_cell +
                f'<td>{html.escape(str(r.get("digest2_feed_score","")))}</td>'
                f'<td>{html.escape(str(r.get("V","")))}</td><td>{html.escape(str(r.get("H","")))}</td>'
                f'<td>{html.escape(str(r.get("nobs","")))}</td><td>{html.escape(str(r.get("arc_days","")))}</td>'
                f'<td>{html.escape(str(r.get("not_seen_dys","")))}</td></tr>')
        tracked_html = (
            f'<details class="deep"><summary>Tracked objects archive ({len(tracked)})</summary>\n'
            '<div class="searchwrap"><input id="trackedSearch" class="tsearch" type="search" '
            'placeholder="Search designation…" autocomplete="off" '
            'aria-label="Search the tracked archive by designation">'
            '<span id="trackedSearch-count" class="scount"></span></div>\n'
            '<div class="tablewrap"><table id="trackedTable">\n'
            + thead + '\n' + "".join(trows) + '\n</table></div>\n'
            '<p class="note">Every candidate our capture has ever seen reach '
            f'<b style="color:{STATUS["tracked"]["color"]}">being tracked</b> (a multi-night arc with recent '
            'follow-up), kept here so they aren&#x27;t lost when they roll off the live NEO Confirmation Page '
            f'and out of the queue above. {len(tracked)} objects, ranked by model P(real NEO). Sourced from '
            '<span class="mono">outputs/daily_vetting.csv</span>; unseen-d is days since last observation.</p></details>')

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    def _p(r):
        try:
            return float(r.get("p_real_neo") or 0)
        except ValueError:
            return 0.0
    at_risk = sum(1 for r in vet if (r.get("followup_status") or classify_row(r)) in ("needs_followup", "going_stale"))
    top_txt = ", ".join(f'<span class="mono">{html.escape(r["trksub"])}</span> ({_p(r):.2f})' for r in vet[:3]) or "none right now"
    start_html = f"""<div class="ss-start-head"><h2>START HERE</h2><span>The live queue in three questions.</span></div>
<div class="ss-starts">
<a class="ss-start primary" href="#queue"><span class="n">01 · RIGHT NOW</span><b>Which candidates are most likely real?</b><span class="d">Top of the ranking: {top_txt}.</span><span class="go">See the ranked queue →</span></a>
<a class="ss-start" href="#map"><span class="n">02 · WHERE</span><b>Where are they in the sky?</b><span class="d">{len(vet)} candidates on the confirmation page; {at_risk} still need follow-up observations.</span><span class="go">Open the sky map →</span></a>
<a class="ss-start" href="#eval"><span class="n">03 · HOW SURE</span><b>Can this ranking be trusted?</b><span class="d">Tested on candidates it had never seen, with its errors counted and its limits stated.</span><span class="go">Read the evaluation →</span></a>
</div>"""
    from urllib.parse import quote
    favicon = quote(comet_svg(64))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="ss-updated" content="{snap_iso}">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asteroid Watch · SpaceSift</title>
<link rel="icon" href="data:image/svg+xml,{favicon}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syncopate:wght@700&family=Comfortaa:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/shared/nav.css">
<style>
:root {{ --bg:#04060d; --card:rgba(16,22,42,.72); --ink:#eef2fc; --dim:#7f88a6; --body:#c9d0e4; --line:rgba(140,160,220,.16);
        --hi:#4ade80; --mid:#fbbf24; --lo:#64748b; --acc:#7aa2ff; }}
* {{ box-sizing:border-box; margin:0; }}
html {{ background:var(--bg); }}
body {{ color:var(--ink); font:14px/1.55 "Comfortaa",ui-sans-serif,system-ui,sans-serif; }}
svg text {{ font-family:"Comfortaa",ui-sans-serif,system-ui,sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 clamp(16px,3vw,28px); }}
main.wrap {{ padding-bottom:30px; }}
h2 {{ font:700 clamp(19px,2vw,24px)/1.25 "Comfortaa",ui-sans-serif,sans-serif; color:var(--ink); margin:46px 0 14px; scroll-margin-top:64px; }}
.lede {{ color:var(--body); font-size:15.5px; line-height:1.7; max-width:860px; margin:0 0 12px; }}
.plainbox {{ margin-top:18px; }}
.plainbox p {{ font-size:14.5px; }}
.statgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr)); gap:12px; margin-top:16px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }}
.stat .v {{ font:700 28px "Syncopate",sans-serif; letter-spacing:.02em; }}
.stat .l {{ color:var(--body); font-size:13px; margin-top:4px; }}
.stat .s {{ color:var(--dim); font-size:11.5px; margin-top:6px; line-height:1.5; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:14px; overflow:hidden; font-size:13.5px; }}
.tablewrap {{ overflow-x:auto; border-radius:14px; }}
th {{ text-align:left; color:var(--dim); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.08em; padding:10px 12px; border-bottom:1px solid var(--line); }}
td {{ padding:8px 12px; border-bottom:1px solid rgba(140,160,220,.09); }}
tr:last-child td {{ border-bottom:none; }}
.mono {{ color:var(--acc); }}
.pcell {{ display:flex; align-items:center; gap:10px; min-width:170px; }}
.bar {{ flex:1; height:7px; background:rgba(140,160,220,.16); border-radius:4px; overflow:hidden; }}
.fill {{ height:100%; border-radius:4px; }}
.fill.hi {{ background:var(--hi); }} .fill.mid {{ background:var(--mid); }}
.fill.lo {{ background:var(--lo); }}
.skycard {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px; }}
.skycard svg {{ width:100%; height:auto; display:block; }}
.fu {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; white-space:nowrap; color:var(--c); border:1px solid var(--c); background:rgba(255,255,255,.04); }}
.big {{ color:#f87171; font-size:10px; }}
.bigrow td {{ background:rgba(248,113,113,.05); }}
th {{ position:relative; }}
.hint {{ display:inline-block; margin-left:4px; width:13px; height:13px; border-radius:50%; border:1px solid var(--dim); color:var(--dim); font-size:9px; line-height:12px;
         text-align:center; font-weight:700; cursor:help; vertical-align:middle; }}
.hint:hover {{ color:var(--acc); border-color:var(--acc); }}
.hint:hover::after {{ content:attr(data-tip); position:absolute; left:0; top:130%; z-index:20; width:210px; padding:8px 10px; background:#0e1430; color:#cdd5ee;
         border:1px solid #2a3358; border-radius:7px; font-size:11px; font-weight:400; line-height:1.45; text-align:left; white-space:normal; letter-spacing:normal;
         text-transform:none; font-family:"Comfortaa",ui-sans-serif,system-ui,sans-serif; box-shadow:0 6px 18px rgba(0,0,0,.45); }}
/* right-half columns: grow the tooltip leftward so the table's overflow:hidden can't clip it */
.hint.r:hover::after {{ left:auto; right:0; }}
.cols {{ display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:24px; align-items:start; }}
.cols > div {{ min-width:0; overflow-x:auto; }}
.note {{ color:var(--dim); font-size:12px; margin-top:10px; line-height:1.6; }}
footer {{ color:var(--dim); font-size:11.5px; padding-top:20px; padding-bottom:30px; border-top:1px solid var(--line); }}
.searchwrap {{ display:flex; align-items:center; gap:10px; margin:0 0 10px; }}
.tsearch {{ background:rgba(11,16,32,.8); border:1px solid var(--line); color:var(--ink); border-radius:999px; padding:8px 14px; font:13px "Comfortaa",ui-sans-serif,sans-serif; width:240px; max-width:60vw; }}
.tsearch:focus {{ outline:none; border-color:var(--acc); }}
.tsearch::placeholder {{ color:var(--dim); }}
.scount {{ color:var(--dim); font-size:12px; white-space:nowrap; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:minmax(0,1fr); }} .cols table {{ display:block; overflow-x:auto; max-width:100%; }} }}
.evalcard {{ border:1px solid var(--line); background:var(--card); border-radius:14px; padding:14px 16px; margin-top:10px; }}
.evalcard p {{ line-height:1.6; }} .evalcard p + table, .evalcard table + p {{ margin-top:12px; }}
.evalcard td, .evalcard th {{ text-align:center; }}
.evalcard table {{ display:block; overflow-x:auto; }}
details.deep {{ margin-top:14px; border:1px solid var(--line); border-radius:14px; background:rgba(11,16,32,.55); }}
details.deep > summary {{ cursor:pointer; padding:14px 16px; font:600 14px "Comfortaa",sans-serif; color:var(--body); list-style:none; display:flex; justify-content:space-between; gap:12px; }}
details.deep > summary::-webkit-details-marker {{ display:none; }}
details.deep > summary::after {{ content:"+"; color:var(--dim); font-size:18px; line-height:1; }}
details.deep[open] > summary::after {{ content:"−"; }}
details.deep > :not(summary) {{ margin-left:16px; margin-right:16px; }}
details.deep > :last-child {{ margin-bottom:16px; }}
details.deep .evalcard {{ margin-top:0; }}
.sk {{ font:700 10px "Syncopate",sans-serif; letter-spacing:.3em; color:var(--acc); margin:52px 0 0; }}
.sk + h2 {{ margin-top:10px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin:10px 2px 0; color:var(--body); font-size:12.5px; }}
.legend span {{ display:inline-flex; align-items:center; gap:7px; }}
.legend i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.legend i.dash {{ width:18px; height:0; border-radius:0; border-top:1.5px dashed var(--body); }}
.legend i.dash.pink {{ border-top-color:#f9a8d4; }}
.legend .clock {{ margin-left:auto; color:var(--dim); }}
footer a {{ color:var(--body); }}
</style>
<script>window.va = window.va || function () {{ (window.vaq = window.vaq || []).push(arguments); }};</script>
<script defer src="/_vercel/insights/script.js"></script>
</head><body>
<div id="ss-nav" data-active="asteroids"></div>
<header class="ss-head"><div class="ss-kicker">INVESTIGATION 03 · NEW ASTEROIDS</div>
<div class="ss-head-row"><h1>Asteroid Watch</h1><nav class="ss-tabs"><a href="#queue">Queue</a><a href="#map">Sky map</a><a href="#eval">Evaluation</a><a href="/evidence/#asteroids">Evidence</a></nav></div>
<p class="ss-q">Which of the new candidates on the Minor Planet Center's confirmation page are <b>real near-Earth objects</b>? A ranking, updated twice a day. Latest capture: {nice(latest_snap)}.</p></header>
<main class="wrap">
{start_html}
<div class="ss-plain plainbox"><h4>IN PLAIN WORDS</h4>
<p>Every night, survey telescopes flag possible new near-Earth asteroids and post them to the
<a href="https://www.minorplanetcenter.net/iau/NEO/toconfirm_tabular.html" target="_blank" style="color:var(--acc)">NEO Confirmation Page</a>.
Astronomers must observe them again within days, or their orbits become too uncertain and they are lost. Only about half turn out to be real;
the rest are camera artifacts, satellites, stars or ordinary main-belt asteroids.</p>
<p>The ranking puts the likeliest real ones first, so scarce telescope time goes where it matters. <b>P(real NEO) is the chance a candidate
is a genuine near-Earth object, not the chance of hitting Earth.</b> No candidate here has an impact assessment; known impact risks are
in NASA's Sentry table further down.</p></div>
<div class="statgrid">{stats}</div>
<div class="sk">LIVE · THE RANKING</div><h2 id="queue">The confirmation queue, ranked</h2>
{limits_html}
<div class="searchwrap"><input id="queueSearch" class="tsearch" type="search" placeholder="Search designation…" autocomplete="off" aria-label="Search the confirmation queue by designation"><span id="queueSearch-count" class="scount"></span></div>
<div class="tablewrap"><table id="queueTable">
{queue_header}
{''.join(rows_html)}
</table></div>
<p class="note">Model: {html.escape(str(res.get('model','feed-score fallback')))} trained on
{res.get('n','?')} labeled objects. Use the scores to order follow-up, not as calibrated odds or impact risk. Green ≥0.8, amber ≥0.4.<br>
Follow-up status: <b style="color:{STATUS['needs_followup']['color']}">Needs follow-up</b>
= fresh single-night candidate, no one's tracked it yet ·
<b style="color:{STATUS['tracked']['color']}">Being tracked</b> = multi-night arc, seen recently ·
<b style="color:{STATUS['going_stale']['color']}">Going stale</b> = not seen in
&gt;{int(STALE_DAYS)} days, at risk of loss.</p>
<div class="sk">LIVE · THE SKY</div><h2 id="map">Where they are right now</h2>
<div class="skycard">{sky_html}
<div class="legend"><span><i style="background:var(--hi)"></i>likely real (≥0.8)</span><span><i style="background:var(--mid)"></i>uncertain (0.4–0.8)</span><span><i style="background:var(--lo)"></i>probably not (&lt;0.4)</span><span><i class="dash"></i>estimated since last seen</span><span><i class="dash pink"></i>ecliptic</span><span class="clock" id="liveclock"></span></div>
<p class="note">Each dot is a candidate at its estimated position right now (right ascension and declination, the sky's longitude and latitude).
Bigger and greener means more likely real; labels mark the top targets. The dashed pink line is the <span style="color:#f9a8d4">ecliptic</span>,
where ordinary main-belt asteroids cluster, so candidates far from it are more interesting. The faint solid line is each object's measured
path across our snapshots; the dashed stretch is extrapolated from its last measured motion, assuming a straight line, so it grows less
certain with time (dots fade after 7 days). That growing uncertainty is exactly how unconfirmed objects get lost. Hover a dot for details.</p>
</div>
{eval_html}
<div class="cols">
<div>{sentry_html}</div>
<div>
<div class="sk">CONTEXT · WHAT HAPPENED TO PAST CANDIDATES</div><h2>Outcome archive</h2>
<table><tr><th>class</th><th>n</th></tr>{oc_rows}</table>
<p class="note">"neo" = confirmed via discovery MPEC. Negatives accrue from live capture;
designated_other are mostly main-belt identifications.</p>
</div>
</div>
{tracked_html}
</main>
<footer class="wrap">SpaceSift · the science of detection · open code and data on <a href="https://github.com/LeahS-blip/Planetary-Defense-Spacesift">GitHub</a>.
Data: Minor Planet Center (NEO Confirmation Page, Previous NEOCP, MPECs), NASA JPL Sentry. Rescored twice a day, retrained every Monday.</footer>
<script>
(function(){{
  function wire(inputId, tableId){{
    var inp=document.getElementById(inputId), tbl=document.getElementById(tableId);
    if(!inp||!tbl) return;
    var rows=Array.prototype.filter.call(tbl.rows, function(r){{ return !r.querySelector('th'); }});
    var note=document.getElementById(inputId+'-count');
    function apply(){{
      var q=inp.value.trim().toLowerCase(), shown=0;
      rows.forEach(function(r){{
        var cell=r.querySelector('.mono');
        var hay=(cell?cell.textContent:r.textContent).toLowerCase();
        var match=!q || hay.indexOf(q)>-1;
        r.style.display=match?'':'none';
        if(match) shown++;
      }});
      if(note) note.textContent = q ? (shown+' match'+(shown===1?'':'es')) : '';
    }}
    inp.addEventListener('input', apply);
  }}
  wire('queueSearch','queueTable');
  wire('trackedSearch','trackedTable');
}})();
</script>
<script src="/shared/nav.js"></script>
</body></html>"""
    out = ROOT / "outputs" / "dashboard.html"
    out.write_text(doc, encoding="utf-8")
    print(f"dashboard: {len(vet)} queue rows, {len(all_lab)} labeled -> {out}")

if __name__ == "__main__":
    main()
