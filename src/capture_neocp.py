#!/usr/bin/env python3
"""Capture one NEOCP frame: feed snapshot + tracklets for new objects + outcome labels.

Run this on any cadence (1-2x daily is plenty). Each run:
  1. Fetches the live NEOCP JSON feed  -> data/capture/snapshots/neocp_<UTC>.json
  2. Fetches the 80-col tracklet for any object without one -> data/capture/tracklets/<desig>.obs80
  3. Fetches the Previous-NEOCP page and appends new outcomes -> data/capture/outcomes.csv

Stdlib only. Polite: descriptive User-Agent, 2s pause between requests, cache-busting
query param (MPC's CDN serves stale copies of neocp.txt/json for hours otherwise —
verified 2026-07-10: txt was 12 days stale, JSON 3 days, HTML live).

The NEOCP JSON feed became VALID JSON at some point before 2026-07-10 (the malformed
concatenated-records issue in the project plan Appendix A.1 appears fixed). We still
fall back to a reconstruction parser just in case.

Tracklet gotcha: showobsorbs.cgi returns lines WITHOUT the 5-column number prefix
(75 chars). digest2 needs true 80-col: prepend 5 spaces before scoring (features.py
handles this). Files here store the CGI output as-is.

Tracklets are ONLY available while an object is on the page — once removed they are
gone from public view forever (negatives never get published). That is why this
capture must run continuously; the historical dataset cannot be backfilled.
"""
import json, re, sys, time, csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "data" / "capture"
UA = "planetary-defense-neocp-vetting (research; contact: via GitHub issues)"

FEED_JSON = "https://www.minorplanetcenter.net/Extended_Files/neocp.json"
OBS_CGI = "https://cgi.minorplanetcenter.net/cgi-bin/showobsorbs.cgi?Obj={}&obs=y"
PREV_DES = "https://www.minorplanetcenter.net/iau/NEO/ToConfirm_PrevDes.html"

def fetch(url, cache_bust=True):
    if cache_bust:
        url += ("&" if "?" in url else "?") + f"nocache={int(time.time())}"
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")

def parse_feed(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback: reconstruct from concatenated key/value lines (Appendix A.1)
        recs, cur = [], {}
        for m in re.finditer(r'"([^"]+)"\s*:\s*("(?:[^"]*)"|[-\d.]+)', text):
            k, v = m.group(1), m.group(2).strip('"')
            if k == "Temp_Desig" and cur:
                recs.append(cur); cur = {}
            cur[k] = v
        if cur:
            recs.append(cur)
        return recs

# outcome line patterns on the Previous-NEOCP page (see MPC removal-reasons doc:
# minorplanetcenter.net/mpcops/documentation/neocp-prev-des-removal/ — free-form text,
# so `other` is a catch-all; review data/capture/outcomes.csv periodically for new phrasings)
NEG_REASONS = [
    "was not confirmed", "does not exist", "was not a minor planet",
    "was probably not a real object", "was suspected artificial",
    "was not interesting", "was a star", "had insufficient followup",
    "had bad astrometry",
]

def classify_outcome(line):
    """Return (trksub, outcome_class, designation, mpec) or None.

    outcome_class:
      neo             designated minor planet announced via MPEC (discovery MPECs => NEO)
      designated_other  received a designation but no MPEC (typically an identification
                        with a known non-NEO object, e.g. "2013 UM4 = P12nL5l") — treat as
                        non-NEO for the binary task but verify orbits when feasible
      comet           moved to comet designation (excluded from the NEO binary by default)
      <negative reasons> mapped to non_neo subclasses
      linked          trksub aliased to another NEOCP object (exclude)
    """
    line = line.strip().lstrip("- ").strip()
    mpec = None
    m = re.search(r"MPEC[\s\*\[\]]*(\d{4}-[A-Z]\d{1,3})", line)
    if m:
        mpec = m.group(1)
    line_clean = re.sub(r"\s*\[?\s*(?:see\b|MPEC\b).*$", "", line, flags=re.I)
    m = re.match(r"^(Comet\s+.+?)\s*=\s*(\S+)\s+\((.+?)\)$", line_clean)
    if m:
        return m.group(2), "comet", m.group(1), mpec
    m = re.match(r"^(.+?)\s*=\s*(\S+)\s+\((.+?)\)$", line_clean)
    if m:
        desig, trksub = m.group(1).strip(), m.group(2)
        # trksub = trksub linking (both look like temp desigs, no space in first part)
        if re.match(r"^\S+$", desig) and not re.match(r"^\d{4}\s", desig):
            return trksub, "linked", desig, mpec
        return trksub, ("neo" if mpec else "designated_other"), desig, mpec
    for reason in NEG_REASONS:
        m = re.match(r"^(\S+)\s+" + re.escape(reason) + r"\s*\((.+?)\)$", line_clean)
        if m:
            return m.group(1), reason.replace(" ", "_"), "", None
    m = re.match(r"^(\S+)\s+(.+?)\s*\((.+?)\)$", line_clean)
    if m and len(m.group(1)) <= 8:
        return m.group(1), "other", m.group(2), None
    return None

def parse_prev_des(html):
    """Parse outcome lines from the Previous-NEOCP page.

    Works on raw HTML (converts structural tags to newlines, strips the rest)
    AND on markdown/plain-text renderings — the previous <li>-only version
    silently parsed ZERO outcomes against the real page structure (bug found
    2026-07-16 after two days of lost labels)."""
    text = re.sub(r"(?i)<\s*(?:li|/li|br|p|/p|div|/div|tr|/tr|ul|/ul)[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", "", text)
    out = []
    for raw in text.splitlines():
        txt = raw.strip()
        if not txt or "UT)" not in txt:
            continue
        rec = classify_outcome(txt)
        if rec:
            out.append(rec)
    if not out:
        print("WARNING: parse_prev_des found no outcomes — page format may have changed!")
    return out

def main():
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    (CAP / "snapshots").mkdir(parents=True, exist_ok=True)
    (CAP / "tracklets").mkdir(parents=True, exist_ok=True)

    # 1. feed snapshot
    feed_raw = fetch(FEED_JSON)
    feed = parse_feed(feed_raw)
    snap_path = CAP / "snapshots" / f"neocp_{now}.json"
    snap_path.write_text(json.dumps(feed, indent=1))
    print(f"snapshot: {len(feed)} objects -> {snap_path.name}")

    # 2. tracklets for objects we haven't captured yet (plus refresh if NObs grew a lot)
    have = {p.stem for p in (CAP / "tracklets").glob("*.obs80")}
    new = [o["Temp_Desig"] for o in feed if o["Temp_Desig"] not in have]
    for desig in new:
        time.sleep(2)
        try:
            body = fetch(OBS_CGI.format(desig), cache_bust=False)
        except Exception as e:
            print(f"  tracklet {desig}: FAILED {e}"); continue
        lines = [l for l in body.splitlines() if desig in l and len(l.strip()) > 40]
        if lines:
            (CAP / "tracklets" / f"{desig}.obs80").write_text("\n".join(lines) + "\n")
            print(f"  tracklet {desig}: {len(lines)} obs")
        else:
            print(f"  tracklet {desig}: EMPTY")

    # 3. outcomes
    time.sleep(2)
    html = fetch(PREV_DES)
    outcomes = parse_prev_des(html)
    out_path = CAP / "outcomes.csv"
    seen = set()
    if out_path.exists():
        with open(out_path) as f:
            seen = {(r["trksub"], r["outcome_class"], r["designation"]) for r in csv.DictReader(f)}
    added = 0
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["trksub", "outcome_class", "designation", "mpec", "first_seen_utc"])
        for trksub, oc, desig, mpec in outcomes:
            if (trksub, oc, desig) in seen:
                continue
            w.writerow([trksub, oc, desig, mpec or "", now])
            added += 1
    print(f"outcomes: {len(outcomes)} parsed, {added} new -> outcomes.csv")

if __name__ == "__main__":
    main()
