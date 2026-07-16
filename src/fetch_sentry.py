#!/usr/bin/env python3
"""Fetch NASA/JPL Sentry impact-risk list -> data/capture/sentry.json

Sentry is JPL's automated impact monitor: for every known asteroid whose orbit
cannot yet rule out an Earth impact, it computes impact probabilities over the
next ~100 years. API docs: https://ssd-api.jpl.nasa.gov/doc/sentry.html

We keep the objects most worth watching, ranked by cumulative Palermo scale
(log10 of risk relative to background; > -2 is 'merits attention', > 0 would
be genuinely serious — nothing is currently close to that).

Runs in the GitHub Actions cycle; safe to run manually too.
"""
import json, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "capture" / "sentry.json"
API = "https://ssd-api.jpl.nasa.gov/sentry.api"
UA = "planetary-defense-neocp-vetting (research; contact: via GitHub issues)"
TOP_N = 12

def main():
    req = urllib.request.Request(API, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        payload = json.loads(r.read().decode())
    rows = payload.get("data", [])
    def f(x, default=None):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default
    objs = []
    for r in rows:
        objs.append({
            "des": r.get("des", ""),
            "fullname": (r.get("fullname") or r.get("des", "")).strip(),
            "ip_cum": f(r.get("ip"), 0.0),          # cumulative impact probability
            "ps_cum": f(r.get("ps_cum"), -99),      # cumulative Palermo scale
            "ts_max": r.get("ts_max"),              # max Torino scale
            "diameter_km": f(r.get("diameter")),
            "h": f(r.get("h")),
            "n_imp": r.get("n_imp"),                # number of potential impacts
            "range": r.get("range", ""),            # years of potential impacts
            "last_obs": r.get("last_obs", ""),
        })
    objs.sort(key=lambda o: -(o["ps_cum"] if o["ps_cum"] is not None else -99))
    out = {
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_monitored": len(objs),
        "top": objs[:TOP_N],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"sentry: {len(objs)} monitored objects, top {TOP_N} by Palermo -> {OUT}")

if __name__ == "__main__":
    main()
