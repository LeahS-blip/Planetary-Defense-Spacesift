#!/usr/bin/env python3
"""Backfill discovery observations for all DESIGNATED objects in outcomes.csv.

RUN ON YOUR OWN MACHINE (the MPC data API needs POST requests, which the Cowork
sandbox cannot send). One run fetches observations for:
  * outcome_class == "neo"              -> positives (currently ~286)
  * outcome_class == "designated_other" -> non-NEO negatives, mostly MBAs (~154)

  python3 src/backfill_designated.py           # everything not yet on disk
  python3 src/backfill_designated.py --limit 50

Output: data/capture/tracklets_api/<trksub>.obs80 (full published astrometry,
80-col). build_backfill.py trims to the discovery night and computes features.

Uses https://data.minorplanetcenter.net/api/get-obs (the API for designated
objects; the NEOCP variant only serves objects still on the page).
API docs: https://docs.minorplanetcenter.net/mpc-ops-docs/apis/get-obs/
Be polite: ~1 req/2s, descriptive User-Agent, resumable (skips existing files).

Note the class asymmetry this does NOT fix: artifacts / satellites / bogus
tracklets ("was not confirmed", "does not exist", ...) are never published and
can only come from the live capture. Until the capture supplies those, the
negatives here are real-but-non-NEO objects only — fine for the NEO-vs-MBA
axis, blind to the junk axis. train.py guards against this.
"""
import argparse, csv, json, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "data" / "capture"
OUT = CAP / "tracklets_api"
API = "https://data.minorplanetcenter.net/api/get-obs"
UA = "planetary-defense-neocp-vetting (research; contact: via GitHub issues)"

def get_obs(designation):
    body = json.dumps({"desigs": [designation], "output_format": ["OBS80"]}).encode()
    req = urllib.request.Request(API, data=body, method="GET",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        payload = json.loads(r.read().decode())
    return payload[0].get("OBS80", "") if payload else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CAP / "outcomes.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["outcome_class"] in ("neo", "designated_other") and r["designation"]]
    todo = [r for r in rows if not (OUT / f'{r["trksub"]}.obs80').exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} designated objects in outcomes.csv; {len(todo)} to fetch")
    ok = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            obs = get_obs(r["designation"])
        except Exception as e:
            print(f'  {r["designation"]}: {e}'); fail += 1
            time.sleep(10); continue
        lines = [l for l in obs.splitlines() if len(l.strip()) > 40]
        if lines:
            (OUT / f'{r["trksub"]}.obs80').write_text("\n".join(lines) + "\n")
            ok += 1
        else:
            print(f'  {r["designation"]}: no obs returned'); fail += 1
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} done")
        time.sleep(args.sleep)
    print(f"done: {ok} written, {fail} failed -> {OUT}")

if __name__ == "__main__":
    main()
