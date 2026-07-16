#!/usr/bin/env python3
"""Turn backfilled tracklets into labeled feature rows.

Sources (either or both):
  data/capture/tracklets_mpec/<trksub>.obs80   (from MPEC pages)
  data/capture/tracklets_api/<trksub>.obs80    (from the get-obs API)

Critical step — POSTING-TIME FAITHFULNESS: published astrometry includes
follow-up and precovery observations. We trim each object to its DISCOVERY
TRACKLET: the earliest night's observations from the discovery observatory
only. That approximates what digest2 saw when the object was posted. (Feed
fields like Score/V/NObs at posting time are unknowable retroactively and
left empty — models must tolerate that; see train.py --with-backfill guard.)

Output: data/processed/backfill_labeled.csv (same schema family as labeled.csv,
plus provenance column source=backfill_mpec|backfill_api).

Usage: python3 src/build_backfill.py [digest2_dir]
"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import parse_obs80, tracklet_features, run_digest2

def discovery_tracklet(lines, window_end_t=None, window_days=120):
    """NEOCP-posting tracklet: earliest night within the posting window.

    Published astrometry for identified objects includes PRECOVERY obs going
    back decades. The tracklet that sat on the NEOCP is the earliest night
    within `window_days` before the outcome was recorded (window_end_t, days
    since J2000). Without a window, falls back to earliest night overall."""
    obs = [parse_obs80(l) for l in lines if l.strip()]
    obs = sorted([o for o in obs if o], key=lambda o: o["t"])
    if window_end_t is not None:
        inwin = [o for o in obs if window_end_t - window_days <= o["t"] <= window_end_t + 2]
        if inwin:
            obs = inwin
    if not obs:
        return [], []
    first_night = int(obs[0]["t"])
    disc_code = obs[0]["obscode"]
    disc = [o for o in obs if int(o["t"]) == first_night and o["obscode"] == disc_code]
    raw = []
    for l in lines:
        p = parse_obs80(l)
        if p and int(p["t"]) == first_night and p["obscode"] == disc_code:
            raw.append(l)
    return disc, raw

def main(digest2_dir=None):
    root = Path(__file__).resolve().parent.parent
    cap = root / "data" / "capture"
    with open(cap / "outcomes.csv") as f:
        outcomes = {r["trksub"]: r for r in csv.DictReader(f)}
    import tempfile
    rows, trimmed_dir = [], Path(tempfile.mkdtemp(prefix="disc_trk_"))
    for src_name, d in (("backfill_mpec", cap / "tracklets_mpec"),
                        ("backfill_api", cap / "tracklets_api")):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.obs80")):
            trksub = p.stem
            o = outcomes.get(trksub)
            if o is None or o["outcome_class"] not in ("neo", "designated_other"):
                continue
            if any(r["trksub"] == trksub for r in rows):   # mpec takes precedence
                continue
            fs = o.get("first_seen_utc", "")
            wet = None
            if len(fs) >= 8:
                from datetime import datetime
                wet = (datetime(int(fs[:4]), int(fs[4:6]), int(fs[6:8]))
                       - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0
            disc, raw = discovery_tracklet(p.read_text().splitlines(), window_end_t=wet)
            if len(disc) < 2:
                continue
            tf = tracklet_features(disc)
            if not tf:
                continue
            (trimmed_dir / f"{trksub}.obs80").write_text("\n".join(raw) + "\n")
            l0 = "     " + raw[0] if not raw[0].startswith("     ") else raw[0]
            disc_ymd = f"{l0[15:19]}-{l0[20:22]}-{float(l0[23:32] or 0):04.1f}"
            row = {"trksub": trksub, "y": 1 if o["outcome_class"] == "neo" else 0,
                   "outcome_class": o["outcome_class"], "designation": o["designation"],
                   "source": src_name, "disc_ymd": disc_ymd}
            row.update(tf)
            rows.append(row)
    # digest2 on the trimmed discovery tracklets (packed desig inside the file
    # differs from trksub, so map by file order via per-file scoring key)
    import os
    precomputed = os.environ.get("D2_PRECOMPUTED")  # path to raw digest2 stdout
    if precomputed and Path(precomputed).exists():
        import re
        from features import D2_CLASSES
        d2 = {}
        for line in Path(precomputed).read_text().splitlines():
            m = re.match(r"^(\S+)\s+([\d.*]+)\s+((?:\d+\s+){25}\d+)\s*$", line)
            if not m:
                continue
            desig, rms, scores = m.group(1), m.group(2), m.group(3).split()
            row = {"d2_rms": None if "*" in rms else float(rms)}
            for i, cls in enumerate(D2_CLASSES):
                row[f"d2_{cls}_raw"] = int(scores[2 * i])
                row[f"d2_{cls}_noid"] = int(scores[2 * i + 1])
            d2[desig] = row
        print(f"loaded {len(d2)} precomputed digest2 rows from {precomputed}")
        for r in rows:
            f = trimmed_dir / f'{r["trksub"]}.obs80'
            if f.exists():
                first = parse_obs80(f.read_text().splitlines()[0])
                if first and first["desig"] in d2:
                    r.update(d2[first["desig"]])
    elif digest2_dir:
        d2 = run_digest2(sorted(trimmed_dir.glob("*.obs80")), digest2_dir)
        for r in rows:
            f = trimmed_dir / f'{r["trksub"]}.obs80'
            if not f.exists():
                continue
            first = parse_obs80(f.read_text().splitlines()[0])
            if first and first["desig"] in d2:
                r.update(d2[first["desig"]])
    out = root / "data" / "processed" / "backfill_labeled.csv"
    if rows:
        keys = sorted({k for r in rows for k in r},
                      key=lambda k: (k != "trksub", k != "y", k))
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    n_pos = sum(r["y"] for r in rows)
    n_d2 = sum(1 for r in rows if r.get("d2_NEO_noid") is not None)
    print(f"backfill: {len(rows)} labeled rows ({n_pos} NEO / {len(rows)-n_pos} non-NEO), "
          f"{n_d2} with digest2 sub-scores -> {out}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
