#!/usr/bin/env python3
"""Rebuild data/capture/outcomes.csv from all saved prev_des_raw_*.txt files.

Used by the automated capture: each run saves the Previous-NEOCP page's outcome
lines to data/capture/prev_des_raw_<YYYYMMDD>.txt (plain text, one outcome per
line, optional " [MPEC 2026-XXX]" suffix), then runs this script to merge and
dedup everything into outcomes.csv. Designated-with-MPEC => class "neo".
"""
import csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_neocp import classify_outcome

def main():
    root = Path(__file__).resolve().parent.parent
    cap = root / "data" / "capture"
    seen, recs = set(), []
    for p in sorted(cap.glob("prev_des_raw_*.txt")):
        stamp = p.stem.replace("prev_des_raw_", "")
        for l in p.read_text().splitlines():
            l = l.strip()
            if not l:
                continue
            mpec = None
            m = re.search(r"\[MPEC\s+([0-9A-Z-]+)\]", l)
            if m:
                mpec = m.group(1)
                l = re.sub(r"\s*\[MPEC[^\]]*\]", "", l)
            r = classify_outcome(l)
            if not r:
                print("UNPARSED:", l[:100])
                continue
            trk, oc, desig, _ = r
            if mpec and oc == "designated_other":
                oc = "neo"
            key = (trk, oc, desig)
            if key in seen:
                continue
            seen.add(key)
            recs.append((trk, oc, desig, mpec or "", stamp))
    with open(cap / "outcomes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trksub", "outcome_class", "designation", "mpec", "first_seen_utc"])
        w.writerows(recs)
    print(f"outcomes.csv rebuilt: {len(recs)} unique outcomes "
          f"from {len(list(cap.glob('prev_des_raw_*.txt')))} raw files")

if __name__ == "__main__":
    main()
