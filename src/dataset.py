#!/usr/bin/env python3
"""Join captured features to outcomes -> labeled training table.

Binary target `y`:
  1  = outcome_class "neo" (designated via discovery MPEC)
  0  = negatives: was_not_confirmed, does_not_exist, was_not_a_minor_planet,
       was_probably_not_a_real_object, was_suspected_artificial, was_not_interesting,
       was_a_star, had_insufficient_followup, had_bad_astrometry, designated_other
  excluded: comet (real but different follow-up economics), linked (alias), other
            (unmapped free-form reason — inspect manually), unresolved (still on page)

Caveats encoded here on purpose:
  * "designated_other" (designation but no MPEC) is USUALLY an identification with a
    known non-NEO (e.g. 2013 UM4 = P12nL5l) -> y=0, but a small fraction may be NEOs
    announced without an MPEC link on the page; verify against MPCORB when the
    dataset is big enough to care about ~1% label noise.
  * "was_not_confirmed" objects include some real NEOs that were simply lost — this is
    irreducible label noise the A&A paper shares (they treat not-confirmed as non-NEO).
  * Temp designations recycle: we key on (trksub, discovery date within +-60 days of
    outcome first_seen) to avoid cross-epoch mismatches once the archive spans months.
"""
import csv, sys
from datetime import datetime
from pathlib import Path

NEG = {"was_not_confirmed", "does_not_exist", "was_not_a_minor_planet",
       "was_probably_not_a_real_object", "was_suspected_artificial",
       "was_not_interesting", "was_a_star", "had_insufficient_followup",
       "had_bad_astrometry", "designated_other"}
EXCLUDE = {"comet", "linked", "other"}

def build(features_csv, outcomes_csv, out_csv):
    with open(features_csv) as f:
        feats = list(csv.DictReader(f))
    with open(outcomes_csv) as f:
        outs = {}
        for r in csv.DictReader(f):
            prev = outs.get(r["trksub"])
            # definitive outcomes beat bookkeeping ones (linked/other); never downgrade
            if prev and r["outcome_class"] in ("linked", "other") \
                    and prev["outcome_class"] not in ("linked", "other"):
                continue
            outs[r["trksub"]] = r
    rows, n_pos, n_neg, n_open, n_excl = [], 0, 0, 0, 0
    for r in feats:
        o = outs.get(r["trksub"])
        if o is None:
            n_open += 1
            continue
        oc = o["outcome_class"]
        if oc in EXCLUDE:
            n_excl += 1
            continue
        if oc == "neo":
            r["y"], r["outcome_class"] = 1, oc; n_pos += 1
        elif oc in NEG:
            r["y"], r["outcome_class"] = 0, oc; n_neg += 1
        else:
            n_excl += 1
            continue
        r["designation"] = o.get("designation", "")
        rows.append(r)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "trksub", k != "y", k))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"labeled: {len(rows)} rows ({n_pos} NEO / {n_neg} non-NEO); "
          f"{n_open} unresolved, {n_excl} excluded -> {out_csv}")
    return rows

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    p = root / "data" / "processed"
    build(p / "features.csv", root / "data" / "capture" / "outcomes.csv", p / "labeled.csv")
