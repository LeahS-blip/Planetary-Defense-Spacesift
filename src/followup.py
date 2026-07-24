#!/usr/bin/env python3
"""Follow-up status for NEOCP candidates — a work-queue tag for the follow-up network.

Turns the ranked vetting list into an actionable queue: which objects still need
telescope time, which are already being tracked, and which are slipping away.

Derived from the feed's own observation bookkeeping (no extra data needed):
  nobs          number of observations so far
  arc_days      span of the observation arc (a single-night discovery is ~hours << 1 day;
                a value >= ~1 day means the object was re-observed on a later night)
  not_seen_dys  days since the most recent observation

Tiers, checked in order:
  going_stale     not seen in > STALE_DAYS  -> at risk of being lost; needs re-acquisition
  tracked         multi-night arc (>= MULTINIGHT_ARC d) AND seen recently -> someone's on it
  needs_followup  fresh single-night candidate seen recently -> nobody has followed up yet

Thresholds are deliberately simple and live here so both the CSV (predict.py) and the
dashboard (make_dashboard.py) classify identically. Tune in one place.
"""

STALE_DAYS = 3.0
MULTINIGHT_ARC = 1.0  # days

# order matters for display (queue priority: act on NEW/STALE, skip TRACKED)
STATUS = {
    "needs_followup": {"label": "Needs follow-up", "short": "NEW",     "color": "#4ade80"},
    "going_stale":    {"label": "Going stale",     "short": "STALE",   "color": "#fbbf24"},
    "tracked":        {"label": "Being tracked",   "short": "TRACKED", "color": "#60a5fa"},
    "unknown":        {"label": "Unknown",         "short": "?",       "color": "#64748b"},
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify(nobs=None, arc_days=None, not_seen_dys=None):
    """Return a status key from STATUS given the three tracking numbers."""
    arc, ns = _f(arc_days), _f(not_seen_dys)
    if arc is None and ns is None:
        return "unknown"
    if ns is not None and ns > STALE_DAYS:
        return "going_stale"
    if arc is not None and arc >= MULTINIGHT_ARC:
        return "tracked"
    return "needs_followup"


def classify_row(r):
    """Classify a dict row from either daily_vetting.csv (nobs/arc_days/not_seen_dys)
    or a feature table (nobs_feed/arc_feed/not_seen_dys)."""
    def g(*keys):
        for k in keys:
            v = r.get(k)
            if v not in (None, "", "None"):
                return v
        return None
    return classify(g("nobs", "nobs_feed"), g("arc_days", "arc_feed"), g("not_seen_dys"))
