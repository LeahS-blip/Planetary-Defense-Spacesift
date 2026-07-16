"""
Sample script: pull one frame of the live NEOCP feed and display it.

The NEOCP (NEO Confirmation Page) lists newly-detected objects awaiting
orbit confirmation — these are the un-catalogued candidates that may be
real near-Earth objects or false positives.

Data source: Minor Planet Center plain-text feed.
"""

import urllib.request
from datetime import datetime, timezone

NEOCP_TXT_URL = "https://www.minorplanetcenter.net/iau/NEO/neocp.txt"
USER_AGENT = "planetary-defense-sample/0.1 (research; polite single pull)"


def fetch_neocp_frame():
    """Fetch one snapshot of the NEOCP plain-text feed."""
    req = urllib.request.Request(NEOCP_TXT_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return raw


def parse_neocp_line(line):
    """
    Parse a single NEOCP text line into a dict.

    Column layout (fixed-width-ish with an embedded free-text 'Updated' field):
      Temp_Desig  Score  Discovery(Y M D.d)  R.A.  Decl.  V  Updated...UT  NObs  Arc  H  Not_Seen_dys

    The 'Updated' field contains spaces (e.g. "July  4.23 UT"), so we parse
    from both ends and treat the middle as free text.
    """
    parts = line.split()
    if len(parts) < 12:
        return None

    temp_desig = parts[0]
    score = int(parts[1])

    disc_year = int(parts[2])
    disc_month = int(parts[3])
    disc_day = float(parts[4])

    ra = parts[5]
    decl = parts[6]
    v_mag = parts[7]

    not_seen = float(parts[-1])
    h_mag = parts[-2]
    arc = parts[-3]
    n_obs = int(parts[-4])

    updated_tokens = parts[8:-4]
    updated_str = " ".join(updated_tokens)

    return {
        "temp_desig": temp_desig,
        "score": score,
        "disc_year": disc_year,
        "disc_month": disc_month,
        "disc_day": disc_day,
        "ra": ra,
        "decl": decl,
        "v_mag": v_mag,
        "updated": updated_str,
        "n_obs": n_obs,
        "arc": arc,
        "h_mag": h_mag,
        "not_seen_days": not_seen,
    }


def display_frame(objects):
    """Pretty-print the NEOCP frame as a table."""
    print(f"\n{'='*80}")
    print(f"  NEOCP LIVE SNAPSHOT - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  {len(objects)} candidates awaiting confirmation")
    print(f"{'='*80}\n")

    header = f"{'Desig':<10} {'D2':>3} {'R.A.':>8} {'Decl.':>8} {'Vmag':>5} {'NObs':>4} {'Arc':>5} {'H':>5} {'Stale(d)':>8}"
    print(header)
    print("-" * len(header))

    for obj in objects:
        print(
            f"{obj['temp_desig']:<10} "
            f"{obj['score']:>3} "
            f"{obj['ra']:>8} "
            f"{obj['decl']:>8} "
            f"{obj['v_mag']:>5} "
            f"{obj['n_obs']:>4} "
            f"{obj['arc']:>5} "
            f"{obj['h_mag']:>5} "
            f"{obj['not_seen_days']:>8.2f}"
        )

    print(f"\n{'-'*80}")
    scores = [o["score"] for o in objects]
    high_score = [o for o in objects if o["score"] >= 90]
    print(f"  Digest2 score range: {min(scores)}-{max(scores)}  |  "
          f"High-confidence (>=90): {len(high_score)} objects")
    print(f"  (Score >65 = posted to NEOCP; higher = more likely real NEO)")
    print()


def main():
    print("Fetching live NEOCP data from Minor Planet Center...")
    raw = fetch_neocp_frame()

    lines = [l for l in raw.strip().splitlines() if l.strip()]
    objects = []
    for line in lines:
        parsed = parse_neocp_line(line)
        if parsed:
            objects.append(parsed)

    if not objects:
        print("No objects currently on the NEOCP (or parsing failed).")
        return

    objects.sort(key=lambda o: o["score"], reverse=True)
    display_frame(objects)


if __name__ == "__main__":
    main()
