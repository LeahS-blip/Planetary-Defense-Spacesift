"""
Visual display of one NEOCP frame: sky map, score distribution, and size/brightness plot.
"""

import urllib.request
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

NEOCP_TXT_URL = "https://www.minorplanetcenter.net/iau/NEO/neocp.txt"
USER_AGENT = "planetary-defense-sample/0.1 (research; polite single pull)"


def fetch_neocp_frame():
    req = urllib.request.Request(NEOCP_TXT_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_neocp_line(line):
    parts = line.split()
    if len(parts) < 12:
        return None
    try:
        ra_hrs = float(parts[5])
        dec_deg = float(parts[6])
        ra_deg = ra_hrs * 15.0
        return {
            "temp_desig": parts[0],
            "score": int(parts[1]),
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "v_mag": float(parts[7]),
            "n_obs": int(parts[-4]),
            "arc": float(parts[-3]),
            "h_mag": float(parts[-2]),
            "not_seen_days": float(parts[-1]),
        }
    except (ValueError, IndexError):
        return None


def score_color(score):
    if score >= 90:
        return "#ff3333"
    elif score >= 70:
        return "#ff9933"
    else:
        return "#3399ff"


def score_label(score):
    if score >= 90:
        return "High (>=90)"
    elif score >= 70:
        return "Medium (70-89)"
    else:
        return "Low (<70)"


def main():
    print("Fetching live NEOCP data...")
    raw = fetch_neocp_frame()
    lines = [l for l in raw.strip().splitlines() if l.strip()]
    objects = [parse_neocp_line(l) for l in lines]
    objects = [o for o in objects if o is not None]

    if not objects:
        print("No objects on NEOCP.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"NEOCP Live Snapshot - {len(objects)} candidates\n{timestamp}",
                 fontsize=13, fontweight="bold")

    # --- Panel 1: Sky map (RA/Dec) ---
    ax1 = fig.add_subplot(2, 2, 1, projection="aitoff")
    ax1.set_title("Sky Position (Aitoff projection)", fontsize=10, pad=12)
    ax1.grid(True, alpha=0.3)

    for obj in objects:
        ra_rad = np.radians(obj["ra_deg"] - 180)
        dec_rad = np.radians(obj["dec_deg"])
        size = max(30, 150 - obj["v_mag"] * 5)
        ax1.scatter(ra_rad, dec_rad, s=size, c=score_color(obj["score"]),
                    edgecolors="black", linewidths=0.5, zorder=5, alpha=0.85)

    legend_patches = [
        mpatches.Patch(color="#ff3333", label="Score >=90 (likely NEO)"),
        mpatches.Patch(color="#ff9933", label="Score 70-89"),
        mpatches.Patch(color="#3399ff", label="Score <70 (likely false pos.)"),
    ]
    ax1.legend(handles=legend_patches, loc="lower right", fontsize=7)

    # --- Panel 2: Digest2 score bar chart ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Digest2 Score per Candidate", fontsize=10)

    sorted_objs = sorted(objects, key=lambda o: o["score"], reverse=True)
    names = [o["temp_desig"] for o in sorted_objs]
    scores = [o["score"] for o in sorted_objs]
    colors = [score_color(s) for s in scores]

    bars = ax2.barh(range(len(names)), scores, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8, fontfamily="monospace")
    ax2.set_xlabel("Digest2 Score")
    ax2.set_xlim(0, 105)
    ax2.axvline(x=65, color="gray", linestyle="--", alpha=0.7, label="NEOCP posting threshold (65)")
    ax2.axvline(x=90, color="red", linestyle=":", alpha=0.7, label="High-confidence (90)")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.invert_yaxis()

    # --- Panel 3: H magnitude (size proxy) vs apparent mag ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Absolute Mag (H) vs Apparent Mag (V)", fontsize=10)

    for obj in objects:
        ax3.scatter(obj["v_mag"], obj["h_mag"], s=80, c=score_color(obj["score"]),
                    edgecolors="black", linewidths=0.5, alpha=0.85)
        ax3.annotate(obj["temp_desig"], (obj["v_mag"], obj["h_mag"]),
                     fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                     textcoords="offset points")

    ax3.set_xlabel("Apparent magnitude V (fainter ->)")
    ax3.set_ylabel("Absolute magnitude H (smaller = larger object)")
    ax3.invert_yaxis()

    ax3.axhline(y=22, color="orange", linestyle="--", alpha=0.5)
    ax3.text(ax3.get_xlim()[0] + 0.1, 22.3, "~140m (city-killer)", fontsize=7, color="orange")
    ax3.axhline(y=26, color="red", linestyle="--", alpha=0.5)
    ax3.text(ax3.get_xlim()[0] + 0.1, 26.3, "~20m (Chelyabinsk-class)", fontsize=7, color="red")

    # --- Panel 4: Staleness vs arc length ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("Staleness vs Observation Arc", fontsize=10)

    for obj in objects:
        ax4.scatter(obj["arc"], obj["not_seen_days"], s=obj["n_obs"] * 5,
                    c=score_color(obj["score"]), edgecolors="black", linewidths=0.5, alpha=0.85)
        ax4.annotate(obj["temp_desig"], (obj["arc"], obj["not_seen_days"]),
                     fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                     textcoords="offset points")

    ax4.set_xlabel("Arc length (days)")
    ax4.set_ylabel("Days since last seen")
    ax4.axhline(y=3, color="red", linestyle=":", alpha=0.5)
    ax4.text(0.1, 3.2, "Loss risk zone (>3 days unseen)", fontsize=7, color="red")

    size_legend = [
        plt.scatter([], [], s=5*5, c="gray", edgecolors="black", linewidths=0.5, label="5 obs"),
        plt.scatter([], [], s=20*5, c="gray", edgecolors="black", linewidths=0.5, label="20 obs"),
        plt.scatter([], [], s=50*5, c="gray", edgecolors="black", linewidths=0.5, label="50 obs"),
    ]
    ax4.legend(handles=size_legend, title="Bubble = NObs", fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig("neocp_snapshot.png", dpi=150, bbox_inches="tight")
    print("Saved: neocp_snapshot.png")
    plt.show()


if __name__ == "__main__":
    main()
