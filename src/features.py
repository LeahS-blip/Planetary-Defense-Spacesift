#!/usr/bin/env python3
"""Feature engineering for NEOCP tracklets.

Two feature groups per object:
  A. digest2 sub-scores (the A&A 2025 feature set): 13 orbit classes x {raw, noid}
     = 26 features, computed by invoking the digest2 binary on the tracklet.
  B. Tracklet/metadata features the A&A paper did NOT use (our edge):
     motion rate & direction, ecliptic latitude, solar elongation, great-circle RMS,
     nobs, arc, mag stats, observatory code, plus feed fields (Score, V, H, Not_Seen_dys).

digest2 binary: build from https://github.com/Smithsonian/digest2 (see README section
"Reproducing the environment"). Pass its directory via --digest2-dir; it must contain
digest2, digest2.model, digest2.obscodes. If missing, group A is skipped.

Note the CGI 75-char line quirk: we prepend 5 spaces to make true 80-col before scoring.
"""
import math, re, subprocess, tempfile, sys, json, csv
from pathlib import Path

D2_CLASSES = ["Int", "NEO", "MC", "Hun", "Pho", "Han", "MB1", "Pal", "MB2", "MB3", "Hil", "JTr", "JFC"]
D2_CONFIG = "headings\nrms\nraw\nnoid\n" + "\n".join(D2_CLASSES) + "\n"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

def parse_obs80(line):
    """Parse one 80-col (or CGI 75-col) observation line -> dict or None."""
    if len(line) < 70:
        return None
    if not line.startswith("     ") and re.match(r"^\S", line):
        line = "     " + line  # CGI variant
    try:
        desig = line[5:12].strip()
        yy, mm = int(line[15:19]), int(line[20:22])
        dd = float(line[23:32])
        ra_h, ra_m, ra_s = int(line[32:34]), int(line[35:37]), float(line[38:44])
        de_sign = -1 if line[44] == "-" else 1
        de_d, de_m, de_s = int(line[45:47]), int(line[48:50]), float(line[51:56])
        mag = line[65:70].strip()
        obscode = line[77:80].strip()
    except (ValueError, IndexError):
        return None
    ra_deg = 15.0 * (ra_h + ra_m / 60 + ra_s / 3600)
    dec_deg = de_sign * (de_d + de_m / 60 + de_s / 3600)
    # days since J2000.0 (good enough for solar-longitude & MJD-ish ordering)
    a = (14 - mm) // 12; y = yy + 4800 - a; m2 = mm + 12 * a - 3
    jdn = (mm and 1) and ( (153 * m2 + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045)
    t_days = jdn - 0.5 + dd - int(dd) + (int(dd) - 1) - 2451544.5 + 1  # JD - J2000
    return {"desig": desig, "t": t_days, "ra": ra_deg, "dec": dec_deg,
            "mag": float(mag) if mag else None, "obscode": obscode}

def _unit(ra, dec):
    ra, dec = math.radians(ra), math.radians(dec)
    return (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec))

def ecliptic_latitude(ra, dec):
    eps = math.radians(23.4393)
    x, y, z = _unit(ra, dec)
    ye = y * math.cos(eps) + z * math.sin(eps)
    ze = -y * math.sin(eps) + z * math.cos(eps)
    return math.degrees(math.asin(ze)), math.degrees(math.atan2(ye, x)) % 360

def solar_longitude(t_days):
    # low-precision sun (Meeus): mean longitude + equation of center
    T = t_days / 36525.0
    L0 = (280.46646 + 36000.76983 * T) % 360
    M = math.radians((357.52911 + 35999.05029 * T) % 360)
    C = (1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M)
    return (L0 + C) % 360

def tracklet_features(obs):
    """Geometric/motion features from a list of parsed observations (>=2)."""
    obs = sorted([o for o in obs if o], key=lambda o: o["t"])
    if len(obs) < 2:
        return None
    first, last = obs[0], obs[-1]
    dt = max(last["t"] - first["t"], 1e-6)
    cosd = math.cos(math.radians((first["dec"] + last["dec"]) / 2))
    dra = (last["ra"] - first["ra"])
    dra = (dra + 180) % 360 - 180
    d_ra_deg = dra * cosd
    d_dec_deg = last["dec"] - first["dec"]
    sky_rate = math.hypot(d_ra_deg, d_dec_deg) / dt          # deg/day
    pa = math.degrees(math.atan2(d_ra_deg, d_dec_deg)) % 360  # position angle of motion
    beta, lam = ecliptic_latitude(first["ra"], first["dec"])
    sl = solar_longitude(first["t"])
    elong = abs((lam - sl + 180) % 360 - 180)                 # solar elongation proxy, deg
    opp_dist = abs((lam - (sl + 180)) % 360)
    opp_dist = min(opp_dist, 360 - opp_dist)                  # distance from opposition, deg
    mags = [o["mag"] for o in obs if o["mag"] is not None]
    # great-circle residual RMS (arcsec): fit linear motion, measure scatter
    res = []
    for o in obs:
        f = (o["t"] - first["t"]) / dt
        pra = first["ra"] + dra * f
        pdec = first["dec"] + d_dec_deg * f
        res.append(((((o["ra"] - pra + 180) % 360 - 180) * cosd) ** 2 + (o["dec"] - pdec) ** 2) ** 0.5)
    rms_arcsec = (sum(r * r for r in res) / len(res)) ** 0.5 * 3600
    span_nights = round(dt)
    return {
        "nobs_tracklet": len(obs),
        "arc_days_tracklet": dt,
        "sky_rate_deg_day": sky_rate,
        "motion_pa_deg": pa,
        "ecl_lat_deg": beta,
        "solar_elong_deg": elong,
        "opposition_dist_deg": opp_dist,
        "gc_rms_arcsec": rms_arcsec,
        "mag_mean": sum(mags) / len(mags) if mags else None,
        "mag_range": (max(mags) - min(mags)) if len(mags) > 1 else 0.0,
        "obscode_first": obs[0]["obscode"],
        "multi_night": int(span_nights >= 1),
    }

def run_digest2(obs80_paths, digest2_dir):
    """Batch-score tracklet files; returns {desig: {<Class>_raw, <Class>_noid, d2_rms}}."""
    d2 = Path(digest2_dir)
    exe = d2 / "digest2"
    if not exe.exists():
        return {}
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "all.config"; cfg.write_text(D2_CONFIG)
        allobs = Path(td) / "all.obs"
        with open(allobs, "w") as f:
            for p in obs80_paths:
                for line in Path(p).read_text().splitlines():
                    if line.strip():
                        f.write(("     " + line if not line.startswith("     ") else line) + "\n")
        r = subprocess.run(
            [str(exe), "-m", str(d2 / "digest2.model"), "-o", str(d2 / "digest2.obscodes"),
             "-c", str(cfg), str(allobs)],
            capture_output=True, text=True, timeout=600)
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(\S+)\s+([\d.*]+)\s+((?:\d+\s+){25}\d+)\s*$", line)
        if not m:
            continue
        desig, rms, scores = m.group(1), m.group(2), m.group(3).split()
        row = {"d2_rms": None if "*" in rms else float(rms)}
        for i, cls in enumerate(D2_CLASSES):
            row[f"d2_{cls}_raw"] = int(scores[2 * i])
            row[f"d2_{cls}_noid"] = int(scores[2 * i + 1])
        out[desig] = row
    return out

def build_feature_table(capture_dir, digest2_dir=None, out_csv=None):
    """Combine latest snapshot fields + tracklet features + digest2 scores per object."""
    cap = Path(capture_dir)
    latest = {}
    for snap in sorted((cap / "snapshots").glob("neocp_*.json")):
        for rec in json.loads(snap.read_text()):
            d = dict(rec); d["snapshot"] = snap.stem
            latest.setdefault(rec["Temp_Desig"], d)  # keep FIRST posting (deployment-faithful)
            latest[rec["Temp_Desig"]]["last_snapshot"] = snap.stem
            for k in ("NObs", "Arc", "Not_Seen_dys", "Score", "V", "H"):
                latest[rec["Temp_Desig"]][f"latest_{k}"] = rec.get(k)
    tk_paths = list((cap / "tracklets").glob("*.obs80"))
    d2 = run_digest2(tk_paths, digest2_dir) if digest2_dir else {}
    if digest2_dir and tk_paths and not d2:
        print("WARNING: digest2 returned no scores — check the binary/model paths; "
              "d2_* feature columns will be missing!", file=sys.stderr)
    rows = []
    for desig, snap in latest.items():
        row = {"trksub": desig,
               "feed_score": snap.get("Score"), "V": snap.get("V"), "H": snap.get("H"),
               "arc_feed": snap.get("Arc"), "nobs_feed": snap.get("NObs"),
               "not_seen_dys": snap.get("Not_Seen_dys"),
               "ra_feed_hr": snap.get("R.A."), "dec_feed_deg": snap.get("Decl."),
               "disc_ymd": f'{snap.get("Discovery_year")}-{snap.get("Discovery_month"):02d}-{snap.get("Discovery_day"):04.1f}'
                           if snap.get("Discovery_year") else ""}
        tkf = cap / "tracklets" / f"{desig}.obs80"
        if tkf.exists():
            obs = [parse_obs80(l) for l in tkf.read_text().splitlines() if l.strip()]
            tf = tracklet_features(obs)
            if tf:
                row.update(tf)
        row.update(d2.get(desig, {}))
        rows.append(row)
    if out_csv and rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "trksub", k))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"features: {len(rows)} objects -> {out_csv}")
    return rows

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    d2dir = sys.argv[1] if len(sys.argv) > 1 else None
    build_feature_table(root / "data" / "capture", d2dir, root / "data" / "processed" / "features.csv")
