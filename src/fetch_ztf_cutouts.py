#!/usr/bin/env python3
"""Fetch ZTF difference-image cutouts for ZTF-detected NEOCP objects.

RUN ON YOUR MACHINE (IRSA is unreachable from the Cowork sandbox). Resumable.

  python3 src/fetch_ztf_cutouts.py            # all objects with an I41 detection
  python3 src/fetch_ztf_cutouts.py --limit 30

For every tracklet (capture + backfill) containing a ZTF observation (obs code
I41), we take that observation's own RA/Dec/UT — so the asteroid is centered in
the cutout by construction (avoids the ephemeris-offset problem noted in the v1
plan Appendix C.4). Then, per the validated Appendix C.2 recipe:

  1. IBE metadata search for the science exposure covering that position+date
  2. build the _scimrefdiffimg (difference image) cutout URL, 60arcsec box
  3. save the FITS to data/capture/ztf_cutouts/<trksub>.fits

ztf_features.py then turns each cutout into per-object ztf_* columns that FUSE
with the tabular features (same object => real fusion, unlike the CSS set).

Needs: numpy, astropy (only for the optional --verify read-back). stdlib fetch.
"""
import argparse, csv, io, re, sys, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = ROOT / "data" / "capture"
OUT = CAP / "ztf_cutouts"
UA = "planetary-defense-neocp-vetting (research; contact: via GitHub issues)"
IBE_SEARCH = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IBE_DATA = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"

# ZTF public data release covers ~2018-2024. Live NEOCP objects (2026) have NO
# public difference images (collaboration-proprietary, ~1-2yr release lag), so we
# only attempt cutouts for I41 observations inside the public window. This
# pipeline is the correct TEMPLATE for Rubin/LSST real-time public cutouts, and
# it activates retroactively for objects as ZTF releases catch up.
ZTF_PUBLIC_MIN, ZTF_PUBLIC_MAX = "2018", "2024"

def parse_i41_obs(lines):
    """Return (ra, dec, 'YYYY-MM-DD') for the EARLIEST ZTF obs in the public window."""
    cands = []
    for l in lines:
        if len(l) < 75:
            continue
        code = l[77:80].strip() if len(l) >= 80 else l[72:75].strip()
        if code != "I41":
            continue
        s = l if l[:5].strip() == "" else "     " + l
        try:
            yy, mm = int(s[15:19]), int(s[20:22])
            dd = float(s[23:32]); day = int(dd)
            ra = 15.0 * (int(s[32:34]) + int(s[35:37]) / 60 + float(s[38:44]) / 3600)
            sign = -1 if s[44] == "-" else 1
            dec = sign * (int(s[45:47]) + int(s[48:50]) / 60 + float(s[51:56]) / 3600)
        except (ValueError, IndexError):
            continue
        cands.append((f"{yy:04d}-{mm:02d}-{day:02d}", ra, dec))
    public = [c for c in cands if ZTF_PUBLIC_MIN <= c[0][:4] <= ZTF_PUBLIC_MAX]
    if not public:
        return None  # only recent obs -> no public ZTF cutout exists
    date, ra, dec = min(public)
    return ra, dec, date, date

def http_get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def find_exposure(ra, dec, date):
    where = urllib.parse.quote(f"obsdate >= '{date} 00:00:00' AND obsdate <= '{date} 23:59:59'")
    url = f"{IBE_SEARCH}?POS={ra:.5f},{dec:.5f}&WHERE={where}&ct=csv"
    txt = http_get(url).decode("utf-8", "replace").strip().splitlines()
    if len(txt) < 2:
        return None
    hdr = txt[0].split(",")
    for row in txt[1:]:
        rec = dict(zip(hdr, row.split(",")))
        if rec.get("filefracday"):
            return rec
    return None

def cutout_url(rec, ra, dec):
    ffd = rec["filefracday"]                    # e.g. 20190104084051
    year, monthday, fracday = ffd[:4], ffd[4:8], ffd[8:]
    field = f'{int(rec["field"]):06d}'
    ccd = f'{int(rec["ccdid"]):02d}'
    q = rec["qid"]; filt = rec["filtercode"]
    fn = f"ztf_{ffd}_{field}_{filt}_c{ccd}_o_q{q}_scimrefdiffimg.fits.fz"
    return (f"{IBE_DATA}/{year}/{monthday}/{fracday}/{fn}"
            f"?center={ra:.5f},{dec:.5f}&size=60arcsec&gzip=false")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # gather every tracklet dir that may hold ZTF detections
    tk_dirs = [CAP / "tracklets", CAP / "tracklets_api", CAP / "tracklets_mpec"]
    seen = {}
    for d in tk_dirs:
        if not d.exists():
            continue
        for f in d.glob("*.obs80"):
            if f.stem in seen or (OUT / f"{f.stem}.fits").exists():
                continue
            info = parse_i41_obs(f.read_text().splitlines())
            if info:
                seen[f.stem] = info
    todo = list(seen.items())
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} ZTF-detected objects to fetch cutouts for")
    ok = miss = 0
    for trksub, (ra, dec, d0, d1) in todo:
        try:
            rec = find_exposure(ra, dec, d0)
            if not rec:
                print(f"  {trksub}: no ZTF exposure found"); miss += 1
                time.sleep(args.sleep); continue
            data = http_get(cutout_url(rec, ra, dec))
            if len(data) < 2000 or b"<html" in data[:200].lower():
                print(f"  {trksub}: empty/invalid cutout"); miss += 1
            else:
                (OUT / f"{trksub}.fits").write_bytes(data)
                ok += 1
                if ok % 10 == 0:
                    print(f"  {ok} cutouts saved")
        except Exception as e:
            print(f"  {trksub}: {e}"); miss += 1
        time.sleep(args.sleep)
    print(f"done: {ok} cutouts, {miss} missing -> {OUT}")

if __name__ == "__main__":
    main()
