#!/usr/bin/env python3
"""Bulk-fetch CSS detection sequences + expert labels from Zooniverse.

RUN THIS ON YOUR OWN MACHINE (not in the Cowork sandbox — Zooniverse is not
reachable from there). Resumable: re-running skips subjects already on disk.

  python3 src/fetch_css_bulk.py --max-subjects 2000        # ~1-2 GB, ~30-60 min
  python3 src/fetch_css_bulk.py --max-subjects 200         # quick starter set

Output layout (same as the July-4 sample fetch):
  data/css_zooniverse/subject_<id>/frame_[1-4].png + subject.json
  data/css_zooniverse/css_subjects.csv   (metadata + label index, appended)

Labels: '#feedback_2_answer' == "0" -> real asteroid, "1" -> not real.
Subjects without feedback labels are still saved (usable for pseudo-labeling
or the volunteer-consensus route later) but flagged unlabeled in the CSV.
"""
import argparse, csv, json, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://www.zooniverse.org/api/subjects"
PROJECT_ID = "19413"
WORKFLOW_ID = "22354"
UA = "planetary-defense-neocp-vetting/0.2 (research; contact via GitHub issues)"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "css_zooniverse"

def api_get(url, timeout=60):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.api+json; version=1", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def dl(url, path, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        path.write_bytes(r.read())

def parse_label(md):
    fb = md.get("#feedback_2_answer")
    if fb == "0":  return True, "expert_feedback"
    if fb == "1":  return False, "expert_feedback"
    if md.get("!sKeyword") == "rjct": return False, "keyword_rjct"
    return None, "unlabeled"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=2000)
    ap.add_argument("--page-size", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--workflow", default=WORKFLOW_ID,
                    help="22354=validation (default), 24646=feedback (expert-labeled), 31240=Bok")
    ap.add_argument("--sort", default="id",
                    help="'id' = oldest first (retired/labeled subjects), '-id' = newest first")
    ap.add_argument("--labeled-only", action="store_true",
                    help="only save subjects that carry an expert feedback label")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / "css_subjects.csv"
    have = {p.name.replace("subject_", "") for p in OUT.glob("subject_*")}
    indexed = set()
    if index_path.exists():
        with open(index_path) as f:
            indexed = {r["subject_id"] for r in csv.DictReader(f)}
    write_header = not index_path.exists()
    f_idx = open(index_path, "a", newline="")
    w = csv.writer(f_idx)
    if write_header:
        w.writerow(["subject_id", "is_real", "label_source", "object_type", "keyword",
                    "digest2", "magnitude", "detection_file", "n_frames", "fetched_utc"])

    fetched, page = 0, 1
    while fetched < args.max_subjects:
        url = (f"{API}?project_id={PROJECT_ID}&workflow_id={args.workflow}"
               f"&page_size={args.page_size}&page={page}&sort={args.sort}")
        try:
            data = api_get(url)
        except Exception as e:
            print(f"page {page}: {e}; retrying in 20s"); time.sleep(20); continue
        subjects = data.get("subjects", [])
        if not subjects:
            print("no more subjects"); break
        for s in subjects:
            sid = str(s["id"])
            if sid in have and sid in indexed:
                continue
            md_peek = s.get("metadata", {})
            if args.labeled_only and md_peek.get("#feedback_2_answer") not in ("0", "1"):
                continue
            sdir = OUT / f"subject_{sid}"
            sdir.mkdir(exist_ok=True)
            md = s.get("metadata", {})
            frames = []
            for i, loc in enumerate(s.get("locations", []), 1):
                url_png = list(loc.values())[0]
                fp = sdir / f"frame_{i}.png"
                if not fp.exists():
                    try:
                        dl(url_png, fp); time.sleep(args.sleep * 0.3)
                    except Exception as e:
                        print(f"  {sid} frame {i}: {e}"); continue
                frames.append(fp.name)
            is_real, src = parse_label(md)
            rec = {"subject_id": sid,
                   "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "metadata": {"object_type": md.get("!sType", ""),
                                 "digest2_score": md.get("fDigest2"),
                                 "magnitude": md.get("sMag"),
                                 "keyword": md.get("!sKeyword", ""),
                                 "designation": md.get("sDesig", ""),
                                 "detection_number": md.get("sDetNum", ""),
                                 "detection_file": md.get("sDetsFile", "")},
                   "label": {"is_real_asteroid": is_real, "label_source": src,
                              "feedback_answer_raw": md.get("#feedback_2_answer"),
                              "success_message": md.get("#feedback_2_successMessage", ""),
                              "failure_message": md.get("#feedback_2_failureMessage", "")},
                   "frames": frames}
            (sdir / "subject.json").write_text(json.dumps(rec, indent=1))
            if sid not in indexed:
                w.writerow([sid, is_real, src, md.get("!sType", ""), md.get("!sKeyword", ""),
                            md.get("fDigest2"), md.get("sMag"), md.get("sDetsFile", ""),
                            len(frames), rec["fetched_utc"]])
            fetched += 1
            if fetched % 50 == 0:
                print(f"{fetched} subjects fetched"); f_idx.flush()
            if fetched >= args.max_subjects:
                break
        page += 1
        time.sleep(args.sleep)
    f_idx.close()
    print(f"done: {fetched} new subjects -> {OUT}")

if __name__ == "__main__":
    main()
