"""
Fetch CSS detection sequences and expert labels from Zooniverse.

Pulls subjects from "The Daily Minor Planet" project (Catalina Sky Survey)
via the public Zooniverse Panoptes API. Each subject includes:
- 4 PNG frames (~6 min apart) showing a candidate detection
- Expert ground-truth label (real asteroid or not)
- Tabular metadata (Digest2 score, magnitude, object type)

No authentication required for subjects + metadata.
"""

import urllib.request
import json
import os
from datetime import datetime, timezone

ZOONIVERSE_API = "https://www.zooniverse.org/api/subjects"
PROJECT_ID = "19413"
WORKFLOW_ID = "22354"
USER_AGENT = "planetary-defense-sample/0.1 (research; polite single pull)"
OUTPUT_DIR = "data/css_zooniverse"


def fetch_subjects(n=2):
    url = f"{ZOONIVERSE_API}?project_id={PROJECT_ID}&workflow_id={WORKFLOW_ID}&page_size={n}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.api+json; version=1",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_label(metadata):
    """Extract expert ground-truth label from subject metadata."""
    feedback_answer = metadata.get("#feedback_2_answer")
    keyword = metadata.get("!sKeyword", "")
    obj_type = metadata.get("!sType", "")

    if feedback_answer == "0":
        is_real = True
        label_source = "expert_feedback"
    elif feedback_answer == "1":
        is_real = False
        label_source = "expert_feedback"
    elif keyword == "rjct":
        is_real = False
        label_source = "keyword_rjct"
    else:
        is_real = None
        label_source = "unlabeled"

    return {
        "is_real_asteroid": is_real,
        "label_source": label_source,
        "object_type": obj_type,
        "keyword": keyword,
        "feedback_answer_raw": feedback_answer,
        "success_message": metadata.get("#feedback_2_successMessage", ""),
        "failure_message": metadata.get("#feedback_2_failureMessage", ""),
    }


def download_image(url, filepath):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(filepath, "wb") as f:
        f.write(data)
    return len(data)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching 2 subjects from Zooniverse (Daily Minor Planet / CSS)...")
    data = fetch_subjects(n=2)
    subjects = data.get("subjects", [])
    print(f"Retrieved {len(subjects)} subjects\n")

    results = []

    for subj in subjects:
        subj_id = subj["id"]
        metadata = subj.get("metadata", {})
        locations = subj.get("locations", [])

        label_info = parse_label(metadata)

        subj_dir = os.path.join(OUTPUT_DIR, f"subject_{subj_id}")
        os.makedirs(subj_dir, exist_ok=True)

        print(f"Subject {subj_id}:")
        print(f"  Type: {metadata.get('!sType', '?')} | Digest2: {metadata.get('fDigest2', '?')} | Mag: {metadata.get('sMag', '?')}")
        print(f"  Label: is_real={label_info['is_real_asteroid']} (source: {label_info['label_source']})")

        frame_paths = []
        for i, loc in enumerate(locations):
            img_url = loc.get("image/png", "")
            if not img_url:
                continue
            fname = f"frame_{i+1}.png"
            fpath = os.path.join(subj_dir, fname)
            size = download_image(img_url, fpath)
            frame_paths.append(fname)
            print(f"  Frame {i+1}: {size:,} bytes -> {fpath}")

        subject_record = {
            "subject_id": subj_id,
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metadata": {
                "object_type": metadata.get("!sType", ""),
                "digest2_score": metadata.get("fDigest2"),
                "magnitude": metadata.get("sMag"),
                "keyword": metadata.get("!sKeyword", ""),
                "designation": metadata.get("!sDesig", "").strip(),
                "detection_number": metadata.get("sDet", ""),
                "detection_file": metadata.get("sDetsFile", ""),
            },
            "label": label_info,
            "frames": frame_paths,
        }
        results.append(subject_record)

        subj_json_path = os.path.join(subj_dir, "subject.json")
        with open(subj_json_path, "w") as f:
            json.dump(subject_record, f, indent=2)
        print(f"  Metadata -> {subj_json_path}")
        print()

    summary_path = os.path.join(OUTPUT_DIR, "fetch_summary.json")
    summary = {
        "source": "Zooniverse Daily Minor Planet (CSS)",
        "project_id": PROJECT_ID,
        "workflow_id": WORKFLOW_ID,
        "api_url": f"{ZOONIVERSE_API}?project_id={PROJECT_ID}&workflow_id={WORKFLOW_ID}",
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_subjects": len(results),
        "subjects": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
