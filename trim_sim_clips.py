"""
Trim all simulated accident clips to keep everything up to 5 seconds after
accident_time, i.e. the kept window is [0, accident_time + 5s] clamped to
the original duration.

The corresponding video_annotations JSON (.json.gz or .json) is also truncated
so that base/collision entries whose iteration exceeds the last kept frame are
dropped. The sensor entry is static and is left unchanged.

Outputs are saved to OUTPUT_DIR (processed_data/), preserving the same
relative directory structure. Originals in sim_dataset/ are not modified.
"""

import csv
import gzip
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "/Users/tang/Desktop/Course-Project-ACCIDENT-CVPR/sim_dataset"
OUTPUT_DIR = "/Users/tang/Desktop/Course-Project-ACCIDENT-CVPR/processed_data"
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")
TRAIL_TIME = 5.0  # seconds after accident_time to keep


def resolve_ann_path(base_dir, annotations_path):
    """Return the actual annotation file path, trying .json fallback if .gz not found."""
    full = os.path.join(base_dir, annotations_path)
    if os.path.exists(full):
        return full
    # Try decompressed version
    if full.endswith(".json.gz"):
        alt = full[:-3]  # strip .gz
        if os.path.exists(alt):
            return alt
    return full  # return original so the caller can report MISSING


def trim_annotations(src_path, dst_path, end_iteration):
    """Read src, filter base/collision entries, write to dst."""
    if not os.path.exists(src_path):
        return "MISSING_ANN"
    try:
        opener = gzip.open if src_path.endswith(".gz") else open
        with opener(src_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        data["base"] = [e for e in data["base"] if e["iteration"] <= end_iteration]
        if "collision" in data:
            data["collision"] = [e for e in data["collision"] if e["iteration"] <= end_iteration]

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        dst_opener = gzip.open if dst_path.endswith(".gz") else open
        with dst_opener(dst_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        return "OK"
    except Exception as ex:
        return f"ERROR: {ex}"


def trim_video(row):
    rgb_path = row["rgb_path"]  # e.g. "videos/head-on/Town03_head-on_clear_00.mp4"
    accident_time = float(row["accident_time"])
    duration = float(row["duration"])
    no_frames = int(row["no_frames"])
    annotations_start_offset = int(row["annotations_start_offset"])
    annotations_path = row["annotations_path"]

    # Keep [0, accident_time + TRAIL_TIME], clamped to original duration
    end = min(accident_time + TRAIL_TIME, duration)

    src_video = os.path.join(BASE_DIR, rgb_path)
    dst_video = os.path.join(OUTPUT_DIR, rgb_path)

    if not os.path.exists(src_video):
        return (rgb_path, "MISSING")

    os.makedirs(os.path.dirname(dst_video), exist_ok=True)

    # Write trimmed output to a temp file next to the destination, then replace
    dst_dir = os.path.dirname(dst_video)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=dst_dir)
    os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", src_video,
        "-t", f"{end:.6f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an",   # no audio in sim clips
        tmp_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (rgb_path, f"FAILED: {result.stderr.decode()[-200:]}")

    os.replace(tmp_path, dst_video)

    # Trim annotation: compute last valid iteration
    fps = no_frames / duration
    end_iteration = annotations_start_offset + int(end * fps)

    src_ann = resolve_ann_path(BASE_DIR, annotations_path)
    # Mirror the same relative path (keeping original extension) in output dir
    rel_ann = os.path.relpath(src_ann, BASE_DIR)
    dst_ann = os.path.join(OUTPUT_DIR, rel_ann)

    ann_status = trim_annotations(src_ann, dst_ann, end_iteration)
    if ann_status != "OK":
        return (rgb_path, f"ANN_{ann_status}")

    return (rgb_path, "OK")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy over metadata files
    for fname in ("labels.csv", "annotation_classes.yaml"):
        src = os.path.join(BASE_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, fname))

    with open(LABELS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Trimming {total} simulated clips to [0, accident_time + {TRAIL_TIME}s]...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    done = 0
    failed = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(trim_video, row): row for row in rows}
        for future in as_completed(futures):
            path, status = future.result()
            done += 1
            if status != "OK":
                failed.append((path, status))
                print(f"[{done}/{total}] {status}: {path}")
            elif done % 200 == 0:
                print(f"[{done}/{total}] processed...")

    print(f"\nFinished. {total - len(failed)}/{total} succeeded.")
    if failed:
        print(f"\n{len(failed)} failures:")
        for path, msg in failed:
            print(f"  {path}: {msg}")


if __name__ == "__main__":
    main()
