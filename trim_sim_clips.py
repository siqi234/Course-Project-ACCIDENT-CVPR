"""
Trim all simulated accident clips to 3 seconds:
  - 2 seconds BEFORE accident_time
  - 1 second AFTER accident_time
Edge cases:
  - If accident_time < 2, start at 0 (clip = [0, 3])
  - If accident_time + 1 > duration, shift window back so end = duration
Overwrites originals in place. Does not touch test_metadata.csv or /videos/.
"""

import csv
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = "/tmp/accident_dataset/sim_dataset"
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")
CLIP_DURATION = 3.0
LEAD_TIME = 2.0   # seconds before accident
TRAIL_TIME = 1.0  # seconds after accident


def trim_video(row):
    rgb_path = row["rgb_path"]  # e.g. "videos/head-on/Town03_head-on_clear_00.mp4"
    accident_time = float(row["accident_time"])
    duration = float(row["duration"])

    # Compute start/end, clamped to [0, duration]
    start = max(0.0, accident_time - LEAD_TIME)
    end = start + CLIP_DURATION
    if end > duration:
        end = duration
        start = max(0.0, duration - CLIP_DURATION)

    video_path = os.path.join(BASE_DIR, rgb_path)
    if not os.path.exists(video_path):
        return (rgb_path, "MISSING")

    # Write trimmed output to a temp file next to the original, then replace
    dir_name = os.path.dirname(video_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=dir_name)
    os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.6f}",
        "-i", video_path,
        "-t", f"{CLIP_DURATION:.6f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an",   # no audio in sim clips
        tmp_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        os.replace(tmp_path, video_path)
        return (rgb_path, "OK")
    else:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return (rgb_path, f"FAILED: {result.stderr.decode()[-200:]}")


def main():
    with open(LABELS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Trimming {total} simulated clips to {CLIP_DURATION}s "
          f"({LEAD_TIME}s before + {TRAIL_TIME}s after accident)...\n")

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
