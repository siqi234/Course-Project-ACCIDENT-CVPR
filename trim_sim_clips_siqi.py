"""
Trim sim_dataset clips from video start to accident_time + POST_SEC.
Keeps the full pre-accident context (accident at its natural position)
and removes the long post-accident tail.

Output layout:
  dataset/sim_dataset_trimmed/
    videos/<type>/<name>.mp4
    video_annotations/<name>.json.gz
    labels.csv
"""

import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# ── config ──────────────────────────────────────────────────────────────────
SIM_ROOT   = Path("dataset/sim_dataset")
OUT_ROOT   = Path("dataset/sim_dataset_trimmed")
LABELS_IN  = SIM_ROOT / "labels.csv"
LABELS_OUT = OUT_ROOT / "labels.csv"

POST_SEC = 5.0   # seconds to keep after accident
# ────────────────────────────────────────────────────────────────────────────


def ffmpeg_trim(src: Path, dst: Path, t_start: float, t_end: float) -> None:
    """Losslessly cut src to [t_start, t_end] and write dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = t_end - t_start
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{t_start:.6f}",
        "-i", str(src),
        "-t", f"{duration:.6f}",
        "-c", "copy",          # stream-copy: fast, no re-encode
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr}")


def trim_annotations(data: dict, start_frame: int, end_frame: int) -> dict:
    """
    Keep annotation frames in [start_frame, end_frame] and
    re-index iteration numbers so the clip starts at iteration 0.
    """
    def in_range(it: int) -> bool:
        return start_frame <= it <= end_frame

    new_base = [
        {**entry, "iteration": entry["iteration"] - start_frame}
        for entry in data.get("base", [])
        if in_range(entry["iteration"])
    ]
    new_collision = [
        {**entry, "iteration": entry["iteration"] - start_frame}
        for entry in data.get("collision", [])
        if in_range(entry["iteration"])
    ]
    # sensor is metadata (2 entries), keep as-is
    return {"base": new_base, "collision": new_collision, "sensor": data.get("sensor", [])}


def process(row: dict, writer, skipped: list) -> None:
    video_rel   = row["rgb_path"]          # e.g. videos/sideswipe/Town05_…mp4
    annot_rel   = row["annotations_path"]  # e.g. video_annotations/…json.gz
    accident_time  = float(row["accident_time"])
    accident_frame = int(row["accident_frame"])
    no_frames      = int(row["no_frames"])
    duration       = float(row["duration"])
    ann_offset     = int(row["annotations_start_offset"])

    fps = no_frames / duration  # derive fps from metadata

    # ── trim window: keep from start to accident + POST_SEC ──────────────
    t_start = 0.0
    t_end   = min(duration, accident_time + POST_SEC)

    f_start = 0
    f_end   = min(no_frames - 1, math.ceil(t_end * fps))

    # ── trim video ───────────────────────────────────────────────────────
    src_video = SIM_ROOT / video_rel
    dst_video = OUT_ROOT / video_rel
    if not src_video.exists():
        skipped.append(str(src_video))
        return

    ffmpeg_trim(src_video, dst_video, t_start, t_end)

    # ── trim annotations ─────────────────────────────────────────────────
    src_annot = SIM_ROOT / annot_rel
    dst_annot = OUT_ROOT / annot_rel
    dst_annot.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(src_annot, "rt") as f:
        ann_data = json.load(f)

    new_ann = trim_annotations(ann_data, f_start, f_end)

    with gzip.open(dst_annot, "wt") as f:
        json.dump(new_ann, f, separators=(",", ":"))

    # ── recompute labels ─────────────────────────────────────────────────
    new_no_frames      = f_end - f_start + 1
    new_duration       = new_no_frames / fps
    new_accident_frame = accident_frame - f_start
    new_accident_time  = new_accident_frame / fps
    new_ann_offset     = max(0, ann_offset - f_start)

    new_row = dict(row)
    new_row["accident_time"]             = round(new_accident_time, 6)
    new_row["accident_frame"]            = new_accident_frame
    new_row["no_frames"]                 = new_no_frames
    new_row["duration"]                  = round(new_duration, 6)
    new_row["annotations_start_offset"]  = new_ann_offset

    writer.writerow(new_row)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy over non-video assets (yaml, etc.)
    for f in SIM_ROOT.iterdir():
        if f.is_file() and f.suffix not in (".mp4", ".gz", ".csv"):
            shutil.copy2(f, OUT_ROOT / f.name)

    skipped = []

    with open(LABELS_IN, newline="", encoding="utf-8") as fin, \
         open(LABELS_OUT, "w", newline="", encoding="utf-8") as fout:

        reader  = csv.DictReader(fin)
        writer  = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        total = 0
        done  = 0
        for row in reader:
            total += 1
            try:
                process(row, writer, skipped)
                done += 1
                if done % 50 == 0:
                    print(f"  {done} / {total} done …", flush=True)
            except Exception as e:
                print(f"[ERROR] {row['rgb_path']}: {e}", file=sys.stderr)

    print(f"\nFinished: {done}/{total} clips trimmed.")
    if skipped:
        print(f"Skipped (missing source): {len(skipped)}")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
