"""
inference_combined.py — Run inference using the combined augmentation checkpoint.

Submission format:
    path, accident_time, center_x, center_y, type
"""

import os
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import IDX_TO_TYPE, TestDataset, DEFAULT_TRANSFORM
from model import AccidentPredictor

CHECKPOINT   = "checkpoints/brightness_gaussian.pth"
METADATA_CSV = "testset/test_metadata.csv"
VIDEO_ROOT   = "testset"
VIDEOS_ZIP   = "testset/videos.zip"
VIDEOS_DIR   = "testset/videos"
BATCH_SIZE   = 4
NUM_WORKERS  = 0
OUTPUT       = "submission_combined.csv"


def unzip_videos():
    if os.path.isdir(VIDEOS_DIR) and len(os.listdir(VIDEOS_DIR)) > 0:
        print(f"Videos already unzipped at {VIDEOS_DIR}/ ({len(os.listdir(VIDEOS_DIR))} files)")
        return
    print(f"Unzipping {VIDEOS_ZIP} → {VIDEOS_DIR}/ ...")
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    with zipfile.ZipFile(VIDEOS_ZIP, "r") as z:
        for member in tqdm(z.infolist(), desc="Extracting", unit="file"):
            if member.filename.startswith("__MACOSX"):
                continue
            member.filename = os.path.basename(member.filename)
            if member.filename:
                z.extract(member, VIDEOS_DIR)
    print(f"Done. {len(os.listdir(VIDEOS_DIR))} videos extracted.")


@torch.no_grad()
def run_inference(model, loader, device, duration_map):
    model.eval()
    rows = []

    for frames, paths in tqdm(loader, desc="Inference", unit="batch"):
        frames = frames.to(device, non_blocking=True)
        pred_time, pred_loc, pred_type = model(frames)

        pred_time = pred_time.cpu()
        pred_loc  = pred_loc.cpu()
        pred_cls  = pred_type.argmax(dim=-1).cpu()

        for i, path in enumerate(paths):
            duration      = duration_map[path]
            accident_time = round(pred_time[i].item() * duration, 3)
            center_x      = round(pred_loc[i, 0].item(), 6)
            center_y      = round(pred_loc[i, 1].item(), 6)
            col_type      = IDX_TO_TYPE[pred_cls[i].item()]

            rows.append({
                "path":          path,
                "accident_time": accident_time,
                "center_x":      center_x,
                "center_y":      center_y,
                "type":          col_type,
            })

    return pd.DataFrame(rows)


def main():
    unzip_videos()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device     : {device}")
    print(f"Checkpoint : {CHECKPOINT}")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    saved_args = ckpt.get("args", {})
    model = AccidentPredictor(
        hidden_size=saved_args.get("hidden_size", 512),
        num_lstm_layers=saved_args.get("num_lstm_layers", 2),
        dropout=saved_args.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    meta_df = pd.read_csv(METADATA_CSV)
    duration_map = dict(zip(meta_df["path"], meta_df["duration"].astype(float)))

    num_frames = saved_args.get("num_frames", 16)
    test_ds = TestDataset(
        metadata_csv=METADATA_CSV,
        video_root=VIDEO_ROOT,
        num_frames=num_frames,
        transform=DEFAULT_TRANSFORM,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")

    df = run_inference(model, test_loader, device, duration_map)
    df = df.set_index("path").reindex(meta_df["path"]).reset_index()

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} predictions → {OUTPUT}")
    print(df.head())


if __name__ == "__main__":
    main()
