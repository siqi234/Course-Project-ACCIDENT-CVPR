"""
Generate submission CSV from a trained checkpoint.
Edit the config block below and run directly.
"""

import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader import IDX_TO_TYPE, TestDataset, DEFAULT_TRANSFORM
from model import AccidentPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT   = "checkpoints/best_model.pth"
METADATA_CSV = "dataset/test_metadata.csv"
VIDEO_ROOT   = "dataset/test_dataset"
NUM_FRAMES   = 16
BATCH_SIZE   = 4
NUM_WORKERS  = 0
OUTPUT       = "submission.csv"
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(model, loader, device, no_frames_map):
    model.eval()
    rows = []

    for frames, paths in tqdm(loader, desc="Inference", unit="batch"):
        frames = frames.to(device, non_blocking=True)
        pred_time, pred_loc, pred_type = model(frames)

        pred_time = pred_time.cpu()
        pred_loc  = pred_loc.cpu()
        pred_cls  = pred_type.argmax(dim=-1).cpu()

        for i, path in enumerate(paths):
            no_frames      = no_frames_map[path]
            accident_time  = pred_time[i].item()
            accident_frame = round(accident_time * no_frames)
            center_x       = pred_loc[i, 0].item()
            center_y       = pred_loc[i, 1].item()
            col_type       = IDX_TO_TYPE[pred_cls[i].item()]

            rows.append({
                "path":           path,
                "accident_time":  round(accident_time, 6),
                "accident_frame": accident_frame,
                "center_x":       round(center_x, 6),
                "center_y":       round(center_y, 6),
                "type":           col_type,
            })

    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    saved_args = ckpt.get("args", {})

    model = AccidentPredictor(
        hidden_size=saved_args.get("hidden_size", 512),
        num_lstm_layers=saved_args.get("num_lstm_layers", 2),
        dropout=saved_args.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint  (epoch {ckpt.get('epoch', '?')}, H-score={ckpt.get('score', '?'):.4f})")

    meta_df = pd.read_csv(METADATA_CSV)
    no_frames_map = dict(zip(meta_df["path"], meta_df["no_frames"].astype(int)))

    num_frames = saved_args.get("num_frames", NUM_FRAMES)
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

    df = run_inference(model, test_loader, device, no_frames_map)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} predictions -> {OUTPUT}")


if __name__ == "__main__":
    main()
