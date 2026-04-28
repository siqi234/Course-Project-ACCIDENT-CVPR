"""
data_augmentation_quarter.py — Train with a 4-way per-clip augmentation split.

Each training clip is assigned one of four transforms with equal probability:
  - 25%: brightness/contrast + Gaussian noise (both)
  - 25%: brightness/contrast only
  - 25%: Gaussian noise only
  - 25%: no augmentation

Validation always uses DEFAULT_TRANSFORM (no augmentation).
Best checkpoint is saved by val_loss.
"""

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import AccidentDataset, DEFAULT_TRANSFORM
from model import AccidentLoss, AccidentPredictor, evaluate, train_one_epoch


# ---------------------------------------------------------------------------
# Custom transform
# ---------------------------------------------------------------------------

class GaussianNoise(nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std + self.mean


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

BOTH_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    GaussianNoise(mean=0.0, std=0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BRIGHTNESS_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

GAUSSIAN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    GaussianNoise(mean=0.0, std=0.05),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORMS = [BOTH_TRANSFORM, BRIGHTNESS_TRANSFORM, GAUSSIAN_TRANSFORM, DEFAULT_TRANSFORM]


# ---------------------------------------------------------------------------
# Dataset: 4-way uniform random choice per clip
# ---------------------------------------------------------------------------

class QuarterAugDataset(AccidentDataset):
    def __init__(self, df, video_root, num_frames):
        super().__init__(df, video_root, num_frames, transform=DEFAULT_TRANSFORM)

    def __getitem__(self, idx):
        self.transform = TRANSFORMS[torch.randint(4, (1,)).item()]
        return super().__getitem__(idx)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def make_loaders(args):
    df = pd.read_csv(args.labels_csv)
    if args.max_samples is not None:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed, stratify=df["type"]
    )

    train_ds = QuarterAugDataset(train_df, args.video_root, args.num_frames)
    val_ds   = AccidentDataset(val_df, args.video_root, args.num_frames, DEFAULT_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(args, device) -> dict:
    torch.manual_seed(args.seed)

    train_loader, val_loader = make_loaders(args)
    print(f"\n{'='*64}")
    print(f"  Augmentation : 25% both | 25% brightness | 25% gaussian | 25% none")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"{'='*64}")

    model = AccidentPredictor(
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = AccidentLoss().to(device)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "augmentation_quarter.pth")

    best_val_loss = float("inf")
    train_losses: list = []
    val_losses: list   = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, scaler, epoch)
        val_loss, t, s, c, acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(round(train_loss, 6))
        val_losses.append(round(val_loss, 6))

        print(
            f"  Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"T={t:.3f} | S={s:.3f} | C={c:.3f} | ACCIDENT={acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                    "augmentation": "25%_both+25%_brightness+25%_gaussian+25%_none",
                },
                ckpt_path,
            )
            print(f"    -> Saved checkpoint (val_loss={val_loss:.4f}) → {ckpt_path}")

    print(f"  Best val_loss={best_val_loss:.4f}  checkpoint → {ckpt_path}")
    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train with 4-way augmentation split")
    p.add_argument("--labels_csv",      default="processed_data/labels.csv")
    p.add_argument("--video_root",      default="processed_data")
    p.add_argument("--num_frames",      type=int,   default=16)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--hidden_size",     type=int,   default=512)
    p.add_argument("--num_lstm_layers", type=int,   default=2)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--val_split",       type=float, default=0.2)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--max_samples",     type=int,   default=None)
    p.add_argument("--results_dir",     default="aug_results")
    p.add_argument("--save_dir",        default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    print("Aug    : 25% both | 25% brightness | 25% gaussian | 25% none")

    os.makedirs(args.results_dir, exist_ok=True)
    result = run_experiment(args, device)

    out_path = os.path.join(args.results_dir, "augmentation_quarter.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), **result}, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--epochs",      "10",
            "--batch_size",  "4",
            "--num_workers", "2",
            "--num_frames",  "10",
        ]
    main()
