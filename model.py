"""
Baseline CNN-LSTM model for traffic accident prediction.

Architecture:
    - ResNet-50 backbone (ImageNet pretrained) extracts per-frame spatial features.
    - 2-layer bidirectional LSTM models temporal relationships across T sampled frames.
    - Three prediction heads:
        * accident_time  — regression, sigmoid output in [0, 1]  (normalised frame index)
        * location       — regression, sigmoid output (cx, cy) in [0, 1]
        * collision_type — 5-class classification

Three losses are combined with learnable log-variance weighting (homoscedastic
uncertainty weighting, Kendall & Gal 2018) so no manual tuning of loss weights is needed.

Usage:
    python model.py          # train with default hyper-parameters
    python model.py --help   # see CLI options
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm as tqdm_bar

from dataloader import COLLISION_TYPES, IDX_TO_TYPE, get_dataloaders

NUM_CLASSES = len(COLLISION_TYPES)  # 5


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AccidentPredictor(nn.Module):
    """ResNet-50 + bidirectional LSTM multi-task accident predictor."""

    def __init__(
        self,
        hidden_size: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # --- CNN backbone ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool + fc; keep everything up to layer4
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # -> (B*T, 2048, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                  # -> (B*T, 2048, 1, 1)
        feature_dim = 2048

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False
            # Unfreeze layer4 for light fine-tuning
            for param in self.cnn[-1].parameters():
                param.requires_grad = True

        # --- Temporal LSTM ---
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional

        # --- Prediction heads ---
        def _head(out_dim):
            return nn.Sequential(
                nn.Linear(lstm_out_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, out_dim),
            )

        self.time_head = nn.Sequential(_head(1), nn.Sigmoid())
        self.location_head = nn.Sequential(_head(2), nn.Sigmoid())
        self.type_head = _head(NUM_CLASSES)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)  — batch of frame sequences

        Returns:
            accident_time  : (B,)       floats in [0, 1]
            location       : (B, 2)     floats in [0, 1]
            collision_type : (B, 5)     raw logits
        """
        B, T, C, H, W = x.shape

        # Per-frame CNN features
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)           # (B*T, 2048, 7, 7)
        feats = self.pool(feats)            # (B*T, 2048, 1, 1)
        feats = feats.view(B, T, -1)       # (B, T, 2048)

        # Temporal LSTM
        lstm_out, _ = self.lstm(feats)     # (B, T, hidden*2)
        # Mean-pool across time (more stable than last-step for variable-length content)
        context = lstm_out.mean(dim=1)     # (B, hidden*2)

        accident_time = self.time_head(context).squeeze(-1)   # (B,)
        location = self.location_head(context)                 # (B, 2)
        collision_type = self.type_head(context)               # (B, 5)

        return accident_time, location, collision_type


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class AccidentLoss(nn.Module):
    """
    Multi-task loss with homoscedastic uncertainty weighting.

    Learnable log-variance parameters (log_var_*) automatically balance the
    three tasks without manual weight tuning.
    Reference: Kendall & Gal, "What Uncertainties Do We Need?" NeurIPS 2017.
    """

    def __init__(self):
        super().__init__()
        self.log_var_time = nn.Parameter(torch.zeros(1))
        self.log_var_loc = nn.Parameter(torch.zeros(1))
        self.log_var_type = nn.Parameter(torch.zeros(1))

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_time, pred_loc, pred_type,
                gt_time, gt_loc, gt_type):
        loss_time = self.mse(pred_time, gt_time)
        loss_loc = self.mse(pred_loc, gt_loc)
        loss_type = self.ce(pred_type, gt_type)

        # Uncertainty-weighted combination
        total = (
            torch.exp(-self.log_var_time) * loss_time + self.log_var_time
            + torch.exp(-self.log_var_loc) * loss_loc + self.log_var_loc
            + torch.exp(-self.log_var_type) * loss_type + self.log_var_type
        )
        return total, loss_time, loss_loc, loss_type


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def temporal_score(pred_time: torch.Tensor, gt_time: torch.Tensor,
                   sigma: float = 0.1) -> float:
    """Gaussian-style temporal similarity (T). Errors near 0 → score near 1."""
    return torch.exp(-((pred_time - gt_time) ** 2) / (2 * sigma ** 2)).mean().item()


def spatial_score(pred_loc: torch.Tensor, gt_loc: torch.Tensor,
                  sigma: float = 0.1) -> float:
    """Gaussian-style spatial similarity (S). Small distance → score near 1."""
    dist_sq = (pred_loc - gt_loc).pow(2).sum(dim=-1)
    return torch.exp(-dist_sq / (2 * sigma ** 2)).mean().item()


def classification_accuracy(pred_type: torch.Tensor, gt_type: torch.Tensor) -> float:
    """Top-1 classification accuracy (C)."""
    return pred_type.argmax(dim=-1).eq(gt_type).float().mean().item()


def accident_score(t: float, s: float, c: float) -> float:
    """Competition ACCIDENT score: harmonic mean of T, S, C."""
    if min(t, s, c) == 0:
        return 0.0
    return 3 / (1 / t + 1 / s + 1 / c)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimiser, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm_bar(loader, desc=f"Train epoch {epoch}", leave=False, unit="batch")
    for frames, targets in pbar:
        frames = frames.to(device, non_blocking=True)
        gt_time = targets["accident_time"].to(device, non_blocking=True)
        gt_loc = targets["location"].to(device, non_blocking=True)
        gt_type = targets["type"].to(device, non_blocking=True)

        optimiser.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred_time, pred_loc, pred_type = model(frames)
            loss, *_ = criterion(pred_time, pred_loc, pred_type,
                                 gt_time, gt_loc, gt_type)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_t, all_s, all_cls = [], [], []

    pbar = tqdm_bar(loader, desc="  Val        ", leave=False, unit="batch")
    for frames, targets in pbar:
        frames = frames.to(device, non_blocking=True)
        gt_time = targets["accident_time"].to(device, non_blocking=True)
        gt_loc = targets["location"].to(device, non_blocking=True)
        gt_type = targets["type"].to(device, non_blocking=True)

        pred_time, pred_loc, pred_type = model(frames)
        loss, *_ = criterion(pred_time, pred_loc, pred_type,
                             gt_time, gt_loc, gt_type)

        total_loss += loss.item()
        all_t.append(temporal_score(pred_time, gt_time))
        all_s.append(spatial_score(pred_loc, gt_loc))
        all_cls.append(classification_accuracy(pred_type, gt_type))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_t   = sum(all_t)   / len(all_t)
    avg_s   = sum(all_s)   / len(all_s)
    avg_cls = sum(all_cls) / len(all_cls)
    avg_acc = accident_score(avg_t, avg_s, avg_cls)
    return total_loss / len(loader), avg_t, avg_s, avg_cls, avg_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline accident predictor")
    parser.add_argument("--labels_csv", default="dataset/sim_dataset/labels.csv")
    parser.add_argument("--video_root", default="dataset/sim_dataset")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap dataset size for a quick smoke-test (e.g. --max_samples 32)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        labels_csv=args.labels_csv,
        video_root=args.video_root,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = AccidentPredictor(
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = AccidentLoss().to(device)
    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad]
        + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # --- Training ---
    best_score = 0.0
    epoch_pbar = tqdm_bar(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, scaler, epoch)
        val_loss, t, s, c, acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", ACCIDENT=f"{acc:.3f}")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"T={t:.3f} | S={s:.3f} | C={c:.3f} | ACCIDENT={acc:.3f}"
        )

        if acc > best_score:
            best_score = acc
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "score": score,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  -> Saved best checkpoint (H-score={score:.3f})")

    print(f"\nTraining complete. Best harmonic score: {best_score:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # ----------------------------------------------------------------
        # IDE "Run" button — quick smoke-test config
        # Change these values to adjust the quick run.
        # ----------------------------------------------------------------
        sys.argv += [
            # "--max_samples", "500",
            "--epochs",      "3",
            "--batch_size",  "4",
            "--num_workers", "0",   # 0 = no multiprocessing (required on Windows)
            "--num_frames",  "8",   # fewer frames → faster per-sample loading
        ]
    main()
