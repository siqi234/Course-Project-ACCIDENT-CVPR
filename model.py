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

def temporal_accuracy(pred_time: torch.Tensor, gt_time: torch.Tensor,
                      tolerance: float = 0.05) -> float:
    """Fraction of predictions within `tolerance` of the ground-truth normalised frame."""
    return (pred_time - gt_time).abs().le(tolerance).float().mean().item()


def spatial_accuracy(pred_loc: torch.Tensor, gt_loc: torch.Tensor,
                     threshold: float = 0.1) -> float:
    """Fraction of predictions whose Euclidean distance to GT centre is below threshold."""
    dist = (pred_loc - gt_loc).pow(2).sum(dim=-1).sqrt()
    return dist.le(threshold).float().mean().item()


def classification_accuracy(pred_type: torch.Tensor, gt_type: torch.Tensor) -> float:
    """Fraction of correctly predicted collision types."""
    return pred_type.argmax(dim=-1).eq(gt_type).float().mean().item()


def harmonic_score(t_acc: float, s_acc: float) -> float:
    """Harmonic mean of temporal and spatial accuracy (competition metric)."""
    if t_acc + s_acc == 0:
        return 0.0
    return 2 * t_acc * s_acc / (t_acc + s_acc)


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
        with torch.amp.autocast(device_type=device.type):
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
    all_t_acc, all_s_acc, all_cls_acc = [], [], []

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
        all_t_acc.append(temporal_accuracy(pred_time, gt_time))
        all_s_acc.append(spatial_accuracy(pred_loc, gt_loc))
        all_cls_acc.append(classification_accuracy(pred_type, gt_type))
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_t = sum(all_t_acc) / len(all_t_acc)
    avg_s = sum(all_s_acc) / len(all_s_acc)
    avg_cls = sum(all_cls_acc) / len(all_cls_acc)
    return total_loss / len(loader), avg_t, avg_s, harmonic_score(avg_t, avg_s), avg_cls


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config — edit here, then run directly
# ---------------------------------------------------------------------------
LABELS_CSV      = "dataset/sim_dataset/labels.csv"
VIDEO_ROOT      = "dataset/sim_dataset"
NUM_FRAMES      = 16
BATCH_SIZE      = 8
EPOCHS          = 30
LR              = 1e-4
HIDDEN_SIZE     = 512
NUM_LSTM_LAYERS = 2
DROPOUT         = 0.3
VAL_SPLIT       = 0.2
NUM_WORKERS     = 0        # 0 = no multiprocessing (required on Windows)
SAVE_DIR        = "checkpoints"
SEED            = 42
AUGMENTATION    = "motion_blur"     # None | "motion_blur" | "resolution"
MAX_SAMPLES     = None     # set e.g. 500 for a quick smoke-test
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        labels_csv=LABELS_CSV,
        video_root=VIDEO_ROOT,
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS,
        seed=SEED,
        augmentation=AUGMENTATION,
        max_samples=MAX_SAMPLES,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = AccidentPredictor(
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = AccidentLoss().to(device)
    optimiser = AdamW(
        [p for p in model.parameters() if p.requires_grad]
        + list(criterion.parameters()),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    # --- Training ---
    best_score = 0.0
    epoch_pbar = tqdm_bar(range(1, EPOCHS + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, scaler, epoch)
        val_loss, t_acc, s_acc, score, cls_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", H=f"{score:.3f}", cls=f"{cls_acc:.3f}")

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"T-acc={t_acc:.3f} | S-acc={s_acc:.3f} | Cls-acc={cls_acc:.3f} | H-score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "criterion_state": criterion.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "score": score,
                    "args": {
                        "hidden_size": HIDDEN_SIZE, "num_lstm_layers": NUM_LSTM_LAYERS,
                        "dropout": DROPOUT, "num_frames": NUM_FRAMES,
                    },
                },
                ckpt_path,
            )
            print(f"  -> Saved best checkpoint (H-score={score:.3f})")

    print(f"\nTraining complete. Best harmonic score: {best_score:.4f}")


if __name__ == "__main__":
    main()
