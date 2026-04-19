import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from data_augmentation import build_train_transform

COLLISION_TYPES = ['head-on', 'rear-end', 'sideswipe', 'single', 't-bone']
TYPE_TO_IDX = {t: i for i, t in enumerate(COLLISION_TYPES)}
IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class AccidentDataset(Dataset):
    """
    Dataset for the synthetic accident prediction data.

    Each sample returns:
        frames  : (T, 3, 224, 224) float tensor  — T uniformly-sampled frames
        targets : dict with keys
            'accident_time'  : scalar float in [0, 1]  (accident_frame / no_frames)
            'location'       : (2,) float tensor        [center_x, center_y], both in [0, 1]
            'type'           : scalar long              collision-type class index
    """

    def __init__(self, df: pd.DataFrame, video_root: str,
                 num_frames: int = 16, transform=None, aug_transform=None,
                 aug_prob: float = 0.5):
        """
        Args:
            df            : DataFrame with columns from labels.csv (already filtered/split).
            video_root    : Root directory that prefixes the 'rgb_path' column values.
            num_frames    : Number of frames to uniformly sample from each video.
            transform     : Base transform applied per frame (no augmentation).
            aug_transform : Augmented transform applied to the whole clip with probability aug_prob.
            aug_prob      : Probability of applying aug_transform to a clip (default 0.5).
        """
        self.df = df.reset_index(drop=True)
        self.video_root = video_root
        self.num_frames = num_frames
        self.transform = transform or DEFAULT_TRANSFORM
        self.aug_transform = aug_transform
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, video_path: str, total_frames: int,
                       transform) -> torch.Tensor:
        """Load `num_frames` uniformly-spaced frames from *video_path*."""
        cap = cv2.VideoCapture(video_path)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        last_valid = torch.zeros(3, 224, 224)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_valid = transform(frame)
            frames.append(last_valid)

        cap.release()
        return torch.stack(frames)  # (T, 3, H, W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_root, row["rgb_path"])
        total_frames = int(row["no_frames"])

        # Decide once per clip whether to augment
        if self.aug_transform is not None and random.random() < self.aug_prob:
            clip_transform = self.aug_transform
        else:
            clip_transform = self.transform

        frames = self._sample_frames(video_path, total_frames, clip_transform)

        targets = {
            "accident_time": torch.tensor(
                float(row["accident_frame"]) / total_frames, dtype=torch.float32
            ),
            "location": torch.tensor(
                [float(row["center_x"]), float(row["center_y"])], dtype=torch.float32
            ),
            "type": torch.tensor(TYPE_TO_IDX[row["type"]], dtype=torch.long),
        }
        return frames, targets


class TestDataset(Dataset):
    """
    Dataset for the real-world test videos (no labels).

    Each sample returns:
        frames    : (T, 3, 224, 224) float tensor
        video_id  : str — the video path (used to identify the sample in submission)
    """

    def __init__(self, metadata_csv: str, video_root: str,
                 num_frames: int = 16, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.video_root = video_root
        self.num_frames = num_frames
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, video_path: str, total_frames: int) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        last_valid = torch.zeros(3, 224, 224)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_valid = self.transform(frame)
            frames.append(last_valid)

        cap.release()
        return torch.stack(frames)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_root, row["path"])
        total_frames = int(row["no_frames"])
        frames = self._sample_frames(video_path, total_frames)  # no augmentation for test
        return frames, row["path"]


def get_dataloaders(
    labels_csv: str,
    video_root: str,
    num_frames: int = 16,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    transform=None,
    augmentation: str | None = None,
    max_samples: int = None,
):
    """
    Build train and validation DataLoaders from *labels_csv*.

    Args:
        max_samples: If set, randomly subsample this many rows before splitting.
                     Useful for quick smoke-tests (e.g. max_samples=32).

    Returns:
        train_loader, val_loader
    """
    df = pd.read_csv(labels_csv)
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed).reset_index(drop=True)
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=df["type"]
    )

    aug_transform = None if (transform or augmentation is None) else build_train_transform(augmentation)
    base_transform = transform or DEFAULT_TRANSFORM

    train_ds = AccidentDataset(train_df, video_root, num_frames,
                               transform=base_transform, aug_transform=aug_transform)
    val_ds = AccidentDataset(val_df, video_root, num_frames,
                             transform=base_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
