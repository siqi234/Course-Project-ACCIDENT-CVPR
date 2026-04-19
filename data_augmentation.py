import random
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class MotionBlurTransform:
    """Apply random directional motion blur to a PIL image (always executes — clip-level p is handled by the dataset)."""

    def __init__(self, max_kernel_size: int = 15):
        self.max_kernel_size = max_kernel_size

    @staticmethod
    def _make_kernel(size: int, angle: float) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        rad = np.deg2rad(angle)
        for i in range(size):
            offset = i - center
            x = center + int(round(offset * np.cos(rad)))
            y = center + int(round(offset * np.sin(rad)))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        total = kernel.sum()
        if total > 0:
            kernel /= total
        return kernel

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.choice(range(3, self.max_kernel_size + 1, 2))
        angle = random.uniform(0, 360)
        kernel = self._make_kernel(k, angle)
        img_np = np.array(img)
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)


class ResolutionDegradationTransform:
    """Downsample then upsample a PIL image (always executes — clip-level p is handled by the dataset)."""

    def __init__(self, scale_range: tuple = (0.25, 0.5)):
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        low_w = max(1, int(w * scale))
        low_h = max(1, int(h * scale))
        img = img.resize((low_w, low_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


def build_train_transform(augmentation: str | None) -> transforms.Compose:
    """
    Build a training transform with the requested augmentation.

    Args:
        augmentation: one of None, "motion_blur", "resolution"
    """
    aug_layers = []
    if augmentation == "motion_blur":
        aug_layers = [MotionBlurTransform(max_kernel_size=15, p=0.5)]
    elif augmentation == "resolution":
        aug_layers = [ResolutionDegradationTransform(scale_range=(0.25, 0.5), p=0.5)]
    elif augmentation is not None:
        raise ValueError(f"Unknown augmentation '{augmentation}'. Choose 'motion_blur' or 'resolution'.")

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        *aug_layers,
        transforms.ToTensor(),
        _IMAGENET_NORMALIZE,
    ])
