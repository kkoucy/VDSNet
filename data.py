from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _image_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


class PairedImageDataset(Dataset):
    """Paired RGB images with identical relative paths in two directories."""

    def __init__(self, input_dir, target_dir, crop_size=None, augment=False, repeat=1):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.repeat = max(1, int(repeat))
        self.pairs = []
        for input_path in _image_files(self.input_dir):
            relative = input_path.relative_to(self.input_dir)
            target_path = self.target_dir / relative
            if not target_path.is_file():
                raise FileNotFoundError(f"Missing target for {input_path}: {target_path}")
            self.pairs.append((input_path, target_path))
        if not self.pairs:
            raise RuntimeError(f"No images found in {self.input_dir}")

    def __len__(self):
        return len(self.pairs) * self.repeat

    @staticmethod
    def _tensor(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index % len(self.pairs)]
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        if input_image.size != target_image.size:
            raise ValueError(
                f"Pair size mismatch: {input_path} {input_image.size} vs "
                f"{target_path} {target_image.size}"
            )

        if self.crop_size is not None:
            crop = int(self.crop_size)
            width, height = input_image.size
            if width < crop or height < crop:
                new_width, new_height = max(width, crop), max(height, crop)
                input_image = input_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
                target_image = target_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
                width, height = new_width, new_height
            left = random.randint(0, width - crop)
            top = random.randint(0, height - crop)
            box = (left, top, left + crop, top + crop)
            input_image, target_image = input_image.crop(box), target_image.crop(box)

        if self.augment:
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target_image = target_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                input_image = input_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                target_image = target_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            if rotations:
                angle = 90 * rotations
                input_image = input_image.rotate(angle, expand=True)
                target_image = target_image.rotate(angle, expand=True)

        return {
            "input": self._tensor(input_image),
            "target": self._tensor(target_image),
            "name": str(input_path.relative_to(self.input_dir)),
        }

