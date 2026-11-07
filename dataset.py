"""Manifest-based polyp dataset and the preprocessing described in the paper."""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "dataset", "group", "image", "mask"}
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_id values in {path}")
    return rows


def preprocess_rgb_image(
    image: Image.Image,
    size: int = 256,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)[:, None, None]
    std_array = np.asarray(std, dtype=np.float32)[:, None, None]
    if np.any(std_array == 0):
        raise ValueError("Normalization standard deviation must be non-zero")
    return torch.from_numpy(((array - mean_array) / std_array).copy())


def preprocess_binary_mask(mask: Image.Image, size: int = 256) -> torch.Tensor:
    mask = mask.convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.uint8)
    return torch.from_numpy((array >= 128).astype(np.float32, copy=False)[None].copy())


class ManifestDataset(Dataset):
    """Load image/mask pairs from an auditable CSV split manifest.

    Training augmentation follows the manuscript exactly: horizontal and
    vertical flips are independent Bernoulli(0.5) operations; with probability
    0.5, one of 90, 180 or 270 degrees is sampled uniformly. Evaluation is
    deterministic and performs no stochastic augmentation.
    """

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        size: int = 256,
        train: bool = False,
        limit: Optional[int] = None,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.rows = read_manifest(self.manifest)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.rows = self.rows[:limit]
        self.size = int(size)
        self.train = bool(train)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = 90 * random.randint(1, 3)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        return image, mask

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = _resolve_path(row["image"], self.data_root)
        mask_path = _resolve_path(row["mask"], self.data_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for {row['sample_id']}: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for {row['sample_id']}: {mask_path}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(mask_path) as handle:
            mask = handle.convert("L")
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {row['sample_id']}: "
                f"{image.size} vs {mask.size}"
            )
        if self.train:
            image, mask = self._augment(image, mask)
        image_tensor = preprocess_rgb_image(image, self.size, self.mean, self.std)
        mask_tensor = preprocess_binary_mask(mask, self.size)
        return image_tensor, mask_tensor, row["sample_id"]


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
