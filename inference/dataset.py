"""OSCDPatchDataset — PyTorch Dataset over tiled OSCD 256px patches.

Consumed by ``inference/train.py``'s DataLoader. Loads a ``(t1, t2, mask)`` patch
triple written by ``prepare_data.py``, ImageNet-normalizes T1/T2, and returns
CHW float tensors. Both T1 and T2 use ImageNet statistics because the encoder is
the ImageNet-pretrained ResNet34 — preprocessing must match ``app/predict.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class OSCDPatchDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split: str = "train",
        augment: bool = False,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.root = Path(processed_root)
        self.split = split
        self.augment = augment and split == "train"
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

        manifest_path = self.root / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        records = manifest["patches"] if isinstance(manifest, dict) else manifest
        self.stems = [r["stem"] for r in records if r.get("split") == split]
        self.split_dir = self.root / split

    def __len__(self) -> int:
        return len(self.stems)

    def _normalize(self, rgb_uint8: np.ndarray) -> np.ndarray:
        arr = rgb_uint8.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr  # HxWx3

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        t1 = np.load(self.split_dir / f"{stem}_t1.npy")
        t2 = np.load(self.split_dir / f"{stem}_t2.npy")
        mask = np.load(self.split_dir / f"{stem}_mask.npy")

        if self.augment:
            t1, t2, mask = self._augment(t1, t2, mask)

        t1n = self._normalize(t1)
        t2n = self._normalize(t2)

        t1t = torch.from_numpy(np.ascontiguousarray(t1n)).permute(2, 0, 1).float()
        t2t = torch.from_numpy(np.ascontiguousarray(t2n)).permute(2, 0, 1).float()
        maskt = torch.from_numpy(np.ascontiguousarray((mask > 0).astype(np.float32))).unsqueeze(0)
        return t1t, t2t, maskt

    @staticmethod
    def _augment(t1: np.ndarray, t2: np.ndarray, mask: np.ndarray):
        """Apply the SAME random hflip/vflip/rot90 to all three (geometry aligned)."""
        if random.random() < 0.5:  # hflip
            t1, t2, mask = t1[:, ::-1], t2[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:  # vflip
            t1, t2, mask = t1[::-1], t2[::-1], mask[::-1]
        k = random.randint(0, 3)  # rot90
        if k:
            t1 = np.rot90(t1, k)
            t2 = np.rot90(t2, k)
            mask = np.rot90(mask, k)
        return t1.copy(), t2.copy(), mask.copy()
