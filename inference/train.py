"""Train the change head on OSCD -> models/change_unet_resnet34.pt (safetensors).

Starts from the ImageNet-pretrained ResNet34 encoder (non-China, torchvision) and
trains the abs-diff fusion decoder + segmentation head on OSCD RGB 256px tiles.
The checkpoint is serialized with ``safetensors.torch.save_file`` — NEVER
``torch.save`` (pickle) — which is the load-bearing security decision. The ``.pt``
filename is kept for continuity but the bytes are safetensors.

Run from the ``inference/`` directory so ``from app.model import ...`` and
``from dataset import ...`` resolve:

    python train.py --data data/oscd --epochs 20 --batch-size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from app.model import SiameseUNet, build_model
from dataset import OSCDPatchDataset


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on sigmoid(logits) vs binary target."""
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    inter = (probs * tgt).sum(dim=1)
    union = probs.sum(dim=1) + tgt.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(logits, target, bce_w: float = 0.5, dice_w: float = 0.5) -> torch.Tensor:
    """bce_w * BCEWithLogitsLoss + dice_w * dice_loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return bce_w * bce + dice_w * dice_loss(logits, target)


def compute_metrics(logits, target, threshold: float = 0.5) -> dict:
    """Pixel-level precision/recall/F1/IoU on binarized predictions."""
    pred = (torch.sigmoid(logits) >= threshold).float()
    tgt = (target >= 0.5).float()
    tp = (pred * tgt).sum().item()
    fp = (pred * (1 - tgt)).sum().item()
    fn = ((1 - pred) * tgt).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall}


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses = []
    for t1, t2, mask in loader:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(t1, t2)
        loss = combined_loss(logits, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    agg = {"f1": [], "iou": [], "precision": [], "recall": []}
    for t1, t2, mask in loader:
        t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
        logits = model(t1, t2)
        m = compute_metrics(logits, mask)
        for k, v in m.items():
            agg[k].append(v)
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in agg.items()}


def save_checkpoint_safetensors(model: nn.Module, path: str = "models/change_unet_resnet34.pt") -> None:
    """Serialize model.state_dict() via safetensors — NEVER torch.save (pickle)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous CPU tensors.
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, path)
    print(f"Saved safetensors checkpoint -> {path}")


def _ensure_processed(data_root: str, processed_root: str, tile: int) -> None:
    manifest = Path(processed_root) / "manifest.json"
    if manifest.exists():
        return
    print("Processed patches missing; running prepare_data.py ...")
    import sys

    import prepare_data

    argv = sys.argv
    sys.argv = [
        "prepare_data.py",
        "--oscd-root", data_root,
        "--out", processed_root,
        "--size", str(tile),
        "--stride", str(tile),
    ]
    try:
        prepare_data.main()
    finally:
        sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser(description="Train OSCD change head")
    ap.add_argument("--data", default="data/oscd")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--out", default="models/change_unet_resnet34.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    _ensure_processed(args.data, args.processed, args.tile)

    train_ds = OSCDPatchDataset(args.processed, split="train", augment=True)
    val_ds = OSCDPatchDataset(args.processed, split="val", augment=False)
    print(f"train patches={len(train_ds)} val patches={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # pretrained_encoder=True -> ImageNet encoder; head/decoder trained here.
    model: SiameseUNet = build_model(device=device, pretrained_encoder=True)
    model.train()
    torch.set_grad_enabled(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = validate(model, val_loader, device) if len(val_ds) else {"f1": 0.0, "iou": 0.0}
        print(f"epoch {epoch:03d} loss={train_loss:.4f} "
              f"f1={metrics['f1']:.4f} iou={metrics['iou']:.4f}")
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint_safetensors(model, args.out)

    if best_f1 < 0:  # no val split — still emit a checkpoint
        save_checkpoint_safetensors(model, args.out)
    print(f"best val F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
