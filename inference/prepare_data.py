"""Tile raw OSCD scenes into 256px .npy patches + manifest.json.

One-shot preprocessing (see BUILD_GUIDE §10). Reads raw OSCD scenes, stacks RGB
bands (B04=red, B03=green, B02=blue), per-acquisition percentile-stretches to
uint8, tiles each aligned (T1, T2, mask) triple into non-overlapping 256x256
patches, applies a deterministic SCENE-level train/val split, and writes patches +
manifest under ``data/processed/``.

Scene roots are resolved by globbing for ``imgs_1``/``imgs_2``/``cm`` rather than
hard-coding the long "Onera Satellite Change Detection dataset - ..." folder names,
so both the Kaggle and IMT mirror layouts work.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import rasterio

try:
    import cv2

    def _read_png(path: Path) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
except Exception:  # pragma: no cover - Pillow fallback
    from PIL import Image

    def _read_png(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L"))


def _find_band(imgs_dir: Path, band: str) -> Path:
    """Locate a band GeoTIFF (e.g. B04) tolerant of naming variants."""
    for pat in (f"*{band}.tif", f"*{band}.tiff", f"*_{band}.tif", f"{band}.tif"):
        hits = sorted(imgs_dir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"band {band} not found under {imgs_dir}")


def read_rgb(scene_imgs_dir: Path) -> np.ndarray:
    """Read B04/B03/B02 -> (H, W, 3) array in R,G,B order (raw reflectance)."""
    bands = []
    for band in ("B04", "B03", "B02"):
        with rasterio.open(_find_band(scene_imgs_dir, band)) as ds:
            bands.append(ds.read(1))
    return np.stack(bands, axis=-1)


def read_mask(scene_label_dir: Path) -> np.ndarray:
    """Load cm/cm.png -> (H, W) uint8 binarized to {0,1} (any >0 -> 1)."""
    cm = scene_label_dir / "cm" / "cm.png"
    if not cm.exists():
        alt = sorted(scene_label_dir.glob("**/cm.png"))
        if not alt:
            raise FileNotFoundError(f"cm.png not found under {scene_label_dir}")
        cm = alt[0]
    arr = _read_png(cm)
    return (arr > 0).astype(np.uint8)


def percentile_stretch(img: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    """Per-channel 2-98 percentile clip -> uint8 [0,255] (independent per image)."""
    out = np.zeros(img.shape, dtype=np.uint8)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        p_lo, p_hi = np.percentile(ch, lo), np.percentile(ch, hi)
        if p_hi <= p_lo:
            p_hi = p_lo + 1.0
        ch = np.clip((ch - p_lo) / (p_hi - p_lo), 0, 1) * 255.0
        out[:, :, c] = ch.astype(np.uint8)
    return out


def tile_pair(t1, t2, mask, size: int = 256, stride: int = 256):
    """Slide a size x size window; drop partial edge tiles. Yields (r,c,t1,t2,mask)."""
    h = min(t1.shape[0], t2.shape[0], mask.shape[0])
    w = min(t1.shape[1], t2.shape[1], mask.shape[1])
    out = []
    for r in range(0, h - size + 1, stride):
        for c in range(0, w - size + 1, stride):
            out.append(
                (
                    r, c,
                    t1[r : r + size, c : c + size],
                    t2[r : r + size, c : c + size],
                    mask[r : r + size, c : c + size],
                )
            )
    return out


def split_scenes(scenes: list[str], val_frac: float = 0.2, seed: int = 42):
    """Scene-level seeded split so no scene leaks across train/val."""
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac))) if shuffled else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def list_scenes(images_root: Path, split_file: str) -> list[str]:
    """Read train.txt/test.txt (comma-separated); fall back to globbing imgs_1 dirs."""
    f = images_root / split_file
    if f.exists():
        text = f.read_text(encoding="utf-8").strip()
        names = [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
        if names:
            return names
    return sorted(p.parent.name for p in images_root.glob("*/imgs_1"))


def _resolve_roots(oscd_root: Path) -> tuple[Path, Path]:
    """Find the images root (has */imgs_1) and labels root (has */cm/cm.png)."""
    images_root = None
    labels_root = None
    for p in oscd_root.rglob("imgs_1"):
        images_root = p.parent.parent
        break
    for p in oscd_root.rglob("cm.png"):
        labels_root = p.parent.parent.parent
        break
    if images_root is None:
        images_root = oscd_root
    if labels_root is None:
        labels_root = oscd_root
    return images_root, labels_root


def save_patch(out_dir: Path, stem: str, t1p, t2p, maskp) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}_t1.npy", t1p.astype(np.uint8))
    np.save(out_dir / f"{stem}_t2.npy", t2p.astype(np.uint8))
    np.save(out_dir / f"{stem}_mask.npy", (maskp > 0).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile OSCD into 256px patches")
    ap.add_argument("--oscd-root", default="data/oscd")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-positive-frac", type=float, default=0.0,
                    help="drop tiles whose change-pixel fraction is below this")
    args = ap.parse_args()

    oscd_root = Path(args.oscd_root)
    out_root = Path(args.out)
    images_root, labels_root = _resolve_roots(oscd_root)

    scenes = list_scenes(images_root, "train.txt")
    scenes = [s for s in scenes if (images_root / s / "imgs_1").exists()]
    if not scenes:
        raise SystemExit(f"No OSCD scenes found under {images_root}")

    train_scenes, val_scenes = split_scenes(scenes, args.val_frac, args.seed)
    split_map = {s: "train" for s in train_scenes}
    split_map.update({s: "val" for s in val_scenes})

    manifest_patches: list[dict] = []
    for scene, split in split_map.items():
        try:
            t1 = percentile_stretch(read_rgb(images_root / scene / "imgs_1"))
            t2 = percentile_stretch(read_rgb(images_root / scene / "imgs_2"))
            mask = read_mask(labels_root / scene)
        except FileNotFoundError as exc:
            print(f"skip {scene}: {exc}")
            continue

        for r, c, t1p, t2p, mp in tile_pair(t1, t2, mask, args.size, args.stride):
            if args.min_positive_frac > 0:
                if (mp > 0).mean() < args.min_positive_frac:
                    continue
            stem = f"{scene}_r{r}_c{c}"
            save_patch(out_root / split, stem, t1p, t2p, mp)
            manifest_patches.append({"stem": stem, "split": split})

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "patches": manifest_patches,
        "meta": {"size": args.size, "stride": args.stride, "seed": args.seed,
                 "val_frac": args.val_frac},
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {len(manifest_patches)} patches to {out_root}")


if __name__ == "__main__":
    main()
