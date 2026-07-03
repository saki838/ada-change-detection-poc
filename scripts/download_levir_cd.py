"""
Downloads the LEVIR-CD+ public change-detection dataset (aerial building
change pairs) used as the bridge dataset per Section 2 of the POC doc:

    "Bridge while site data is annotated: start on public pairs — LEVIR-CD
     and WHU-CD (aerial building change) — to stand the pipeline up
     immediately, then swap in site data for fine-tuning."

Source: https://huggingface.co/datasets/blanchon/LEVIR_CDPlus
(LEVIR-CD+, the extended version — 985 image pairs, 1024x1024, 0.5m/px,
20 regions in Texas. This is a *dataset-format* HF repo — image1/image2/mask
columns in parquet — so it's downloaded with `datasets.load_dataset` and
exported to PNGs, not fetched as raw files.)

Cross-platform: pure Python, works identically on Windows/macOS/Linux.
No bash required.

Usage:
    pip install datasets pillow
    python scripts/download_levir_cd.py
"""
from __future__ import annotations

import pathlib

DEST = pathlib.Path("data/levir_cd")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency. Run: pip install datasets pillow")
        return

    print("Downloading blanchon/LEVIR_CDPlus from Hugging Face "
          "(this dataset only ships train/test splits, no val — "
          "we carve 10% of train off as val below)...")
    ds = load_dataset("blanchon/LEVIR_CDPlus")

    train_val = ds["train"].train_test_split(test_size=0.1, seed=0)
    splits = {
        "train": train_val["train"],
        "val": train_val["test"],
        "test": ds["test"],
    }

    for split_name, split_ds in splits.items():
        a_dir = DEST / split_name / "A"
        b_dir = DEST / split_name / "B"
        label_dir = DEST / split_name / "label"
        for d in (a_dir, b_dir, label_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"Exporting {split_name}: {len(split_ds)} pairs -> {DEST / split_name}")
        for i, row in enumerate(split_ds):
            row["image1"].save(a_dir / f"{i:05d}.png")
            row["image2"].save(b_dir / f"{i:05d}.png")
            row["mask"].save(label_dir / f"{i:05d}.png")

    print(f"\nDone. Layout now matches what BIT training expects:")
    print(f"  {DEST}/train/A|B|label/*.png")
    print(f"  {DEST}/val/A|B|label/*.png")
    print(f"  {DEST}/test/A|B|label/*.png")
    print("\nIf this Hugging Face mirror ever goes stale, fall back to the "
          "official page for manual download: https://justchenhao.github.io/LEVIR/")


if __name__ == "__main__":
    main()
