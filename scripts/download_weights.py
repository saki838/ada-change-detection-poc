"""
Prints setup instructions for Stage 1 (BIT) and Stage 3 (SAM 2.1 / RT-DETR)
pretrained weights. This just prints commands — it doesn't run anything
itself — so it's plain Python rather than bash for Windows compatibility.

Usage:
    python scripts/download_weights.py
"""
import pathlib

WEIGHTS_DIR = pathlib.Path("weights")


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)

    print("""
1) BIT (change detection) — clone the repo, then get weights:
   git clone https://github.com/justchenhao/BIT_CD.git external/BIT_CD
   Pretrained LEVIR-CD checkpoint is linked from the repo's README
   (Google Drive). Save it as: weights/bit_levir.pt

2) SAM 2.1 (segmentation):
   git clone https://github.com/facebookresearch/sam2.git
   pip install -e sam2
   Download the tiny checkpoint directly (this URL works in a browser
   or via curl/wget/Invoke-WebRequest):
     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
   Save it as: weights/sam2.1_tiny.pt

   PowerShell equivalent of wget:
     Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" -OutFile "weights/sam2.1_tiny.pt"

3) RT-DETR (classification) — base weights come from Hugging Face
   `transformers` automatically (PekingU/rtdetr_r50vd_coco_o365).
   You still need to fine-tune the classification head on your own
   violation classes (illegal_build, encroachment, ...) once labeled
   site regions exist, then save that fine-tuned state_dict as:
     weights/rtdetr_violations.pt

Loading discipline reminder (POC doc, Section 2): every checkpoint loaded
by this repo goes through src/utils/safe_load.py, which prefers
.safetensors and otherwise forces torch.load(..., weights_only=True).
No arbitrary pickle execution from downloaded checkpoints.
""")


if __name__ == "__main__":
    main()
