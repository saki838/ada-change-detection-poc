"""
Stage 1 — Change detection model wrappers.

All wrappers implement the same interface:

    detector = SomeChangeDetector(...)
    mask = detector.predict(t1_rgb, t2_rgb)   # -> (H, W) uint8, {0,1}

so Stage 1 (and the pipeline) never needs to know which concrete model is
behind it. Swap `model:` in config.yaml to switch.
"""
from __future__ import annotations

import abc
import sys
import pathlib

import numpy as np

from src.utils.safe_load import load_state_dict


class BaseChangeDetector(abc.ABC):
    @abc.abstractmethod
    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
        """Return a binary (H, W) change mask, values in {0, 1}."""


class DummyDiffChangeDetector(BaseChangeDetector):
    """
    No-weights, no-GPU stand-in for BIT, used to prove the pipeline plumbing
    end-to-end (Stage 0 -> 1 -> 2 -> ...) before real weights are wired in,
    or in this sandbox where no GPU / weight host is reachable.

    NOT a substitute for BIT's accuracy — swap to `bit` in config once you
    have weights, per Section 2/6 of the POC doc (F1 >= 0.85 requirement).
    """

    def __init__(self, threshold: float = 30.0, min_blur: int = 3):
        self.threshold = threshold
        self.min_blur = min_blur

    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
        import cv2

        g1 = cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(t2_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g1 = cv2.GaussianBlur(g1, (self.min_blur, self.min_blur), 0)
        g2 = cv2.GaussianBlur(g2, (self.min_blur, self.min_blur), 0)
        diff = np.abs(g1 - g2)
        mask = (diff > self.threshold).astype(np.uint8)
        return mask


class BITChangeDetector(BaseChangeDetector):
    """
    Wrapper around the official BIT (Bitemporal Image Transformer) model
    from https://github.com/justchenhao/BIT_CD.

    Setup (on the workstation, not this sandbox):
        git clone https://github.com/justchenhao/BIT_CD.git external/BIT_CD
        # download the pretrained checkpoint into weights/bit_levir.pt
        # (see scripts/download_weights.sh)

    This wrapper tiles the input if it's larger than `input_size`, runs each
    tile through the model, and stitches results back with overlap blending,
    since BIT expects fixed-size inputs (default 256x256).
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        input_size: int = 256,
        tile_overlap: int = 32,
        binarize_threshold: float = 0.5,
        bit_repo_path: str = "external/BIT_CD",
    ):
        import torch

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            print(f"[BITChangeDetector] CUDA not available — falling back to "
                  f"CPU (requested device={device!r}). BIT is lightweight so "
                  f"this is fine for a handful of POC tiles.")
        self.input_size = input_size
        self.tile_overlap = tile_overlap
        self.binarize_threshold = binarize_threshold

        repo = pathlib.Path(bit_repo_path)
        if not repo.exists():
            raise RuntimeError(
                f"BIT_CD repo not found at {repo}. Clone it first:\n"
                f"  git clone https://github.com/justchenhao/BIT_CD.git {repo}\n"
                "This must be done on a machine with internet access "
                "(not this sandbox)."
            )
        sys.path.insert(0, str(repo))
        from models.basic_model import CDEvaluator  # type: ignore  # from BIT_CD repo

        self.model = CDEvaluator(args=self._build_args()).net_G
        state_dict = load_state_dict(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def _build_args(self):
        # Matches what BIT_CD's CDEvaluator.__init__ / define_G actually read
        # (models/basic_model.py, models/networks.py) — n_class, net_G,
        # gpu_ids (must be a *list*, not a string — init_net does
        # `net.to(gpu_ids[0])` / `len(gpu_ids) > 0`), checkpoint_dir and
        # output_folder (unused by us since we load weights ourselves below,
        # but __init__ still touches them, including creating output_folder).
        class Args:
            # NOTE: must match the architecture the checkpoint was actually
            # trained with. BIT_CD's own eval.sh (which is what the
            # BIT_LEVIR / best_ckpt.pt release corresponds to) uses the
            # "_dedim8" variant (decoder_dim_head=8, giving an inner
            # attention dim of 8*8=64), NOT the plain "base_transformer_pos_s4_dd8"
            # used in their train script (decoder_dim_head=64, inner dim
            # 8*64=512) — using the wrong one causes a tensor size mismatch
            # when loading the checkpoint (64 vs 512 in to_q/to_k/to_v/to_out).
            net_G = "base_transformer_pos_s4_dd8_dedim8"
            n_class = 2
            gpu_ids = [0] if self.device == "cuda" else []
            checkpoint_dir = "weights"
            output_folder = "outputs/_bit_tmp"

        return Args()

    def _tile_bounds(self, h: int, w: int):
        step = self.input_size - self.tile_overlap
        ys = list(range(0, max(h - self.input_size, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - self.input_size, 0) + 1, step)) or [0]
        if ys[-1] + self.input_size < h:
            ys.append(h - self.input_size)
        if xs[-1] + self.input_size < w:
            xs.append(w - self.input_size)
        return ys, xs

    def predict(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = t1_rgb.shape[:2]
        prob_sum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        ys, xs = self._tile_bounds(h, w)
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    t1_tile = t1_rgb[y : y + self.input_size, x : x + self.input_size]
                    t2_tile = t2_rgb[y : y + self.input_size, x : x + self.input_size]
                    t1_t = self._to_tensor(t1_tile)
                    t2_t = self._to_tensor(t2_tile)
                    out = self.model(t1_t, t2_t)
                    # BASE_Transformer.forward() returns a single tensor
                    # directly (batch, n_class, H, W) — NOT a list of
                    # multi-stage outputs, confirmed against BIT_CD's own
                    # basic_model.py (`self.G_pred = self.net_G(...)`, used
                    # directly, no indexing). Indexing out[-1] here was wrong
                    # (it silently sliced the batch dim instead).
                    prob = torch.softmax(out, dim=1)[:, 1].squeeze(0).cpu().numpy()
                    prob_sum[y : y + self.input_size, x : x + self.input_size] += prob
                    weight[y : y + self.input_size, x : x + self.input_size] += 1.0

        weight[weight == 0] = 1.0
        avg_prob = prob_sum / weight
        return (avg_prob >= self.binarize_threshold).astype(np.uint8)

    def _to_tensor(self, img_rgb: np.ndarray):
        torch = self.torch
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)


def build_change_detector(cfg: dict) -> BaseChangeDetector:
    """Factory used by the pipeline — picks the detector named in config.yaml."""
    model = cfg.get("model", "dummy_diff")
    if model == "dummy_diff":
        return DummyDiffChangeDetector()
    if model == "bit":
        return BITChangeDetector(
            weights_path=cfg["weights_path"],
            device=cfg.get("device", "cuda"),
            input_size=cfg.get("input_size", 256),
            tile_overlap=cfg.get("tile_overlap", 32),
            binarize_threshold=cfg.get("binarize_threshold", 0.5),
        )
    if model == "changeformer":
        raise NotImplementedError(
            "ChangeFormer wrapper not wired yet — it's the higher-accuracy "
            "alternative per the doc if BIT's F1 falls short. Follow the "
            "same pattern as BITChangeDetector against "
            "https://github.com/wgcban/ChangeFormer when needed."
        )
    raise ValueError(f"Unknown change detection model: {model}")
