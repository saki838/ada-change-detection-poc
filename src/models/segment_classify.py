"""
Stage 3 — Segmentation (SAM 2.1) + classification (RT-DETR) wrappers.

Interface every segmenter/classifier pair implements:

    seg = SomeSegmenter(...)
    polygon = seg.segment(image_rgb, box)       # -> (N, 2) pixel polygon

    clf = SomeClassifier(...)
    class_name, score = clf.classify(image_rgb, box)
"""
from __future__ import annotations

import abc
import pathlib

import numpy as np

from src.utils.safe_load import load_state_dict


# ---------------------------------------------------------------- segmenter --

class BaseSegmenter(abc.ABC):
    @abc.abstractmethod
    def segment(self, image_rgb: np.ndarray, box_xywh: tuple) -> np.ndarray:
        """Return an (N, 2) array of (col, row) polygon vertices for the box prompt."""


class DummyContourSegmenter(BaseSegmenter):
    """
    No-weights stand-in for SAM 2.1: just returns the box itself as a
    rectangle polygon. Proves the Stage 2 -> 3 -> 4 handoff shape is correct;
    swap to `sam2` for real precise polygon boundaries.
    """

    def segment(self, image_rgb: np.ndarray, box_xywh: tuple) -> np.ndarray:
        x, y, w, h = box_xywh
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


class SAM2Segmenter(BaseSegmenter):
    """
    Wrapper around SAM 2.1 (https://github.com/facebookresearch/sam2).

    Setup (on the workstation):
        git clone https://github.com/facebookresearch/sam2.git
        pip install -e sam2
        # download sam2.1_hiera_tiny.pt into weights/ (see download_weights.sh)
    """

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cuda"):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise RuntimeError(
                "sam2 package not installed. On the workstation: "
                "git clone https://github.com/facebookresearch/sam2.git && "
                "pip install -e sam2"
            ) from e

        if not pathlib.Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {checkpoint}. "
                "Run scripts/download_weights.sh on the workstation first."
            )

        # Same auto-fallback as BITChangeDetector / RTDETRClassifier: honor
        # an explicit "cpu" request, but don't hard-fail if "cuda" was left
        # as the config default on a machine with no GPU.
        resolved_device = device if torch.cuda.is_available() else "cpu"
        if resolved_device != device:
            print(f"[SAM2Segmenter] CUDA not available — falling back to CPU "
                  f"(requested device={device!r}). This will be noticeably "
                  f"slower than GPU; expect several seconds per region.")
        self.device = resolved_device

        sam_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self._current_image_id = None

    def _ensure_image(self, image_rgb: np.ndarray):
        # Cheap identity check to avoid re-encoding the same image per box.
        image_id = id(image_rgb)
        if image_id != self._current_image_id:
            self.predictor.set_image(image_rgb)
            self._current_image_id = image_id

    def segment(self, image_rgb: np.ndarray, box_xywh: tuple) -> np.ndarray:
        import cv2

        self._ensure_image(image_rgb)
        x, y, w, h = box_xywh
        box_xyxy = np.array([x, y, x + w, y + h])
        masks, scores, _ = self.predictor.predict(box=box_xyxy, multimask_output=False)
        mask = masks[0].astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # SAM returned an empty mask for this box — fall back to the box itself.
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2)


def build_segmenter(cfg: dict) -> BaseSegmenter:
    kind = cfg.get("segmenter", "dummy_contour")
    if kind == "dummy_contour":
        return DummyContourSegmenter()
    if kind == "sam2":
        return SAM2Segmenter(
            checkpoint=cfg["sam_checkpoint"],
            model_cfg=cfg["sam_model_cfg"],
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown segmenter: {kind}")


# --------------------------------------------------------------- classifier --

class BaseClassifier(abc.ABC):
    @abc.abstractmethod
    def classify(self, image_rgb: np.ndarray, box_xywh: tuple) -> tuple[str, float]:
        """Return (class_name, confidence) for the crop inside box_xywh."""


class DummyRuleClassifier(BaseClassifier):
    """
    No-weights stand-in: labels everything "change" (per the doc's suggestion
    that classification is optional in v1 — "can label all regions 'change'
    first"). Swap to `rtdetr` once fine-tuned weights exist.
    """

    def classify(self, image_rgb: np.ndarray, box_xywh: tuple) -> tuple[str, float]:
        return "change", 1.0


class RTDETRClassifier(BaseClassifier):
    """
    Wrapper around RT-DETR via Hugging Face `transformers`
    (RTDetrForObjectDetection), fine-tuned on violation classes.

    Setup (on the workstation):
        pip install transformers
        # fine-tune or download a checkpoint into weights/rtdetr_violations.pt
    """

    def __init__(self, weights_path: str, class_names: list[str], device: str = "cuda"):
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device != device:
            print(f"[RTDETRClassifier] CUDA not available — falling back to "
                  f"CPU (requested device={device!r}).")
        self.class_names = class_names

        self.processor = RTDetrImageProcessor.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        )
        self.model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365",
            num_labels=len(class_names),
            ignore_mismatched_sizes=True,
        )
        state_dict = load_state_dict(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

    def classify(self, image_rgb: np.ndarray, box_xywh: tuple) -> tuple[str, float]:
        x, y, w, h = box_xywh
        crop = image_rgb[y : y + h, x : x + w]
        if crop.size == 0:
            return "other_change", 0.0

        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = logits.softmax(-1)
        best_idx = int(probs[:, :-1].max(0).values.argmax())  # best box, excluding "no object"
        cls_id = int(probs[best_idx, :-1].argmax())
        score = float(probs[best_idx, cls_id])
        class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "other_change"
        return class_name, score


def build_classifier(cfg: dict) -> BaseClassifier:
    kind = cfg.get("classifier", "dummy_rule")
    if kind == "dummy_rule":
        return DummyRuleClassifier()
    if kind == "rtdetr":
        return RTDETRClassifier(
            weights_path=cfg["classifier_weights"],
            class_names=cfg["class_names"],
            device=cfg.get("device", "cuda"),
        )
    raise ValueError(f"Unknown classifier: {kind}")
