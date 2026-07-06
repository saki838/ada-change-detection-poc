"""Bitemporal Siamese U-Net (FC-Siam-diff style) + safe weight loader.

This single module serves BOTH:
  * the TRAINING side (§20): ``SiameseUNet``, ``build_model()``, ``load_weights()``
    imported by ``inference/train.py``.
  * the SERVING side (§30): ``get_model()`` cached singleton + ``_load_weights_safe()``
    imported by ``inference/app/main.py`` / ``inference/app/predict.py``.

Architecture: a single shared-weight ``smp.Unet`` ResNet34 encoder is run on T1 and
T2 separately; the per-skip-level feature maps are fused by absolute difference
``|f(T1) - f(T2)|`` (channel preserving, so the stock smp decoder consumes them
unchanged), then decoded to a single change-logit channel ``(B, 1, H, W)``. Sigmoid
is applied downstream in ``predict.py`` — this module always returns raw logits.

Non-China posture: the ONLY pretrained weight is torchvision ResNet34 IMAGENET1K_V1
(``resnet34-b627a593.pth``, BSD-3-Clause, download.pytorch.org). If a local copy is
present at ``app/weights/resnet34-b627a593.pth`` it is loaded offline; otherwise smp
resolves it via torchvision. The locally-trained change head loads with SAFE
semantics only (safetensors, or ``torch.load(..., weights_only=True)``) — never an
executable pickle.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

logger = logging.getLogger("inference.model")

DEVICE = "cpu"
# Env MODEL_PATH; compose sets /app/models/change_unet_resnet34.pt.
MODEL_WEIGHTS_PATH = os.getenv("MODEL_PATH", "models/change_unet_resnet34.pt")

# Offline, non-China torchvision ResNet34 ImageNet encoder weights, pre-fetched
# into the image at build time. Relative to this file so it works from any CWD.
_LOCAL_ENCODER_WEIGHTS = Path(__file__).resolve().parent / "weights" / "resnet34-b627a593.pth"

ENCODER_NAME = "resnet34"


class SiameseUNet(nn.Module):
    """Shared-weight ResNet34 Siamese encoder + abs-diff fusion + smp U-Net decoder."""

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        # Decide whether smp should try to fetch ImageNet weights. If we have a
        # local offline copy we build the encoder cold (encoder_weights=None) and
        # inject the local state dict ourselves — avoids any network access.
        want_imagenet = encoder_weights == "imagenet"
        smp_encoder_weights = None if (want_imagenet and _LOCAL_ENCODER_WEIGHTS.exists()) else encoder_weights

        try:
            self.net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=smp_encoder_weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as exc:  # offline & no local weights, or hub failure
            logger.warning("smp.Unet(encoder_weights=%s) failed (%s); building cold encoder", smp_encoder_weights, exc)
            self.net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )

        if want_imagenet and _LOCAL_ENCODER_WEIGHTS.exists():
            self._load_local_encoder_weights(_LOCAL_ENCODER_WEIGHTS)

        # Expose the smp sub-modules under the names referenced across the guide.
        self.encoder = self.net.encoder
        self.decoder = self.net.decoder
        self.segmentation_head = self.net.segmentation_head

    def _load_local_encoder_weights(self, path: Path) -> None:
        """Load offline torchvision ResNet34 ImageNet weights into the smp encoder.

        smp's ResNetEncoder subclasses torchvision ResNet, so the torchvision
        state dict maps directly (``fc.*`` keys are simply unused). weights_only
        blocks pickle RCE on this foreign file.
        """
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            # NOTE: smp's ResNetEncoder overrides load_state_dict to strip the
            # torchvision ``fc.*`` classifier keys and returns None (not the
            # standard (missing, unexpected) tuple), so we must NOT unpack the
            # result. Compute the diff explicitly instead for logging.
            enc = self.net.encoder
            enc_keys = set(enc.state_dict().keys())
            src_keys = set(state.keys()) - {"fc.weight", "fc.bias"}
            enc.load_state_dict(state, strict=False)
            missing = enc_keys - src_keys
            unexpected = src_keys - enc_keys
            logger.info(
                "Loaded local encoder weights %s (missing=%d unexpected=%d)",
                path, len(missing), len(unexpected),
            )
        except Exception as exc:
            logger.warning("Could not load local encoder weights %s: %s", path, exc)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """t1,t2: (B,3,H,W) ImageNet-normalized -> (B,1,H,W) raw change logits."""
        feats_t1 = self.encoder(t1)
        feats_t2 = self.encoder(t2)
        # Abs-diff fuse each skip level (channel preserving).
        fused = [torch.abs(a - b) for a, b in zip(feats_t1, feats_t2)]

        # smp decoder signature differs across versions: some take a list, some
        # take *features. Support both.
        try:
            decoder_output = self.decoder(fused)
        except TypeError:
            decoder_output = self.decoder(*fused)

        logits = self.segmentation_head(decoder_output)
        return logits


def build_model(device: str = "cpu", pretrained_encoder: bool = False) -> SiameseUNet:
    """Factory used by training (pretrained_encoder=True) and serving.

    In production the encoder comes bundled inside the loaded checkpoint, so
    ``pretrained_encoder=False`` builds a cold encoder that the head checkpoint
    then fills in. Returns the model on ``device`` in eval() mode.
    """
    encoder_weights = "imagenet" if pretrained_encoder else None
    model = SiameseUNet(encoder_name=ENCODER_NAME, encoder_weights=encoder_weights)
    model.to(device)
    model.eval()
    return model


def _safe_load_state_dict(path: str) -> dict:
    """Return a plain state-dict from ``path`` using SAFE semantics only.

    Prefers a ``.safetensors`` sibling; else loads the ``.pt`` with
    ``weights_only=True``. Never executes pickle.
    """
    p = Path(path)
    sibling = p.with_suffix(".safetensors")
    if sibling.exists():
        from safetensors.torch import load_file

        logger.info("Loading change head via safetensors: %s", sibling)
        return load_file(str(sibling))
    if p.exists():
        logger.info("Loading change head via torch.load(weights_only=True): %s", p)
        return torch.load(str(p), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No weights found at {sibling} or {p}")


def load_weights(model: SiameseUNet, path: str = MODEL_WEIGHTS_PATH) -> SiameseUNet:
    """SECURITY-CRITICAL. Safe-load head+encoder state dict and apply (strict=False)."""
    state = _safe_load_state_dict(path)
    model.load_state_dict(state, strict=False)
    return model


def _load_weights_safe(model: SiameseUNet, path: str) -> None:
    """Serving-side safe loader. Raises RuntimeError if the file is absent."""
    try:
        state = _safe_load_state_dict(path)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded change head from %s (missing=%d unexpected=%d)",
        path, len(missing), len(unexpected),
    )


@functools.lru_cache(maxsize=1)
def get_model() -> SiameseUNet:
    """Cached serving singleton: build, safe-load head, CPU eval, grad disabled.

    If the head checkpoint is missing, the model still boots with an UNTRAINED
    (random) head so the service and ``diff`` mode stay available; ``ml`` mode
    then produces meaningless masks but does not crash. The load failure is
    logged so ``/health`` can report the situation.
    """
    model = build_model(device=DEVICE, pretrained_encoder=True)
    try:
        _load_weights_safe(model, MODEL_WEIGHTS_PATH)
        get_model.weights_loaded = True
    except RuntimeError as exc:
        logger.warning("Change head weights not loaded (%s); using untrained head", exc)
        get_model.weights_loaded = False
    model.to(DEVICE)
    model.eval()
    torch.set_grad_enabled(False)
    return model


# Attribute default so /health can read it before the first get_model() call.
get_model.weights_loaded = False
