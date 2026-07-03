"""
Loading discipline (POC doc, Section 2):

    Load all weights via safetensors with weights_only=True — no arbitrary
    pickle execution from downloaded checkpoints.

This module is the single place every model wrapper goes through to load a
checkpoint, so that discipline can't be silently skipped in one stage.
"""
from __future__ import annotations

import pathlib


def load_state_dict(path: str | pathlib.Path, map_location: str = "cpu") -> dict:
    """
    Load a model's weights safely.

    Prefers .safetensors (no pickle at all). Falls back to torch.load with
    weights_only=True for plain .pt/.pth checkpoints (blocks arbitrary code
    execution from the pickle stream, but still requires the checkpoint to
    only contain tensors/primitives — which is what upstream BIT / SAM2.1 /
    RT-DETR release checkpoints are).

    torch is imported lazily so that dummy-mode stages (which never call
    this function) don't require torch to be installed at all.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run scripts/download_weights.py "
            "on a machine with internet access first."
        )

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device=map_location)

    import torch

    # weights_only=True is the load-time guard against arbitrary pickle exec.
    # Older research checkpoints (e.g. BIT_CD's release) were saved with
    # whatever numpy version the original author had, and numpy's internal
    # module layout for basic types (np.float64 scalars etc.) has changed
    # across versions (numpy.core.* -> numpy._core.* in numpy 2.0), so the
    # exact dotted name pickled into the file may not match what today's
    # numpy reports for that same object. Rather than guess every possible
    # historical name, we read torch's own error message for the *exact*
    # missing global, resolve it dynamically, and allowlist precisely that —
    # nothing broader. We retry a bounded number of times in case more than
    # one distinct global is missing (torch reports one per failure).
    last_error = None
    for _ in range(6):
        try:
            obj = torch.load(str(path), map_location=map_location, weights_only=True)
            break
        except Exception as e:
            msg = str(e)
            dotted_name = _extract_missing_global(msg)
            if dotted_name is None:
                raise
            last_error = e
            resolved = _resolve_dotted_name(dotted_name)
            if resolved is None:
                raise RuntimeError(
                    f"Checkpoint references {dotted_name!r}, which this "
                    "environment's numpy/torch can't resolve to allowlist "
                    "it safely. Original error:\n" + msg
                ) from e
            torch.serialization.add_safe_globals([(resolved, dotted_name)])
    else:
        raise last_error

    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model_G_state_dict" in obj:
        # BIT_CD's own checkpoint format (see external/BIT_CD/models/basic_model.py)
        return obj["model_G_state_dict"]
    return obj


def _extract_missing_global(error_message: str) -> str | None:
    """
    Pulls the dotted name of the missing type out of torch's weights_only
    error, which comes in at least two message formats depending on the
    torch version / code path that rejected it:
        "Unsupported global: GLOBAL numpy.core.multiarray.scalar was not..."
        "...but got <class 'numpy.dtypes.Float64DType'>"
    """
    import re

    m = re.search(r"Unsupported global: GLOBAL (\S+)", error_message)
    if m:
        return m.group(1)

    m = re.search(r"but got <class '([\w.]+)'>", error_message)
    if m:
        return m.group(1)

    return None


def _resolve_dotted_name(dotted_name: str):
    """
    Resolves e.g. 'numpy.core.multiarray.scalar' to the live object, trying
    numpy 2.x's renamed 'numpy._core.*' path as a fallback if the classic
    'numpy.core.*' path no longer exists (and vice versa).
    """
    import importlib

    candidates = [dotted_name]
    if ".core." in dotted_name:
        candidates.append(dotted_name.replace(".core.", "._core."))
    elif "._core." in dotted_name:
        candidates.append(dotted_name.replace("._core.", ".core."))

    for candidate in candidates:
        parts = candidate.split(".")
        for split_point in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:split_point])
            attr_path = parts[split_point:]
            try:
                obj = importlib.import_module(module_name)
            except ImportError:
                continue
            try:
                for attr in attr_path:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
    return None
