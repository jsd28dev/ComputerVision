"""Device selection and reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config device string into a real ``torch.device``.

    ``auto`` is resolved by asking torch at call time, never by inspecting the
    host. A config that says ``cuda`` on a machine without one fails loudly
    here rather than after the first epoch of data loading.
    """
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {spec!r} was requested but CUDA is not available in this "
            "process. Use device: auto to fall back to CPU."
        )
    return device


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy (if present), and torch.

    Full determinism is off by default: it forces slower cuDNN kernels, and a
    detection training run is dominated by data-order effects that the seed
    already controls.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    """A generator for DataLoader shuffling, so epochs are reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"{device} ({name}, {total:.1f} GiB)"
    return str(device)


def autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    """The mixed-precision dtype worth using on this device, if any."""
    if device.type == "cuda":
        return torch.float16
    if device.type == "cpu":
        return torch.bfloat16
    return None
