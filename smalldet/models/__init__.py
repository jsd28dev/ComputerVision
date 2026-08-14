"""Detector construction."""

from __future__ import annotations

from .anchors import anchors_per_location, build_anchor_generator, describe, pyramid_sizes
from .factory import (
    ARCHITECTURES,
    FAMILIES,
    ArchitectureSpec,
    available_architectures,
    build_model,
    load_checkpoint,
)

__all__ = [
    "ARCHITECTURES",
    "FAMILIES",
    "ArchitectureSpec",
    "anchors_per_location",
    "available_architectures",
    "build_anchor_generator",
    "build_model",
    "describe",
    "load_checkpoint",
    "pyramid_sizes",
]
