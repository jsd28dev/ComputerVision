"""Inference: postprocessing, tiled inference, and the Predictor facade."""

from __future__ import annotations

from .postprocess import apply_postprocess, apply_postprocess_batch, empty_prediction
from .predictor import PredictionResult, Predictor, to_uint8_tensor
from .tiling import (
    Tile,
    crop_tiles,
    generate_tiles,
    merge_tile_predictions,
    tiles_for,
)

__all__ = [
    "PredictionResult",
    "Predictor",
    "Tile",
    "apply_postprocess",
    "apply_postprocess_batch",
    "crop_tiles",
    "empty_prediction",
    "generate_tiles",
    "merge_tile_predictions",
    "tiles_for",
    "to_uint8_tensor",
]
