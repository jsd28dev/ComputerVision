"""Inference-time filtering of raw detector output.

A detector's eval-mode output is deliberately permissive — torchvision's
default ``box_score_thresh`` is 0.05, so a frame comes back with hundreds of
boxes. That is correct for computing AP, which integrates over the whole
precision/recall curve, and useless for a user looking at an image. Everything
here is therefore *presentation* filtering, applied after evaluation, never
before it.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torchvision.ops import batched_nms, nms

from ..config import PostprocessConfig


def apply_postprocess(
    prediction: Mapping[str, torch.Tensor], config: PostprocessConfig
) -> Dict[str, torch.Tensor]:
    """Filter one prediction dict. Returns a new dict; the input is untouched."""
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    keep = scores >= config.score_threshold

    if config.min_box_area > 0:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep &= areas >= config.min_box_area

    if config.allowed_labels:
        allowed = torch.tensor(
            config.allowed_labels, dtype=labels.dtype, device=labels.device
        )
        keep &= torch.isin(labels, allowed)

    indices = torch.nonzero(keep, as_tuple=False).flatten()

    if config.nms_iou_threshold is not None and indices.numel():
        # An extra pass on top of the model's own NMS. Useful when tiling has
        # merged several passes, or when a lower IoU than the model's default
        # is wanted for densely packed small parts.
        selected = (
            nms(boxes[indices], scores[indices], config.nms_iou_threshold)
            if config.class_agnostic_nms
            else batched_nms(
                boxes[indices],
                scores[indices],
                labels[indices],
                config.nms_iou_threshold,
            )
        )
        indices = indices[selected]

    # Sort by score so max_detections keeps the most confident, and so any
    # consumer that truncates the list gets a sensible prefix.
    order = torch.argsort(scores[indices], descending=True)
    indices = indices[order]

    if config.max_detections is not None:
        indices = indices[: config.max_detections]

    return _index(prediction, indices)


def apply_postprocess_batch(
    predictions: Sequence[Mapping[str, torch.Tensor]], config: PostprocessConfig
) -> list[Dict[str, torch.Tensor]]:
    return [apply_postprocess(prediction, config) for prediction in predictions]


def _index(
    prediction: Mapping[str, Any], indices: torch.Tensor
) -> Dict[str, Any]:
    """Select the same rows from every per-detection tensor in the dict.

    ``masks`` is carried through alongside boxes and labels so a Mask R-CNN
    prediction survives filtering intact.
    """
    result: Dict[str, Any] = {}
    for key, value in prediction.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == prediction["boxes"].shape[:1]:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def empty_prediction(
    reference: Optional[Mapping[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """A well-formed prediction with no detections in it."""
    device = reference["boxes"].device if reference is not None else torch.device("cpu")
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "scores": torch.zeros((0,), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
    }
