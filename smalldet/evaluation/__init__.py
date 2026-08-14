"""COCO-style evaluation, with AP_small / AP_medium as first-class outputs."""

from __future__ import annotations

from .coco_eval import (
    DEFAULT_IOU_THRESHOLDS,
    DEFAULT_REC_THRESHOLDS,
    UNDEFINED,
    CocoEvaluator,
    EvalResult,
    GroundTruth,
    evaluate_detections,
)
from .runner import (
    evaluate_model,
    replace_area_ranges,
    resolve_eval_config,
    save_detections,
    save_report,
)

__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "DEFAULT_REC_THRESHOLDS",
    "UNDEFINED",
    "CocoEvaluator",
    "EvalResult",
    "GroundTruth",
    "evaluate_detections",
    "evaluate_model",
    "replace_area_ranges",
    "resolve_eval_config",
    "save_detections",
    "save_report",
]
