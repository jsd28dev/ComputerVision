"""Run a model over a dataloader and score it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config, EvalConfig
from ..data.loaders import detach_to_cpu, move_to_device
from ..data.stats import area_ranges_from_percentiles, summarize_areas
from .coco_eval import CocoEvaluator, EvalResult, GroundTruth


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    ground_truth: GroundTruth,
    config: Optional[EvalConfig] = None,
    *,
    device: torch.device | str = "cpu",
    max_batches: Optional[int] = None,
    progress: Optional[Any] = None,
) -> EvalResult:
    """Score ``model`` on ``loader`` with COCO metrics.

    No score threshold is applied here on purpose. AP integrates precision over
    the whole recall range, so discarding low-confidence detections before
    scoring truncates the tail of the PR curve and reports an AP that is lower
    than the model's actual one. Thresholding belongs at prediction time, not
    evaluation time.
    """
    config = config or EvalConfig()
    evaluator = CocoEvaluator(ground_truth, config)

    was_training = model.training
    model.eval()
    try:
        for index, (images, targets) in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images, _ = move_to_device(images, targets, device)
            outputs = detach_to_cpu(model(images))
            evaluator.update(
                {
                    int(target["image_id"]): output
                    for target, output in zip(targets, outputs)
                }
            )
            if progress is not None:
                progress(index, len(loader))
    finally:
        model.train(was_training)

    return evaluator.accumulate()


def resolve_eval_config(
    config: Config, ground_truth: GroundTruth, *, verbose: bool = True
) -> EvalConfig:
    """Apply ``eval.auto_area_ranges`` against real ground-truth areas.

    Returns the eval config to actually use. When auto ranges are off this is a
    no-op, but the size summary is still worth printing: an ``AP_medium`` of
    -1 in the final table is otherwise a mystery, and the histogram immediately
    explains it as an empty bucket.
    """
    eval_config = config.eval
    areas = ground_truth.areas()
    if not areas:
        return eval_config

    if eval_config.auto_area_ranges:
        ranges = area_ranges_from_percentiles(
            areas, eval_config.auto_area_percentiles
        )
        eval_config = replace_area_ranges(eval_config, ranges)

    if verbose:
        stats = summarize_areas(areas, eval_config.area_ranges)
        print(stats.describe())
        empty = [
            label
            for label, fraction in stats.bucket_fractions.items()
            if label != "all" and fraction == 0.0
        ]
        if empty:
            print(
                f"note: area bucket(s) {', '.join(empty)} contain no ground truth, "
                f"so AP_{empty[0]} will report -1. Set eval.auto_area_ranges: true "
                "to derive buckets from this dataset instead of COCO's."
            )
    return eval_config


def replace_area_ranges(
    config: EvalConfig, area_ranges: Dict[str, List[float]]
) -> EvalConfig:
    """Return a copy of ``config`` with different area buckets."""
    import dataclasses

    return dataclasses.replace(config, area_ranges=area_ranges)


def save_report(
    result: EvalResult,
    directory: str | Path,
    *,
    name: str = "evaluation",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write metrics as JSON plus a readable summary next to it."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "metrics": result.metrics,
        "per_class": result.per_class,
        "area_labels": result.area_labels,
        "max_dets": result.max_dets,
        "num_ground_truth": result.num_ground_truth,
        "num_detections": result.num_detections,
    }
    if extra:
        payload.update(extra)

    json_path = target / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (target / f"{name}.txt").write_text(result.table(), encoding="utf-8")
    return json_path


def save_detections(
    detections: Sequence[Dict[str, Any]],
    path: str | Path,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """Dump raw detections as JSON, in COCO's XYWH result convention."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = int(detection["label"])
        record = {
            "image_id": int(detection["image_id"]),
            "category_id": label,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(detection.get("score", 1.0)),
        }
        if class_names and 0 <= label < len(class_names):
            record["category_name"] = class_names[label]
        records.append(record)
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return target
