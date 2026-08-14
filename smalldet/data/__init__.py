"""Datasets, augmentation, and batching."""

from __future__ import annotations

from .coco import CocoDetectionDataset, build_dataset
from .loaders import Batch, build_dataloader, collate_fn, detach_to_cpu, move_to_device
from .stats import AreaStats, area_ranges_from_percentiles, percentile, summarize_areas
from .transforms import TRANSFORMS, build_transform, build_transforms

__all__ = [
    "TRANSFORMS",
    "AreaStats",
    "Batch",
    "CocoDetectionDataset",
    "area_ranges_from_percentiles",
    "build_dataloader",
    "build_dataset",
    "build_transform",
    "build_transforms",
    "collate_fn",
    "detach_to_cpu",
    "move_to_device",
    "percentile",
    "summarize_areas",
]
