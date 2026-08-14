"""Diagnostic figures: object-size distribution, PR curves, training history.

The size histogram is the one worth running first on any new dataset. It shows
where the small/medium/large cut-offs actually fall relative to the data, which
is the difference between an ``AP_medium`` that means something and one that
reports COCO's -1 sentinel because the bucket is empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

# Chosen before pyplot is imported: figures are written to disk and served
# through Gradio, never shown in a desktop window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ..config import VisualizationConfig  # noqa: E402
from ..evaluation.coco_eval import UNDEFINED, EvalResult  # noqa: E402


def plot_area_histogram(
    areas: Sequence[float],
    area_ranges: Optional[Mapping[str, Sequence[float]]] = None,
    path: Optional[str | Path] = None,
    *,
    config: Optional[VisualizationConfig] = None,
    title: str = "Ground-truth object size distribution",
) -> Any:
    """Histogram of object areas with the evaluation cut-offs drawn on."""
    config = config or VisualizationConfig()
    figure, axis = plt.subplots(figsize=(7, 4), dpi=config.dpi)

    # Areas span orders of magnitude on a small-object dataset; a linear x-axis
    # would compress everything interesting into the first bin.
    sides = [max(area, 1.0) ** 0.5 for area in areas]
    axis.hist(sides, bins=40, color="#4363d8", alpha=0.85)
    axis.set_xscale("log")
    axis.set_xlabel("object size (√area, pixels)")
    axis.set_ylabel("count")
    axis.set_title(title)

    if area_ranges:
        for label, bounds in area_ranges.items():
            if label == "all":
                continue
            edge = float(bounds[1]) ** 0.5
            if edge < max(sides) * 4:
                axis.axvline(edge, color="#e6194b", linestyle="--", linewidth=1)
                axis.text(
                    edge, axis.get_ylim()[1] * 0.92, f" {label} ≤ {edge:.0f}px",
                    color="#e6194b", fontsize=8, rotation=90, va="top",
                )

    figure.tight_layout()
    return _finish(figure, path)


def plot_pr_curves(
    result: EvalResult,
    path: Optional[str | Path] = None,
    *,
    iou_threshold: float = 0.5,
    area_labels: Sequence[str] = ("all", "small", "medium", "large"),
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Precision/recall per area bucket at one IoU threshold.

    Separating the buckets shows *where* precision is being lost. A curve that
    holds up for "large" and collapses for "small" is the signature of an
    anchor pyramid or an input resolution that is too coarse.
    """
    config = config or VisualizationConfig()
    figure, axis = plt.subplots(figsize=(6, 5), dpi=config.dpi)

    for label in area_labels:
        if label not in result.area_labels:
            continue
        recall, precision = result.pr_curve(iou_threshold, label)
        valid = precision > UNDEFINED
        if not valid.any():
            continue
        axis.plot(recall[valid], precision[valid], label=label, linewidth=1.6)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title(f"Precision–recall at IoU={iou_threshold:g}")
    axis.grid(alpha=0.25)
    axis.legend(title="area bucket", fontsize=8)

    figure.tight_layout()
    return _finish(figure, path)


def plot_history(
    history: Sequence[Mapping[str, Any]],
    path: Optional[str | Path] = None,
    *,
    metrics: Sequence[str] = ("AP", "AP_small", "AP_medium"),
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Validation metrics against epoch."""
    config = config or VisualizationConfig()
    figure, axis = plt.subplots(figsize=(7, 4), dpi=config.dpi)

    epochs = [entry.get("epoch", index + 1) for index, entry in enumerate(history)]
    for metric in metrics:
        values = [entry.get(metric, UNDEFINED) for entry in history]
        # -1 is "no ground truth in this bucket", not a score of -1; plotting it
        # would drag the axis below zero and imply a real measurement.
        points = [
            (epoch, value)
            for epoch, value in zip(epochs, values)
            if value is not None and value > UNDEFINED
        ]
        if points:
            axis.plot(*zip(*points), marker="o", markersize=3, label=metric)

    axis.set_xlabel("epoch")
    axis.set_ylabel("AP")
    axis.set_title("Validation metrics")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure.tight_layout()
    return _finish(figure, path)


def plot_per_class(
    result: EvalResult,
    path: Optional[str | Path] = None,
    *,
    metric: str = "AP",
    config: Optional[VisualizationConfig] = None,
) -> Any:
    """Per-class bar chart, sorted worst-first."""
    config = config or VisualizationConfig()
    entries = [
        (name, values.get(metric, UNDEFINED))
        for name, values in result.per_class.items()
        if values.get(metric, UNDEFINED) > UNDEFINED
    ]
    entries.sort(key=lambda item: item[1])

    figure, axis = plt.subplots(
        figsize=(7, max(2.5, 0.35 * len(entries) + 1)), dpi=config.dpi
    )
    if entries:
        names, values = zip(*entries)
        axis.barh(range(len(names)), values, color="#3cb44b", alpha=0.85)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_xlabel(metric)
    axis.set_title(f"Per-class {metric}")
    axis.grid(alpha=0.25, axis="x")

    figure.tight_layout()
    return _finish(figure, path)


def _finish(figure: Any, path: Optional[str | Path]) -> Any:
    """Save and close, or hand the figure back for the caller to embed."""
    if path is None:
        return figure
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target
