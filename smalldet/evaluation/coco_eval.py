"""COCO-style detection evaluation, implemented in NumPy.

This is a faithful reimplementation of ``pycocotools.cocoeval.COCOeval`` for
bounding boxes: the same greedy score-ordered matching, the same crowd handling
(IoA rather than IoU against crowd regions), the same 101-point interpolated
precision, and the same ``-1`` sentinel for an empty area bucket. When
``pycocotools`` is installed, ``tests/test_coco_eval.py`` asserts the two agree
to 1e-6, so the reimplementation is checked rather than assumed.

It exists for two reasons. Practically, ``pycocotools`` is a compiled dependency
that is awkward on Windows, and evaluation is the one thing this project cannot
do without. Substantively, ``AP_small`` and ``AP_medium`` are the metrics being
optimised, and those depend entirely on the ``area_ranges`` cut-offs — owning
the implementation means those cut-offs are a config value rather than a
constant buried in a C extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..config import EvalConfig

#: 101-point recall grid, as in COCO.
DEFAULT_REC_THRESHOLDS = np.linspace(0.0, 1.00, 101)
#: IoU=0.50:0.05:0.95.
DEFAULT_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)

#: COCO's "this bucket had no ground truth" sentinel. Not a score of zero.
UNDEFINED = -1.0


# ------------------------------------------------------------------ containers


@dataclass
class GroundTruth:
    """Annotations in the flat form the evaluator consumes."""

    image_ids: List[int]
    #: Each: {image_id, label, bbox (xyxy), area, iscrowd}
    annotations: List[Dict[str, Any]]
    class_names: List[str] = field(default_factory=list)

    @classmethod
    def from_dataset(cls, dataset: Any) -> "GroundTruth":
        payload = dataset.coco_ground_truth()
        return cls(
            image_ids=payload["image_ids"],
            annotations=payload["annotations"],
            class_names=payload.get("class_names", []),
        )

    @classmethod
    def from_targets(
        cls,
        targets: Iterable[Mapping[str, Any]],
        class_names: Optional[Sequence[str]] = None,
    ) -> "GroundTruth":
        """Build from the target dicts a DataLoader yields.

        Useful when the ground truth has already passed through augmentation
        and no longer matches the annotation file on disk.
        """
        image_ids: List[int] = []
        annotations: List[Dict[str, Any]] = []
        for target in targets:
            image_id = int(_as_scalar(target["image_id"]))
            image_ids.append(image_id)
            boxes = _as_array(target["boxes"]).reshape(-1, 4)
            labels = _as_array(target["labels"]).reshape(-1).astype(int)
            areas = (
                _as_array(target["area"]).reshape(-1)
                if "area" in target
                else (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            )
            iscrowd = (
                _as_array(target["iscrowd"]).reshape(-1).astype(int)
                if "iscrowd" in target
                else np.zeros(len(boxes), dtype=int)
            )
            for box, label, area, crowd in zip(boxes, labels, areas, iscrowd):
                annotations.append(
                    {
                        "image_id": image_id,
                        "label": int(label),
                        "bbox": [float(v) for v in box],
                        "area": float(area),
                        "iscrowd": int(crowd),
                    }
                )
        return cls(
            image_ids=image_ids,
            annotations=annotations,
            class_names=list(class_names or []),
        )

    def labels(self) -> List[int]:
        return sorted({int(annotation["label"]) for annotation in self.annotations})

    def areas(self, include_crowd: bool = False) -> List[float]:
        return [
            float(annotation["area"])
            for annotation in self.annotations
            if include_crowd or not annotation.get("iscrowd", 0)
        ]


@dataclass
class EvalResult:
    """Scalar metrics plus the raw precision/recall tensors behind them."""

    metrics: Dict[str, float]
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: (T, R, K, A, M) — IoU x recall x class x area bucket x max-dets.
    precision: Optional[np.ndarray] = None
    #: (T, K, A, M)
    recall: Optional[np.ndarray] = None
    iou_thresholds: Optional[np.ndarray] = None
    recall_thresholds: Optional[np.ndarray] = None
    area_labels: List[str] = field(default_factory=list)
    max_dets: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    num_ground_truth: int = 0
    num_detections: int = 0

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def get(self, key: str, default: float = UNDEFINED) -> float:
        return self.metrics.get(key, default)

    def pr_curve(
        self,
        iou_threshold: float = 0.5,
        area_label: str = "all",
        class_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precision against the 101-point recall grid, for plotting."""
        if self.precision is None or self.iou_thresholds is None:
            raise ValueError("this result carries no precision tensor")
        t = int(np.argmin(np.abs(self.iou_thresholds - iou_threshold)))
        a = self.area_labels.index(area_label)
        m = len(self.max_dets) - 1
        block = self.precision[t, :, :, a, m]
        if class_name is not None:
            k = self.class_names.index(class_name)
            curve = block[:, k]
        else:
            valid = block > UNDEFINED
            curve = np.where(
                valid.any(axis=1),
                np.divide(
                    np.where(valid, block, 0).sum(axis=1),
                    np.maximum(valid.sum(axis=1), 1),
                ),
                UNDEFINED,
            )
        return np.asarray(self.recall_thresholds), curve

    def table(self) -> str:
        """The familiar 12-line COCO summary, plus a small-object headline."""
        order = [
            ("AP", "IoU=0.50:0.95", "all", "maxDets=%d" % (self.max_dets[-1] if self.max_dets else 100)),
            ("AP50", "IoU=0.50", "all", ""),
            ("AP75", "IoU=0.75", "all", ""),
            ("AP_small", "IoU=0.50:0.95", "small", ""),
            ("AP_medium", "IoU=0.50:0.95", "medium", ""),
            ("AP_large", "IoU=0.50:0.95", "large", ""),
        ]
        lines = ["metric      value    IoU            area"]
        for key, iou, area, _ in order:
            if key in self.metrics:
                lines.append(
                    f"{key:<11} {_fmt(self.metrics[key]):>6}   {iou:<14} {area}"
                )
        for key in sorted(k for k in self.metrics if k.startswith("AR")):
            lines.append(f"{key:<11} {_fmt(self.metrics[key]):>6}")
        lines.append(
            f"({self.num_ground_truth} ground-truth objects, "
            f"{self.num_detections} detections)"
        )
        return "\n".join(lines)


# ------------------------------------------------------------------- evaluator


class CocoEvaluator:
    """Accumulates predictions image by image, then scores them.

    Usage mirrors torchvision's reference evaluator::

        evaluator = CocoEvaluator(GroundTruth.from_dataset(dataset), config.eval)
        for images, targets in loader:
            outputs = model(images)
            evaluator.update({int(t["image_id"]): o for t, o in zip(targets, outputs)})
        result = evaluator.accumulate()
    """

    def __init__(self, ground_truth: GroundTruth, config: Optional[EvalConfig] = None):
        self.ground_truth = ground_truth
        self.config = config or EvalConfig()
        if self.config.iou_type != "bbox":
            raise ValueError(
                f"eval.iou_type {self.config.iou_type!r} is not supported; this "
                "evaluator scores bounding boxes only"
            )
        self._detections: List[Dict[str, Any]] = []
        self._seen_images: set[int] = set()

    def update(self, predictions: Mapping[int, Mapping[str, Any]]) -> None:
        """Add one batch, keyed by image id."""
        for image_id, prediction in predictions.items():
            image_id = int(_as_scalar(image_id))
            self._seen_images.add(image_id)
            boxes = _as_array(prediction["boxes"]).reshape(-1, 4)
            scores = _as_array(prediction["scores"]).reshape(-1)
            labels = _as_array(prediction["labels"]).reshape(-1).astype(int)
            if not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(
                    f"image {image_id}: boxes/scores/labels have mismatched "
                    f"lengths ({len(boxes)}, {len(scores)}, {len(labels)})"
                )
            for box, score, label in zip(boxes, scores, labels):
                self._detections.append(
                    {
                        "image_id": image_id,
                        "label": int(label),
                        "bbox": [float(v) for v in box],
                        "score": float(score),
                    }
                )

    def reset(self) -> None:
        self._detections.clear()
        self._seen_images.clear()

    @property
    def detections(self) -> List[Dict[str, Any]]:
        return self._detections

    def accumulate(self) -> EvalResult:
        return evaluate_detections(
            self.ground_truth, self._detections, self.config
        )


def evaluate_detections(
    ground_truth: GroundTruth,
    detections: Sequence[Mapping[str, Any]],
    config: Optional[EvalConfig] = None,
) -> EvalResult:
    """Score ``detections`` against ``ground_truth``."""
    config = config or EvalConfig()

    iou_thresholds = np.asarray(
        config.iou_thresholds
        if config.iou_thresholds is not None
        else DEFAULT_IOU_THRESHOLDS,
        dtype=np.float64,
    )
    recall_thresholds = DEFAULT_REC_THRESHOLDS
    area_labels = list(config.area_ranges)
    area_ranges = [
        (float(config.area_ranges[label][0]), float(config.area_ranges[label][1]))
        for label in area_labels
    ]
    max_dets = sorted(int(value) for value in config.max_dets)
    top_max_det = max_dets[-1]

    image_ids = sorted(set(ground_truth.image_ids))
    labels = ground_truth.labels()
    if not labels:
        labels = sorted({int(detection["label"]) for detection in detections})

    gt_index = _index_by_image_label(ground_truth.annotations)
    dt_index = _index_by_image_label(detections)
    for entries in dt_index.values():
        entries.sort(key=lambda entry: -entry["score"])

    n_iou = len(iou_thresholds)
    n_rec = len(recall_thresholds)
    n_cls = len(labels)
    n_area = len(area_ranges)
    n_det = len(max_dets)

    precision = np.full((n_iou, n_rec, n_cls, n_area, n_det), UNDEFINED)
    recall = np.full((n_iou, n_cls, n_area, n_det), UNDEFINED)

    for k, label in enumerate(labels):
        # IoU between every detection and every ground truth is independent of
        # the area bucket and the max-dets cap, so it is computed once per
        # (image, class) and reused across both loops below.
        ious = {
            image_id: _iou_matrix(
                [entry["bbox"] for entry in dt_index.get((image_id, label), [])[:top_max_det]],
                [entry["bbox"] for entry in gt_index.get((image_id, label), [])],
                [
                    int(entry.get("iscrowd", 0))
                    for entry in gt_index.get((image_id, label), [])
                ],
            )
            for image_id in image_ids
        }

        for a, area_range in enumerate(area_ranges):
            per_image = [
                _evaluate_image(
                    gt_index.get((image_id, label), []),
                    dt_index.get((image_id, label), [])[:top_max_det],
                    ious[image_id],
                    iou_thresholds,
                    area_range,
                )
                for image_id in image_ids
            ]
            per_image = [entry for entry in per_image if entry is not None]
            if not per_image:
                continue

            for m, max_det in enumerate(max_dets):
                _accumulate_bucket(
                    per_image,
                    max_det,
                    recall_thresholds,
                    precision[:, :, k, a, m],
                    recall[:, k, a, m],
                )

    metrics = _summarize(
        precision, recall, iou_thresholds, area_labels, max_dets
    )
    per_class = (
        _summarize_per_class(
            precision, iou_thresholds, area_labels, max_dets, labels, ground_truth
        )
        if config.per_class
        else {}
    )

    return EvalResult(
        metrics=metrics,
        per_class=per_class,
        precision=precision,
        recall=recall,
        iou_thresholds=iou_thresholds,
        recall_thresholds=recall_thresholds,
        area_labels=area_labels,
        max_dets=max_dets,
        class_names=[_class_name(ground_truth, label) for label in labels],
        num_ground_truth=len(ground_truth.annotations),
        num_detections=len(detections),
    )


# ---------------------------------------------------------------- the algorithm


def _evaluate_image(
    gts: Sequence[Mapping[str, Any]],
    dts: Sequence[Mapping[str, Any]],
    ious: np.ndarray,
    iou_thresholds: np.ndarray,
    area_range: Tuple[float, float],
) -> Optional[Dict[str, np.ndarray]]:
    """Greedy score-ordered matching for one (image, class) pair.

    Ground truth is ignored when it is a crowd region or falls outside the
    current area bucket. Ignored ground truth neither counts as a miss nor
    penalises a detection that matches it — that is what makes ``AP_small``
    a measurement of small objects rather than of everything-minus-large ones.
    """
    if not gts and not dts:
        return None

    gt_ignore = np.array(
        [
            bool(gt.get("iscrowd", 0))
            or not (area_range[0] <= float(gt["area"]) <= area_range[1])
            for gt in gts
        ],
        dtype=bool,
    )
    # Stable sort putting non-ignored ground truth first, so the matcher prefers
    # a real object over an ignored one at equal IoU.
    gt_order = np.argsort(gt_ignore, kind="mergesort")
    gt_ignore = gt_ignore[gt_order] if len(gts) else gt_ignore
    gt_iscrowd = np.array(
        [int(gts[i].get("iscrowd", 0)) for i in gt_order], dtype=bool
    )
    ordered_ious = ious[:, gt_order] if ious.size else ious

    n_iou = len(iou_thresholds)
    n_gt = len(gts)
    n_dt = len(dts)

    gt_matched = np.zeros((n_iou, n_gt), dtype=bool)
    dt_matched = np.zeros((n_iou, n_dt), dtype=bool)
    dt_ignore = np.zeros((n_iou, n_dt), dtype=bool)

    if ordered_ious.size:
        for t, threshold in enumerate(iou_thresholds):
            for d in range(n_dt):
                # 1 - 1e-10 keeps a perfect overlap matchable at threshold 1.0.
                best_iou = min(threshold, 1 - 1e-10)
                best = -1
                for g in range(n_gt):
                    if gt_matched[t, g] and not gt_iscrowd[g]:
                        continue
                    # Once a non-ignored match exists, stop before the ignored
                    # block: everything from here on is ignored ground truth.
                    if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                        break
                    if ordered_ious[d, g] < best_iou:
                        continue
                    best_iou = ordered_ious[d, g]
                    best = g
                if best == -1:
                    continue
                dt_ignore[t, d] = gt_ignore[best]
                dt_matched[t, d] = True
                gt_matched[t, best] = True

    # An unmatched detection outside the area bucket is neither a hit nor a
    # false positive for this bucket; it belongs to another one.
    dt_areas = np.array(
        [
            (dt["bbox"][2] - dt["bbox"][0]) * (dt["bbox"][3] - dt["bbox"][1])
            for dt in dts
        ],
        dtype=np.float64,
    ).reshape(1, n_dt)
    out_of_range = (dt_areas < area_range[0]) | (dt_areas > area_range[1])
    dt_ignore = dt_ignore | (~dt_matched & np.repeat(out_of_range, n_iou, axis=0))

    return {
        "dt_matched": dt_matched,
        "dt_ignore": dt_ignore,
        "dt_scores": np.array([float(dt["score"]) for dt in dts], dtype=np.float64),
        "gt_ignore": gt_ignore,
    }


def _accumulate_bucket(
    per_image: Sequence[Dict[str, np.ndarray]],
    max_det: int,
    recall_thresholds: np.ndarray,
    precision_out: np.ndarray,
    recall_out: np.ndarray,
) -> None:
    """Pool one (class, area, max-dets) bucket across images into a PR curve."""
    scores = np.concatenate(
        [entry["dt_scores"][:max_det] for entry in per_image]
    )
    gt_ignore = np.concatenate([entry["gt_ignore"] for entry in per_image])
    n_positives = int(np.count_nonzero(~gt_ignore))
    if n_positives == 0:
        # No ground truth in this bucket: leave the -1 sentinel in place.
        return

    # Detections are ranked globally by score, not per image — AP measures one
    # ranking over the whole split.
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    matched = np.concatenate(
        [entry["dt_matched"][:, :max_det] for entry in per_image], axis=1
    )[:, order]
    ignored = np.concatenate(
        [entry["dt_ignore"][:, :max_det] for entry in per_image], axis=1
    )[:, order]

    true_positives = np.cumsum(matched & ~ignored, axis=1).astype(np.float64)
    false_positives = np.cumsum(~matched & ~ignored, axis=1).astype(np.float64)

    for t in range(matched.shape[0]):
        tp, fp = true_positives[t], false_positives[t]
        n = len(tp)
        rc = tp / n_positives
        pr = tp / (tp + fp + np.spacing(1))
        recall_out[t] = rc[-1] if n else 0.0

        # Make precision monotonically non-increasing in recall, then read it
        # off at the 101 fixed recall points. This is COCO's interpolation.
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        indices = np.searchsorted(rc, recall_thresholds, side="left")
        curve = np.zeros(len(recall_thresholds))
        valid = indices < n
        curve[valid] = pr[indices[valid]]
        precision_out[t, :] = curve


# ------------------------------------------------------------------ summarizing


def _mean(values: np.ndarray) -> float:
    kept = values[values > UNDEFINED]
    return float(np.mean(kept)) if kept.size else UNDEFINED


def _summarize(
    precision: np.ndarray,
    recall: np.ndarray,
    iou_thresholds: np.ndarray,
    area_labels: Sequence[str],
    max_dets: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    m_top = len(max_dets) - 1
    a_all = area_labels.index("all")

    metrics["AP"] = _mean(precision[:, :, :, a_all, m_top])
    for name, threshold in (("AP50", 0.5), ("AP75", 0.75)):
        matches = np.isclose(iou_thresholds, threshold)
        if matches.any():
            t = int(np.argmax(matches))
            metrics[name] = _mean(precision[t : t + 1, :, :, a_all, m_top])

    for a, label in enumerate(area_labels):
        if label == "all":
            continue
        metrics[f"AP_{label}"] = _mean(precision[:, :, :, a, m_top])
        metrics[f"AR_{label}"] = _mean(recall[:, :, a, m_top])

    for m, max_det in enumerate(max_dets):
        metrics[f"AR_{max_det}"] = _mean(recall[:, :, a_all, m])

    return metrics


def _summarize_per_class(
    precision: np.ndarray,
    iou_thresholds: np.ndarray,
    area_labels: Sequence[str],
    max_dets: Sequence[int],
    labels: Sequence[int],
    ground_truth: GroundTruth,
) -> Dict[str, Dict[str, float]]:
    """Per-class AP, including the small/medium split.

    On a small-object dataset the classes rarely share a size distribution —
    one class may be 8px and another 40px — so a single AP hides which class
    the model is actually failing.
    """
    m_top = len(max_dets) - 1
    a_all = area_labels.index("all")
    per_class: Dict[str, Dict[str, float]] = {}
    for k, label in enumerate(labels):
        name = _class_name(ground_truth, label)
        entry = {"AP": _mean(precision[:, :, k, a_all, m_top])}
        matches = np.isclose(iou_thresholds, 0.5)
        if matches.any():
            t = int(np.argmax(matches))
            entry["AP50"] = _mean(precision[t : t + 1, :, k, a_all, m_top])
        for a, area_label in enumerate(area_labels):
            if area_label == "all":
                continue
            entry[f"AP_{area_label}"] = _mean(precision[:, :, k, a, m_top])
        per_class[name] = entry
    return per_class


# ---------------------------------------------------------------------- detail


def _iou_matrix(
    dt_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    iscrowd: Sequence[int],
) -> np.ndarray:
    """IoU between detections and ground truth, in XYXY.

    For crowd ground truth the denominator is the detection's own area, not the
    union — a detection inside a crowd region should not be penalised for not
    covering the whole region. This matches ``pycocotools``.
    """
    if not dt_boxes or not gt_boxes:
        return np.zeros((len(dt_boxes), len(gt_boxes)), dtype=np.float64)

    dt = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    crowd = np.asarray(iscrowd, dtype=bool).reshape(1, -1)

    dt_area = np.clip(dt[:, 2] - dt[:, 0], 0, None) * np.clip(
        dt[:, 3] - dt[:, 1], 0, None
    )
    gt_area = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(
        gt[:, 3] - gt[:, 1], 0, None
    )

    left = np.maximum(dt[:, None, 0], gt[None, :, 0])
    top = np.maximum(dt[:, None, 1], gt[None, :, 1])
    right = np.minimum(dt[:, None, 2], gt[None, :, 2])
    bottom = np.minimum(dt[:, None, 3], gt[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)

    union = dt_area[:, None] + gt_area[None, :] - intersection
    denominator = np.where(crowd, dt_area[:, None], union)
    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection),
        where=denominator > 0,
    )


def _index_by_image_label(
    entries: Iterable[Mapping[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    index: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for entry in entries:
        key = (int(entry["image_id"]), int(entry["label"]))
        index.setdefault(key, []).append(dict(entry))
    return index


def _class_name(ground_truth: GroundTruth, label: int) -> str:
    if 0 <= label < len(ground_truth.class_names):
        return ground_truth.class_names[label]
    return f"class_{label}"


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _as_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return value.item()
    return value


def _fmt(value: float) -> str:
    return "n/a" if value <= UNDEFINED else f"{value:.4f}"
