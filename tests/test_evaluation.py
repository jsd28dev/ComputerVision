"""COCO-style evaluation, with AP_small and AP_medium as the metrics of record.

These are the highest-value tests in the suite: every training decision is
judged by these numbers, so an evaluator that is subtly wrong invalidates
everything else. Each case has an analytically known answer.
"""

from __future__ import annotations

import dataclasses

from _support import assert_close, has_module, skip, synthetic_dataset

from smalldet.config import EvalConfig
from smalldet.evaluation import (
    UNDEFINED,
    CocoEvaluator,
    GroundTruth,
    evaluate_detections,
)

# Areas: 20x20 = 400 (small), 60x60 = 3600 (medium), 200x200 = 40000 (large).
SMALL_BOX = [0.0, 0.0, 20.0, 20.0]
MEDIUM_BOX = [100.0, 100.0, 160.0, 160.0]
LARGE_BOX = [300.0, 300.0, 500.0, 500.0]

DEFAULT_RANGES = {
    "all": [0.0, 1e10],
    "small": [0.0, 1024.0],
    "medium": [1024.0, 9216.0],
    "large": [9216.0, 1e10],
}


def _gt(*boxes, image_id: int = 1, label: int = 1, iscrowd: int = 0):
    return [
        {
            "image_id": image_id,
            "label": label,
            "bbox": list(box),
            "area": (box[2] - box[0]) * (box[3] - box[1]),
            "iscrowd": iscrowd,
        }
        for box in boxes
    ]


def _dt(*boxes, image_id: int = 1, label: int = 1, score: float = 0.9):
    return [
        {"image_id": image_id, "label": label, "bbox": list(box), "score": score}
        for box in boxes
    ]


def _config(**kwargs) -> EvalConfig:
    return EvalConfig(area_ranges=dict(DEFAULT_RANGES), **kwargs)


def _ground_truth(annotations, image_ids=(1,)):
    return GroundTruth(
        image_ids=list(image_ids),
        annotations=annotations,
        class_names=["__background__", "part", "other"],
    )


# --------------------------------------------------------------- known answers


def test_perfect_predictions_score_one():
    gt = _ground_truth(_gt(SMALL_BOX, MEDIUM_BOX, LARGE_BOX))
    result = evaluate_detections(gt, _dt(SMALL_BOX, MEDIUM_BOX, LARGE_BOX), _config())

    assert_close(result["AP"], 1.0, label="AP")
    assert_close(result["AP50"], 1.0, label="AP50")
    assert_close(result["AP75"], 1.0, label="AP75")
    assert_close(result["AP_small"], 1.0, label="AP_small")
    assert_close(result["AP_medium"], 1.0, label="AP_medium")
    assert_close(result["AP_large"], 1.0, label="AP_large")


def test_no_predictions_scores_zero_not_undefined():
    """Zero is a real score (the model found nothing); -1 means "no ground
    truth to score against". Conflating them hides a broken model."""
    gt = _ground_truth(_gt(SMALL_BOX))
    result = evaluate_detections(gt, [], _config())
    assert_close(result["AP"], 0.0, label="AP")
    assert_close(result["AP_small"], 0.0, label="AP_small")


def test_iou_threshold_separates_ap50_from_ap75():
    """A box shifted by a fifth of its side overlaps by IoU 2/3: a hit at 0.50,
    a miss at 0.75. This is what makes AP@[.50:.95] stricter than AP50.

    The shift is chosen so the IoU falls between two thresholds rather than on
    one, so the test does not depend on how a float comparison rounds at the
    boundary.
    """
    box = [0.0, 0.0, 40.0, 40.0]
    shifted = [8.0, 0.0, 48.0, 40.0]  # intersection 1280 / union 1920 = 0.667

    result = evaluate_detections(_ground_truth(_gt(box)), _dt(shifted), _config())
    assert_close(result["AP50"], 1.0, label="AP50")
    assert_close(result["AP75"], 0.0, label="AP75")
    # Of the ten thresholds, 0.50/0.55/0.60/0.65 hit and the rest miss.
    assert_close(result["AP"], 0.4, tolerance=1e-6, label="AP")


def test_duplicate_detections_count_as_false_positives():
    """Only one detection may match a given ground truth; the rest are FPs.
    Without this, a model that floods the frame would score perfectly."""
    gt = _ground_truth(_gt(SMALL_BOX))
    detections = _dt(SMALL_BOX, score=0.9) + _dt(SMALL_BOX, score=0.8)
    result = evaluate_detections(gt, detections, _config())
    # Recall 1.0 is reached at precision 1.0, but the 101-point interpolation
    # takes the running maximum, so AP stays 1.0 while precision at the tail
    # halves. What must NOT happen is the duplicate counting as a second hit.
    assert result.num_detections == 2
    assert_close(result["AP50"], 1.0, label="AP50")

    # A duplicate that arrives BEFORE any true positive does cost precision.
    misplaced = _dt([500.0, 500.0, 520.0, 520.0], score=0.99) + _dt(SMALL_BOX, score=0.5)
    lowered = evaluate_detections(gt, misplaced, _config())
    assert lowered["AP50"] < 1.0


def test_score_ranking_drives_the_metric():
    """AP measures one ranking over the whole split. A confident wrong box
    must hurt more than a diffident one."""
    gt = _ground_truth(_gt(SMALL_BOX, MEDIUM_BOX))
    false_positive = [400.0, 0.0, 420.0, 20.0]

    confident = evaluate_detections(
        gt,
        _dt(false_positive, score=0.99) + _dt(SMALL_BOX, MEDIUM_BOX, score=0.5),
        _config(),
    )
    diffident = evaluate_detections(
        gt,
        _dt(false_positive, score=0.01) + _dt(SMALL_BOX, MEDIUM_BOX, score=0.9),
        _config(),
    )
    assert confident["AP50"] < diffident["AP50"]


# ----------------------------------------------------------------- area buckets


def test_area_buckets_isolate_object_sizes():
    """The property AP_small depends on: a detector that only finds small
    objects must score 1.0 on small and 0.0 on medium, not something in between.
    """
    gt = _ground_truth(_gt(SMALL_BOX, MEDIUM_BOX))
    result = evaluate_detections(gt, _dt(SMALL_BOX), _config())

    assert_close(result["AP_small"], 1.0, label="AP_small")
    assert_close(result["AP_medium"], 0.0, label="AP_medium")
    # Half the objects found overall.
    assert 0.0 < result["AP"] < 1.0


def test_a_detection_outside_the_bucket_is_ignored_not_penalised():
    """A small false positive must not damage AP_medium — it belongs to another
    bucket. Otherwise every bucket would be polluted by every other one."""
    gt = _ground_truth(_gt(MEDIUM_BOX))
    clean = evaluate_detections(gt, _dt(MEDIUM_BOX), _config())
    with_small_fp = evaluate_detections(
        gt, _dt(MEDIUM_BOX) + _dt([400.0, 0.0, 410.0, 10.0], score=0.95), _config()
    )
    assert_close(with_small_fp["AP_medium"], clean["AP_medium"], label="AP_medium")


def test_empty_bucket_reports_the_sentinel_not_a_score():
    """-1 means "no ground truth in this bucket". Anything that consumes
    metrics must be able to tell it apart from a genuine zero."""
    gt = _ground_truth(_gt(SMALL_BOX))
    result = evaluate_detections(gt, _dt(SMALL_BOX), _config())
    # The sentinel is compared exactly — it is a flag, not a measurement.
    assert result["AP_large"] == UNDEFINED
    assert result["AR_large"] == UNDEFINED
    # The score is compared with a tolerance: COCO's precision denominator
    # carries an epsilon, so a perfect score is 1 - 2e-16 rather than exactly 1.
    assert_close(result["AP_small"], 1.0, label="AP_small")
    assert "n/a" in result.table()


def test_custom_area_ranges_change_which_objects_are_small():
    """COCO's 32^2 cut-off is calibrated to COCO's resolution. Re-cutting the
    buckets must actually move objects between them."""
    gt = _ground_truth(_gt(MEDIUM_BOX))  # 3600 px^2

    with_coco_cuts = evaluate_detections(gt, _dt(MEDIUM_BOX), _config())
    assert with_coco_cuts["AP_small"] == UNDEFINED  # nothing in the small bucket

    recut = _config()
    recut = dataclasses.replace(
        recut,
        area_ranges={
            "all": [0.0, 1e10],
            "small": [0.0, 10000.0],  # now 3600 counts as small
            "medium": [10000.0, 50000.0],
        },
    )
    with_recut = evaluate_detections(gt, _dt(MEDIUM_BOX), recut)
    assert_close(with_recut["AP_small"], 1.0, label="AP_small")
    assert with_recut["AP_medium"] == UNDEFINED


def test_auto_area_ranges_keep_every_bucket_populated():
    from smalldet.data import area_ranges_from_percentiles
    from smalldet.data.coco import CocoDetectionDataset

    paths = synthetic_dataset()
    dataset = CocoDetectionDataset(paths["images"], paths["train"])
    gt = GroundTruth.from_dataset(dataset)

    config = dataclasses.replace(
        EvalConfig(),
        area_ranges=area_ranges_from_percentiles(gt.areas(), [33.3, 66.6]),
    )
    perfect = [
        {**annotation, "score": 0.9} for annotation in gt.annotations
    ]
    result = evaluate_detections(gt, perfect, config)
    for label in ("small", "medium", "large"):
        assert result[f"AP_{label}"] > UNDEFINED, f"AP_{label} is undefined"
        assert_close(result[f"AP_{label}"], 1.0, label=f"AP_{label}")


# ----------------------------------------------------------------------- crowd


def test_detections_inside_a_crowd_region_are_ignored():
    """Crowd ground truth is scored by intersection-over-detection-area, so a
    detection inside a crowd is neither a hit nor a false positive."""
    annotations = _gt(SMALL_BOX) + _gt(
        [100.0, 100.0, 200.0, 200.0], iscrowd=1
    )
    gt = _ground_truth(annotations)

    inside_crowd = [120.0, 120.0, 140.0, 140.0]
    result = evaluate_detections(
        gt, _dt(SMALL_BOX, score=0.9) + _dt(inside_crowd, score=0.8), _config()
    )
    # Without crowd handling the second detection would be a false positive and
    # drag AP below 1.0.
    assert_close(result["AP50"], 1.0, label="AP50")


# ------------------------------------------------------------------- max_dets


def test_max_dets_caps_recall():
    """AR_1 can never exceed 1/N when there are N objects in an image."""
    gt = _ground_truth(
        _gt(
            [0.0, 0.0, 20.0, 20.0],
            [40.0, 0.0, 60.0, 20.0],
            [80.0, 0.0, 100.0, 20.0],
            [120.0, 0.0, 140.0, 20.0],
        )
    )
    detections = [{**a, "score": 0.9} for a in gt.annotations]
    result = evaluate_detections(gt, detections, _config())

    assert_close(result["AR_1"], 0.25, label="AR_1")
    assert_close(result["AR_10"], 1.0, label="AR_10")
    assert_close(result["AR_100"], 1.0, label="AR_100")


# ---------------------------------------------------------------- multi-class


def test_labels_do_not_cross_match():
    """A perfectly placed box with the wrong class is a miss and a false
    positive, not a hit."""
    gt = _ground_truth(_gt(SMALL_BOX, label=1))
    result = evaluate_detections(gt, _dt(SMALL_BOX, label=2), _config())
    assert_close(result["AP"], 0.0, label="AP")


def test_per_class_breakdown_is_reported():
    annotations = _gt(SMALL_BOX, label=1) + _gt(MEDIUM_BOX, label=2)
    gt = _ground_truth(annotations)
    detections = _dt(SMALL_BOX, label=1)  # class 2 missed entirely
    result = evaluate_detections(gt, detections, _config(per_class=True))

    assert set(result.per_class) == {"part", "other"}
    assert_close(result.per_class["part"]["AP"], 1.0, label="part AP")
    assert_close(result.per_class["other"]["AP"], 0.0, label="other AP")


# ------------------------------------------------------------------- plumbing


def test_evaluator_accumulates_batches_keyed_by_image_id():
    import torch

    gt = _ground_truth(
        _gt(SMALL_BOX, image_id=1) + _gt(MEDIUM_BOX, image_id=2), image_ids=(1, 2)
    )
    evaluator = CocoEvaluator(gt, _config())
    evaluator.update(
        {
            1: {
                "boxes": torch.tensor([SMALL_BOX]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            },
            2: {
                "boxes": torch.tensor([MEDIUM_BOX]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
            },
        }
    )
    result = evaluator.accumulate()
    assert result.num_detections == 2
    assert_close(result["AP"], 1.0, label="AP")


def test_mismatched_prediction_lengths_are_rejected():
    import torch

    from _support import expect_error

    evaluator = CocoEvaluator(_ground_truth(_gt(SMALL_BOX)), _config())
    expect_error(
        lambda: evaluator.update(
            {
                1: {
                    "boxes": torch.zeros(3, 4),
                    "scores": torch.zeros(2),
                    "labels": torch.zeros(3, dtype=torch.int64),
                }
            }
        ),
        ValueError,
        contains="mismatched",
    )


def test_ground_truth_can_be_built_from_dataloader_targets():
    import torch

    targets = [
        {
            "image_id": torch.tensor(1),
            "boxes": torch.tensor([SMALL_BOX]),
            "labels": torch.tensor([1]),
            "area": torch.tensor([400.0]),
            "iscrowd": torch.tensor([0]),
        }
    ]
    gt = GroundTruth.from_targets(targets, ["__background__", "part"])
    assert gt.image_ids == [1]
    assert len(gt.annotations) == 1
    assert gt.annotations[0]["area"] == 400.0


def test_pr_curve_is_available_per_area_bucket():
    gt = _ground_truth(_gt(SMALL_BOX, MEDIUM_BOX))
    result = evaluate_detections(gt, _dt(SMALL_BOX, MEDIUM_BOX), _config())
    recall, precision = result.pr_curve(0.5, "small")
    assert len(recall) == len(precision) == 101
    assert precision.max() > 0.9


# ------------------------------------------------------- external cross-checks


def _pycocotools_reference(annotations, detections, area_ranges):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = sorted({a["image_id"] for a in annotations})
    labels = sorted({a["label"] for a in annotations})

    coco = COCO()
    coco.dataset = {
        "images": [{"id": i} for i in image_ids],
        "categories": [{"id": label} for label in labels],
        "annotations": [
            {
                "id": index + 1,
                "image_id": a["image_id"],
                "category_id": a["label"],
                "bbox": [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][2] - a["bbox"][0],
                    a["bbox"][3] - a["bbox"][1],
                ],
                "area": a["area"],
                "iscrowd": a.get("iscrowd", 0),
            }
            for index, a in enumerate(annotations)
        ],
    }
    coco.createIndex()

    results = coco.loadRes(
        [
            {
                "image_id": d["image_id"],
                "category_id": d["label"],
                "bbox": [
                    d["bbox"][0],
                    d["bbox"][1],
                    d["bbox"][2] - d["bbox"][0],
                    d["bbox"][3] - d["bbox"][1],
                ],
                "score": d["score"],
            }
            for d in detections
        ]
    )

    evaluator = COCOeval(coco, results, "bbox")
    evaluator.params.areaRng = [list(area_ranges[k]) for k in area_ranges]
    evaluator.params.areaRngLbl = list(area_ranges)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def test_matches_pycocotools_on_a_realistic_split():
    """The reimplementation is checked against the reference, not assumed.

    Skipped when pycocotools is absent — which is precisely why this project
    ships its own implementation, but the check runs wherever it can.
    """
    if not has_module("pycocotools"):
        skip("pycocotools is not installed (the NumPy evaluator is the fallback)")

    import random

    from smalldet.data.coco import CocoDetectionDataset

    paths = synthetic_dataset()
    dataset = CocoDetectionDataset(paths["images"], paths["train"])
    gt = GroundTruth.from_dataset(dataset)

    # A plausible detector: most objects found with jitter, some missed, a few
    # false positives, varied scores. Exercises matching, ranking, and buckets.
    rng = random.Random(11)
    detections = []
    for annotation in gt.annotations:
        if rng.random() < 0.15:
            continue  # missed
        x1, y1, x2, y2 = annotation["bbox"]
        jitter = (x2 - x1) * 0.12
        detections.append(
            {
                "image_id": annotation["image_id"],
                "label": annotation["label"],
                "bbox": [
                    x1 + rng.uniform(-jitter, jitter),
                    y1 + rng.uniform(-jitter, jitter),
                    x2 + rng.uniform(-jitter, jitter),
                    y2 + rng.uniform(-jitter, jitter),
                ],
                "score": rng.uniform(0.2, 1.0),
            }
        )
    for _ in range(12):
        x, y = rng.uniform(0, 120), rng.uniform(0, 120)
        size = rng.uniform(6, 40)
        detections.append(
            {
                "image_id": rng.choice(gt.image_ids),
                "label": rng.randint(1, 3),
                "bbox": [x, y, x + size, y + size],
                "score": rng.uniform(0.05, 0.9),
            }
        )

    ours = evaluate_detections(gt, detections, _config())
    reference = _pycocotools_reference(gt.annotations, detections, DEFAULT_RANGES)

    # COCOeval.stats order: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ...
    expected = {
        "AP": reference.stats[0],
        "AP50": reference.stats[1],
        "AP75": reference.stats[2],
        "AP_small": reference.stats[3],
        "AP_medium": reference.stats[4],
        "AP_large": reference.stats[5],
        "AR_1": reference.stats[6],
        "AR_10": reference.stats[7],
        "AR_100": reference.stats[8],
    }
    for key, value in expected.items():
        assert_close(ours[key], value, tolerance=1e-6, label=f"{key} vs pycocotools")


def test_matches_torchmetrics_map():
    """A second independent implementation of the same algorithm."""
    if not has_module("torchmetrics"):
        skip("torchmetrics is not installed")

    import torch
    from torchmetrics.detection import MeanAveragePrecision

    gt = _ground_truth(_gt(SMALL_BOX, MEDIUM_BOX, LARGE_BOX))
    detections = _dt(SMALL_BOX, MEDIUM_BOX, LARGE_BOX, score=0.9)

    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(
        [
            {
                "boxes": torch.tensor([d["bbox"] for d in detections]),
                "scores": torch.tensor([d["score"] for d in detections]),
                "labels": torch.tensor([d["label"] for d in detections]),
            }
        ],
        [
            {
                "boxes": torch.tensor([a["bbox"] for a in gt.annotations]),
                "labels": torch.tensor([a["label"] for a in gt.annotations]),
            }
        ],
    )
    reference = metric.compute()
    ours = evaluate_detections(gt, detections, _config())
    assert_close(ours["AP"], float(reference["map"]), tolerance=1e-4, label="AP")
