"""Dataset, augmentation, and batching."""

from __future__ import annotations

import json

import torch
from torchvision import tv_tensors

from _support import (
    TOY_DATASET,
    expect_error,
    scratch_dir,
    skip,
    synthetic_dataset,
)

from smalldet.config import LoaderConfig, TransformOp
from smalldet.data import (
    CocoDetectionDataset,
    area_ranges_from_percentiles,
    build_dataloader,
    build_transform,
    collate_fn,
    move_to_device,
    percentile,
    summarize_areas,
)


def _dataset(**kwargs):
    paths = synthetic_dataset()
    return CocoDetectionDataset(paths["images"], paths["train"], **kwargs)


# ---------------------------------------------------------------------- dataset


def test_target_matches_the_torchvision_detection_contract():
    dataset = _dataset()
    image, target = dataset[0]

    assert image.dtype == torch.uint8 and image.shape[0] == 3
    assert isinstance(image, tv_tensors.Image)
    # Boxes must be tv_tensors or transforms.v2 will move pixels without moving
    # the boxes with them — silently, which is the dangerous part.
    assert isinstance(target["boxes"], tv_tensors.BoundingBoxes)
    assert target["boxes"].format == tv_tensors.BoundingBoxFormat.XYXY
    assert target["boxes"].shape[1] == 4
    for key in ("labels", "image_id", "area", "iscrowd"):
        assert key in target, key
    assert target["labels"].dtype == torch.int64
    assert len(target["labels"]) == len(target["boxes"]) == len(target["area"])


def test_boxes_are_converted_from_xywh_to_xyxy():
    """COCO stores XYWH; torchvision needs XYXY. Getting this wrong produces
    boxes that look plausible and score near zero."""
    paths = synthetic_dataset()
    document = json.loads(paths["train"].read_text(encoding="utf-8"))
    first_image = document["images"][0]["id"]
    raw = next(a for a in document["annotations"] if a["image_id"] == first_image)

    dataset = CocoDetectionDataset(paths["images"], paths["train"])
    index = dataset.image_ids.index(first_image)
    _, target = dataset[index]

    x, y, w, h = raw["bbox"]
    expected = torch.tensor([x, y, x + w, y + h])
    matches = (target["boxes"] - expected).abs().sum(dim=1) < 1e-4
    assert matches.any(), "no box matched the annotation converted to XYXY"


def test_category_ids_are_remapped_to_contiguous_labels_with_background_at_zero():
    dataset = _dataset()
    assert dataset.class_names[0] == "__background__"
    assert dataset.num_classes == len(dataset.class_names)
    assert set(dataset.category_id_to_label.values()) == set(
        range(1, dataset.num_classes)
    )
    for _, target in [dataset[i] for i in range(len(dataset))]:
        if len(target["labels"]):
            assert int(target["labels"].min()) >= 1
            assert int(target["labels"].max()) < dataset.num_classes


def test_sparse_category_ids_still_map_contiguously():
    """COCO's own ids skip numbers (12, 26, 29...). The heads need 1..K."""
    directory = scratch_dir("sparse_categories")
    (directory / "images").mkdir(exist_ok=True)
    from PIL import Image

    Image.new("RGB", (32, 32)).save(directory / "images" / "a.png")
    document = {
        "images": [{"id": 5, "file_name": "a.png", "width": 32, "height": 32}],
        "annotations": [
            {"id": 1, "image_id": 5, "category_id": 90, "bbox": [1, 1, 8, 8], "area": 64},
            {"id": 2, "image_id": 5, "category_id": 12, "bbox": [12, 12, 6, 6], "area": 36},
        ],
        "categories": [{"id": 12, "name": "b"}, {"id": 90, "name": "z"}],
    }
    path = directory / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    dataset = CocoDetectionDataset(directory / "images", path)
    assert dataset.class_names == ["__background__", "b", "z"]
    assert dataset.category_id_to_label == {12: 1, 90: 2}
    _, target = dataset[0]
    assert sorted(target["labels"].tolist()) == [1, 2]


def test_degenerate_boxes_are_dropped():
    """Zero-extent boxes make the box-regression loss produce NaN."""
    directory = scratch_dir("degenerate")
    (directory / "images").mkdir(exist_ok=True)
    from PIL import Image

    Image.new("RGB", (32, 32)).save(directory / "images" / "a.png")
    document = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 0, 8], "area": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [4, 4, 6, 6], "area": 36},
        ],
        "categories": [{"id": 1, "name": "part"}],
    }
    path = directory / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    dataset = CocoDetectionDataset(directory / "images", path)
    _, target = dataset[0]
    assert len(target["boxes"]) == 1


def test_min_box_size_default_keeps_small_objects():
    """Regression guard: the default must not delete what AP_small measures."""
    permissive = _dataset(min_box_size=1.0)
    aggressive = _dataset(min_box_size=20.0)
    kept = sum(len(permissive[i][1]["boxes"]) for i in range(len(permissive)))
    culled = sum(len(aggressive[i][1]["boxes"]) for i in range(len(aggressive)))
    assert kept > culled, "the fixture needs objects below 20px to test this"


def test_malformed_annotation_file_is_rejected_clearly():
    directory = scratch_dir("malformed")
    (directory / "images").mkdir(exist_ok=True)
    path = directory / "bad.json"
    path.write_text(json.dumps({"images": [], "categories": []}), encoding="utf-8")
    expect_error(
        lambda: CocoDetectionDataset(directory / "images", path),
        ValueError,
        contains="annotations",
    )


# ------------------------------------------------------------------- transforms


def test_geometric_augmentation_moves_boxes_with_pixels():
    """The whole reason for tv_tensors. A horizontal flip with p=1.0 must
    mirror the boxes too."""
    dataset = _dataset()
    image, target = dataset[0]
    width = image.shape[-1]
    original = target["boxes"].clone()

    flip = build_transform([TransformOp(name="random_horizontal_flip", params={"p": 1.0})])
    flipped_image, flipped_target = flip(image, target)

    assert flipped_image.shape == image.shape
    expected_x1 = width - original[:, 2]
    assert torch.allclose(flipped_target["boxes"][:, 0], expected_x1, atol=1e-3)


def test_pipeline_appends_a_dtype_conversion_when_missing():
    """Detection backbones need float input; a pipeline that forgets fails deep
    inside the backbone with an error that says nothing about augmentation."""
    dataset = _dataset()
    image, target = dataset[0]
    pipeline = build_transform([TransformOp(name="random_horizontal_flip", params={"p": 0.0})])
    converted, _ = pipeline(image, target)
    assert converted.dtype == torch.float32
    assert float(converted.max()) <= 1.0


def test_unknown_transform_name_lists_the_valid_ones():
    error = expect_error(
        lambda: build_transform([TransformOp(name="mixup")]), Exception, contains="mixup"
    )
    assert "random_horizontal_flip" in str(error)


def test_transform_with_bad_params_names_the_step():
    expect_error(
        lambda: build_transform(
            [TransformOp(name="random_horizontal_flip", params={"probability": 0.5})]
        ),
        ValueError,
        contains="random_horizontal_flip",
    )


# ------------------------------------------------------------------ dataloader


def test_collate_keeps_variable_sized_batches_as_parallel_tuples():
    """The default collate calls torch.stack, which cannot work when images
    differ in size and every image has a different object count."""
    batch = [
        (torch.zeros(3, 40, 50), {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2)}),
        (torch.zeros(3, 60, 30), {"boxes": torch.zeros(5, 4), "labels": torch.zeros(5)}),
    ]
    images, targets = collate_fn(batch)
    assert isinstance(images, tuple) and len(images) == 2
    assert images[0].shape != images[1].shape
    assert len(targets[0]["boxes"]) == 2 and len(targets[1]["boxes"]) == 5


def test_dataloader_yields_the_model_input_contract():
    dataset = _dataset()
    loader = build_dataloader(dataset, LoaderConfig(batch_size=2, shuffle=False))
    images, targets = next(iter(loader))
    assert len(images) == len(targets) == 2
    assert all(image.ndim == 3 for image in images)


def test_persistent_workers_is_not_requested_without_workers():
    """torch raises rather than ignoring it, so the builder has to suppress it."""
    dataset = _dataset()
    loader = build_dataloader(
        dataset, LoaderConfig(batch_size=1, num_workers=0, persistent_workers=True)
    )
    assert loader.persistent_workers is False


def test_move_to_device_handles_lists_not_tensors():
    images = (torch.zeros(3, 8, 8), torch.zeros(3, 6, 6))
    targets = ({"boxes": torch.zeros(1, 4), "image_id": torch.tensor(3)},) * 2
    moved_images, moved_targets = move_to_device(images, targets, "cpu")
    assert isinstance(moved_images, list) and len(moved_images) == 2
    assert moved_targets[0]["boxes"].device.type == "cpu"


# ------------------------------------------------------------------------ stats


def test_percentile_matches_numpy():
    import numpy as np

    values = [1.0, 2.0, 3.0, 4.0, 10.0, 40.0, 7.5]
    for q in (0.0, 25.0, 33.3, 50.0, 90.0, 100.0):
        assert abs(percentile(values, q) - float(np.percentile(values, q))) < 1e-9


def test_auto_area_ranges_populate_every_bucket():
    """The point of auto-calibration: COCO's fixed cut-offs can leave a bucket
    empty, which makes AP_medium report -1 instead of a score."""
    dataset = _dataset()
    areas = dataset.box_areas()
    ranges = area_ranges_from_percentiles(areas, [33.3, 66.6])

    assert set(ranges) == {"all", "small", "medium", "large"}
    stats = summarize_areas(areas, ranges)
    for label in ("small", "medium", "large"):
        assert stats.bucket_fractions[label] > 0.0, f"{label} bucket is empty"


def test_area_summary_reports_bucket_occupancy():
    dataset = _dataset()
    stats = summarize_areas(
        dataset.box_areas(),
        {"all": [0.0, 1e10], "small": [0.0, 1024.0], "medium": [1024.0, 9216.0]},
    )
    assert stats.count > 0
    assert 0.0 <= stats.bucket_fractions["small"] <= 1.0
    assert "ground-truth objects" in stats.describe()


# ------------------------------------------------------------------- real data


def test_real_toy_dataset_loads():
    """Exercises the loader against genuine photographs and RLE-carrying
    annotations, which the synthetic fixture cannot cover."""
    if not TOY_DATASET.is_dir():
        skip(f"the toy dataset is not present at {TOY_DATASET}")

    dataset = CocoDetectionDataset(
        TOY_DATASET / "images", TOY_DATASET / "annotations_train.json"
    )
    assert len(dataset) > 0
    assert dataset.num_classes == 21  # 20 VOC classes plus background
    image, target = dataset[0]
    assert image.shape[0] == 3 and image.dtype == torch.uint8
    assert len(target["boxes"]) == len(target["labels"])

    # All three COCO buckets are populated here, which is what makes this the
    # useful dataset for checking that AP_large is a real number, not -1.
    stats = summarize_areas(
        dataset.box_areas(),
        {
            "all": [0.0, 1e10],
            "small": [0.0, 1024.0],
            "medium": [1024.0, 9216.0],
            "large": [9216.0, 1e10],
        },
    )
    for label in ("small", "medium", "large"):
        assert stats.bucket_fractions[label] > 0.0, f"{label} bucket is empty"
