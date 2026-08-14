"""Postprocessing, tiled inference, and the Predictor facade."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from _support import expect_error, synthetic_dataset, tiny_config

from smalldet.config import PostprocessConfig, TilingConfig
from smalldet.inference import (
    Predictor,
    apply_postprocess,
    generate_tiles,
    merge_tile_predictions,
    to_uint8_tensor,
)


def _prediction(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def assert_scores(actual: torch.Tensor, expected: list) -> None:
    """Compare score tensors with a tolerance.

    float32 cannot represent 0.9 exactly, so ``tensor.tolist() == [0.9]`` is
    false even when the value is correct.
    """
    values = actual.tolist()
    assert len(values) == len(expected), f"expected {expected}, got {values}"
    for got, want in zip(values, expected):
        assert abs(got - want) < 1e-5, f"expected {expected}, got {values}"


# ------------------------------------------------------------- postprocessing


def test_score_threshold_filters_and_preserves_ordering():
    prediction = _prediction(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        [0.9, 0.3, 0.6],
        [1, 2, 3],
    )
    kept = apply_postprocess(prediction, PostprocessConfig(score_threshold=0.5))
    assert_scores(kept["scores"], [0.9, 0.6])  # sorted, low one dropped
    assert kept["labels"].tolist() == [1, 3]


def test_max_detections_keeps_the_most_confident():
    prediction = _prediction(
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], [0.5, 0.9, 0.7], [1, 1, 1]
    )
    kept = apply_postprocess(
        prediction, PostprocessConfig(score_threshold=0.0, max_detections=2)
    )
    assert_scores(kept["scores"], [0.9, 0.7])


def test_min_box_area_defaults_to_keeping_everything():
    """A non-zero default here would silently delete small objects, which is
    the opposite of this project's purpose."""
    assert PostprocessConfig().min_box_area == 0.0
    tiny = _prediction([[0, 0, 3, 3]], [0.9], [1])
    kept = apply_postprocess(tiny, PostprocessConfig(score_threshold=0.5))
    assert len(kept["boxes"]) == 1


def test_min_box_area_filters_when_asked():
    prediction = _prediction([[0, 0, 3, 3], [0, 0, 40, 40]], [0.9, 0.9], [1, 1])
    kept = apply_postprocess(
        prediction, PostprocessConfig(score_threshold=0.0, min_box_area=100.0)
    )
    assert len(kept["boxes"]) == 1


def test_allowed_labels_restricts_output():
    prediction = _prediction(
        [[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [1, 2]
    )
    kept = apply_postprocess(
        prediction, PostprocessConfig(score_threshold=0.0, allowed_labels=[2])
    )
    assert kept["labels"].tolist() == [2]


def test_class_aware_nms_keeps_overlapping_boxes_of_different_classes():
    """Two different parts genuinely overlapping is a real configuration;
    class-agnostic NMS would silently drop one of them."""
    prediction = _prediction(
        [[0, 0, 20, 20], [1, 1, 21, 21]], [0.9, 0.8], [1, 2]
    )
    class_aware = apply_postprocess(
        prediction,
        PostprocessConfig(score_threshold=0.0, nms_iou_threshold=0.5),
    )
    assert len(class_aware["boxes"]) == 2

    agnostic = apply_postprocess(
        prediction,
        PostprocessConfig(
            score_threshold=0.0, nms_iou_threshold=0.5, class_agnostic_nms=True
        ),
    )
    assert len(agnostic["boxes"]) == 1


def test_nms_suppresses_duplicates_of_the_same_class():
    prediction = _prediction([[0, 0, 20, 20], [1, 1, 21, 21]], [0.9, 0.8], [1, 1])
    kept = apply_postprocess(
        prediction, PostprocessConfig(score_threshold=0.0, nms_iou_threshold=0.5)
    )
    assert len(kept["boxes"]) == 1


def test_masks_survive_filtering():
    prediction = _prediction([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.1], [1, 1])
    prediction["masks"] = torch.zeros(2, 1, 32, 32)
    prediction["masks"][0] = 1.0
    kept = apply_postprocess(prediction, PostprocessConfig(score_threshold=0.5))
    assert kept["masks"].shape[0] == 1
    assert float(kept["masks"].sum()) > 0


# -------------------------------------------------------------------- tiling


def test_tiles_cover_the_whole_frame():
    tiles = generate_tiles(1000, 700, [512, 512], overlap=0.2, include_full_image=False)
    assert tiles
    assert min(x0 for x0, _, _, _ in tiles) == 0
    assert max(x1 for _, _, x1, _ in tiles) == 1000
    assert max(y1 for _, _, _, y1 in tiles) == 700


def test_tiles_overlap_by_the_configured_fraction():
    """Overlap is what makes tiling correct rather than merely faster-looking:
    an object straddling a boundary must lie wholly inside some tile."""
    tiles = generate_tiles(1000, 512, [512, 512], overlap=0.25, include_full_image=False)
    xs = sorted({x0 for x0, _, _, _ in tiles})
    assert len(xs) >= 2
    stride = xs[1] - xs[0]
    assert stride == 384  # 512 * (1 - 0.25)


def test_a_frame_smaller_than_one_tile_is_not_tiled():
    """Tiling a small frame would be a slower no-op."""
    assert generate_tiles(300, 200, [512, 512], 0.2) == [(0, 0, 300, 200)]


def test_full_image_pass_is_included_when_asked():
    """Catches objects larger than one tile, which no crop can contain."""
    with_full = generate_tiles(1000, 700, [512, 512], 0.2, include_full_image=True)
    without = generate_tiles(1000, 700, [512, 512], 0.2, include_full_image=False)
    assert len(with_full) == len(without) + 1
    assert (0, 0, 1000, 700) in with_full


def test_invalid_overlap_is_rejected():
    expect_error(
        lambda: generate_tiles(100, 100, [32, 32], overlap=1.0),
        ValueError,
        contains="overlap",
    )


def test_merging_shifts_tile_boxes_into_frame_coordinates():
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    predictions = [
        _prediction([[10, 10, 30, 30]], [0.9], [1]),
        _prediction([[5, 5, 25, 25]], [0.8], [1]),
    ]
    merged = merge_tile_predictions(predictions, tiles, iou_threshold=0.5)
    assert merged["boxes"].tolist() == [[10, 10, 30, 30], [105, 5, 125, 25]]


def test_merging_deduplicates_the_overlap_band():
    """The same object seen in two overlapping tiles must come back once."""
    tiles = [(0, 0, 100, 100), (50, 0, 150, 100)]
    predictions = [
        _prediction([[60, 20, 80, 40]], [0.9], [1]),  # frame coords 60..80
        _prediction([[10, 20, 30, 40]], [0.85], [1]),  # frame coords 60..80 too
    ]
    merged = merge_tile_predictions(predictions, tiles, iou_threshold=0.5)
    assert len(merged["boxes"]) == 1
    assert_scores(merged["scores"], [0.9])  # the more confident one survives


def test_merging_rejects_a_length_mismatch():
    expect_error(
        lambda: merge_tile_predictions([_prediction([[0, 0, 1, 1]], [0.5], [1])], [], 0.5),
        ValueError,
        contains="tile",
    )


# ------------------------------------------------------------ image conversion


def test_every_supported_input_type_becomes_one_canonical_tensor():
    """Gradio hands over NumPy, the CLI hands over paths, tests hand over
    tensors. They must all land in the same uint8 (3, H, W) form."""
    paths = synthetic_dataset()
    path = sorted(paths["images"].glob("*.png"))[0]

    from_path = to_uint8_tensor(path)
    from_pil = to_uint8_tensor(PILImage.open(path))
    from_numpy = to_uint8_tensor(np.array(PILImage.open(path).convert("RGB")))
    from_tensor = to_uint8_tensor(from_path.clone())

    for tensor in (from_path, from_pil, from_numpy, from_tensor):
        assert tensor.dtype == torch.uint8
        assert tensor.shape == from_path.shape
        assert tensor.shape[0] == 3
    assert torch.equal(from_path, from_numpy)


def test_float_images_are_scaled_by_range_not_guessed_by_dtype():
    unit = to_uint8_tensor(torch.ones(3, 4, 4, dtype=torch.float32))  # [0,1]
    assert int(unit.max()) == 255

    byte_range = to_uint8_tensor(torch.full((3, 4, 4), 200.0))  # [0,255] float
    assert int(byte_range.max()) == 200


def test_grayscale_and_rgba_become_rgb():
    assert to_uint8_tensor(torch.zeros(1, 8, 8, dtype=torch.uint8)).shape[0] == 3
    assert to_uint8_tensor(torch.zeros(4, 8, 8, dtype=torch.uint8)).shape[0] == 3


# ------------------------------------------------------------------ predictor


def test_predictor_runs_end_to_end_and_reports_size_buckets():
    config = tiny_config()
    from smalldet.models import build_model

    model = build_model(config.model, 4)
    predictor = Predictor(
        model, ["__background__", "screw", "washer", "nut"], config.predict
    )

    paths = synthetic_dataset()
    result = predictor.predict(sorted(paths["images"].glob("*.png"))[0])

    assert result.image is not None and result.image.dtype == torch.uint8
    assert len(result.boxes) == len(result.scores) == len(result.labels)
    buckets = result.size_histogram()
    assert set(buckets) == {"small", "medium", "large"}
    assert sum(buckets.values()) == len(result)
    for record in result.to_records():
        assert set(record) == {"label", "class_name", "score", "box_xyxy", "area"}


def test_tiling_runs_more_forward_passes_than_the_plain_path():
    import dataclasses

    config = tiny_config()
    from smalldet.models import build_model

    model = build_model(config.model, 4)
    predict_config = dataclasses.replace(
        config.predict,
        tiling=TilingConfig(enabled=True, tile_size=[64, 64], overlap=0.25),
        postprocess=PostprocessConfig(score_threshold=0.0, max_detections=10),
    )
    predictor = Predictor(model, ["__background__", "a", "b", "c"], predict_config)

    paths = synthetic_dataset()
    image = sorted(paths["images"].glob("*.png"))[0]

    tiled = predictor.predict(image)
    plain = predictor.predict(image, tiling=False)
    assert tiled.num_tiles > 1
    assert plain.num_tiles == 1


def test_runtime_overrides_do_not_mutate_the_configured_defaults():
    """The Gradio sliders override per call; the config must stay untouched so
    the next request starts from the same place."""
    config = tiny_config()
    from smalldet.models import build_model

    predictor = Predictor(
        build_model(config.model, 4), ["__background__", "a", "b", "c"], config.predict
    )
    original = predictor.config.postprocess.score_threshold

    paths = synthetic_dataset()
    predictor.predict(
        sorted(paths["images"].glob("*.png"))[0],
        postprocess=PostprocessConfig(score_threshold=0.99),
    )
    assert predictor.config.postprocess.score_threshold == original
