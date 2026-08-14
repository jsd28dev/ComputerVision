"""Detector construction, head retargeting, and the small-object anchor pyramid."""

from __future__ import annotations

import torch

from _support import expect_error, scratch_dir

from smalldet.config import AnchorConfig, ModelConfig
from smalldet.models import (
    available_architectures,
    build_anchor_generator,
    build_model,
    load_checkpoint,
    pyramid_sizes,
)

NUM_CLASSES = 4  # background + 3


def _config(**kwargs) -> ModelConfig:
    defaults = dict(
        architecture="fasterrcnn_resnet50_fpn_v2",
        weights=None,
        weights_backbone=None,
        min_size=128,
        max_size=160,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


# ------------------------------------------------------------------- anchors


def test_pyramid_sizes_one_scale_per_level():
    assert pyramid_sizes([8, 16, 32], 1) == ((8,), (16,), (32,))


def test_pyramid_sizes_three_scales_per_octave():
    """RetinaNet's convention: 2^0, 2^(1/3), 2^(2/3) multiples of the base."""
    sizes = pyramid_sizes([32], 3)
    assert sizes == ((32, 40, 51),)


def test_anchor_generator_rejects_a_level_count_mismatch():
    """A silent mismatch here surfaces as an opaque shape error inside the RPN."""
    error = expect_error(
        lambda: build_anchor_generator(
            AnchorConfig(enabled=True, base_sizes=[8, 16, 32]), num_levels=5
        ),
        ValueError,
        contains="3 entries",
    )
    assert "5 feature map" in str(error)


def test_small_anchors_are_actually_installed_on_the_rpn():
    """The single highest-leverage change for AP_small: an object smaller than
    the smallest anchor can never clear the RPN's IoU threshold."""
    model = build_model(
        _config(anchors=AnchorConfig(enabled=True, base_sizes=[4, 8, 16, 32, 64])),
        NUM_CLASSES,
    )
    assert model.rpn.anchor_generator.sizes == ((4,), (8,), (16,), (32,), (64,))


def test_default_anchors_are_left_alone_when_disabled():
    model = build_model(_config(anchors=AnchorConfig(enabled=False)), NUM_CLASSES)
    assert model.rpn.anchor_generator.sizes[0] == (32,)


def test_anchor_count_change_rebuilds_the_rpn_head():
    """The RPN head's output convolutions are sized by anchors-per-location, so
    a different count invalidates the pretrained head and it must be rebuilt."""
    baseline = build_model(_config(anchors=AnchorConfig(enabled=False)), NUM_CLASSES)
    original = baseline.rpn.head.cls_logits.out_channels

    model = build_model(
        _config(
            anchors=AnchorConfig(
                enabled=True,
                base_sizes=[4, 8, 16, 32, 64],
                aspect_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            )
        ),
        NUM_CLASSES,
    )
    assert model.rpn.head.cls_logits.out_channels == 5
    assert model.rpn.head.cls_logits.out_channels != original
    # It still runs, which is the point of rebuilding rather than erroring.
    model.eval()
    with torch.inference_mode():
        model([torch.rand(3, 128, 128)])


def test_same_anchor_count_preserves_the_rpn_head_shape():
    """Changing only the scales keeps 3 anchors per location, so a pretrained
    RPN head still fits — which is why base_sizes is the recommended lever."""
    model = build_model(
        _config(anchors=AnchorConfig(enabled=True, base_sizes=[4, 8, 16, 32, 64])),
        NUM_CLASSES,
    )
    assert model.rpn.head.cls_logits.out_channels == 3


# -------------------------------------------------------------------- families


def test_every_supported_family_builds_and_runs_both_modes():
    """Train mode must return a loss dict, eval mode a prediction list. This is
    the contract the whole trainer and evaluator are built on."""
    cases = [
        ("fasterrcnn_resnet50_fpn_v2", AnchorConfig(enabled=True, base_sizes=[4, 8, 16, 32, 64])),
        ("retinanet_resnet50_fpn_v2", AnchorConfig(enabled=True, base_sizes=[8, 16, 32, 64, 128], scales_per_octave=3)),
        ("fcos_resnet50_fpn", AnchorConfig(enabled=False)),
        ("maskrcnn_resnet50_fpn_v2", AnchorConfig(enabled=True, base_sizes=[4, 8, 16, 32, 64])),
    ]
    images = [torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
        }
    ]

    for architecture, anchors in cases:
        model = build_model(_config(architecture=architecture, anchors=anchors), NUM_CLASSES)

        model.train()
        losses = model(images, targets)
        assert isinstance(losses, dict) and losses, architecture
        assert all(torch.isfinite(value) for value in losses.values()), architecture

        model.eval()
        with torch.inference_mode():
            outputs = model(images)
        assert len(outputs) == 1, architecture
        for key in ("boxes", "labels", "scores"):
            assert key in outputs[0], f"{architecture} is missing {key}"


def test_head_is_retargeted_to_the_dataset_class_count():
    model = build_model(_config(), NUM_CLASSES)
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES


def test_mask_head_is_retargeted_too():
    model = build_model(_config(architecture="maskrcnn_resnet50_fpn_v2"), NUM_CLASSES)
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == NUM_CLASSES


def test_retinanet_head_reflects_both_class_count_and_anchor_count():
    """RetinaNet fuses num_classes and num_anchors into one output conv, so
    they have to be applied together rather than in two passes."""
    anchors = AnchorConfig(
        enabled=True, base_sizes=[8, 16, 32, 64, 128], scales_per_octave=3
    )
    model = build_model(
        _config(architecture="retinanet_resnet50_fpn_v2", anchors=anchors), NUM_CLASSES
    )
    per_location = model.anchor_generator.num_anchors_per_location()[0]
    assert per_location == 9  # 3 scales x 3 aspect ratios
    assert model.head.classification_head.num_classes == NUM_CLASSES
    assert model.head.regression_head.bbox_reg.out_channels == per_location * 4


def test_anchor_free_architecture_rejects_anchor_config_instead_of_ignoring_it():
    error = expect_error(
        lambda: build_model(
            _config(
                architecture="fcos_resnet50_fpn",
                anchors=AnchorConfig(enabled=True, base_sizes=[4, 8, 16, 32, 64]),
            ),
            NUM_CLASSES,
        ),
        ValueError,
        contains="anchor-free",
    )
    assert "fcos" in str(error).lower()


def test_ssd_explains_the_supported_path_rather_than_failing_obscurely():
    error = expect_error(
        lambda: build_model(
            _config(architecture="ssd300_vgg16", weights="DEFAULT"), NUM_CLASSES
        ),
        ValueError,
        contains="weights_backbone",
    )
    assert "null" in str(error)


def test_unknown_architecture_lists_the_available_ones():
    error = expect_error(
        lambda: build_model(_config(architecture="yolov8"), NUM_CLASSES),
        ValueError,
        contains="yolov8",
    )
    assert "fasterrcnn_resnet50_fpn_v2" in str(error)
    assert len(available_architectures()) >= 8


def test_num_classes_below_two_is_rejected():
    expect_error(
        lambda: build_model(_config(), 1), ValueError, contains="background"
    )


def test_input_resolution_is_applied_to_the_internal_transform():
    """min_size is the second-biggest lever on AP_small, so it must actually
    reach the model's resize rather than being silently dropped."""
    model = build_model(_config(min_size=1024, max_size=1333), NUM_CLASSES)
    assert 1024 in tuple(model.transform.min_size)
    assert model.transform.max_size == 1333


def test_extra_kwargs_reach_the_constructor():
    model = build_model(
        _config(kwargs={"box_detections_per_img": 300, "box_score_thresh": 0.01}),
        NUM_CLASSES,
    )
    assert model.roi_heads.detections_per_img == 300
    assert model.roi_heads.score_thresh == 0.01


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trips():
    model = build_model(_config(), NUM_CLASSES)
    path = scratch_dir("checkpoints") / "model.pt"
    torch.save({"model": model.state_dict(), "epoch": 2}, path)

    restored = build_model(_config(), NUM_CLASSES)
    load_checkpoint(restored, str(path))
    for (name, a), (_, b) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        assert torch.equal(a, b), name


def test_mismatched_checkpoint_explains_the_likely_cause():
    model = build_model(_config(), NUM_CLASSES)
    path = scratch_dir("checkpoints") / "wrong_classes.pt"
    torch.save({"model": model.state_dict()}, path)

    other = build_model(_config(), NUM_CLASSES + 3)
    error = expect_error(
        lambda: load_checkpoint(other, str(path)), RuntimeError, contains="class count"
    )
    assert "anchor" in str(error)
