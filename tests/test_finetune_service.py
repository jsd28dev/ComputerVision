"""The finetuning page's service: config building, validation, and streaming.

No Gradio import anywhere — this is the layer the UI is a thin shell over, so
it is where the finetuning behaviour is actually pinned down.
"""

from __future__ import annotations

import json

from _support import expect_error, scratch_dir, synthetic_dataset, tiny_config

from smalldet.app.finetune_service import (
    FinetuneService,
    history_markdown,
)
from smalldet.config import ConfigError
from smalldet.data.split import split_coco, summarize_split


def _ui(**overrides):
    paths = synthetic_dataset()
    values = dict(
        data_root=str(paths["root"]),
        train_images="images",
        train_annotations="annotations_train.json",
        val_images="images",
        val_annotations="annotations_val.json",
        architecture="fasterrcnn_resnet50_fpn_v2",
        pretrained=False,
        min_size=160,
        max_size=200,
        anchors_enabled=True,
        anchor_base_sizes="4, 8, 16, 32, 64",
        strategy="partial",
        trainable_backbone_layers=1,
        optimizer="sgd",
        learning_rate=0.005,
        scheduler="multistep",
        milestones="1",
        epochs=1,
        batch_size=2,
        monitor="AP_small",
        output_dir=str(scratch_dir("ft_service_run", clean=True)),
        device="cpu",
    )
    values.update(overrides)
    return values


def _service():
    return FinetuneService(tiny_config())


# ------------------------------------------------------------ config building


def test_ui_values_become_a_valid_config():
    config = _service().build_config(**_ui())
    assert config.finetune.strategy == "partial"
    assert config.model.anchors.base_sizes == [4, 8, 16, 32, 64]
    assert config.train.checkpoint.monitor == "AP_small"
    assert config.data.train_loader.batch_size == 2


def test_every_strategy_builds():
    service = _service()
    for strategy in service.strategies():
        config = service.build_config(**_ui(strategy=strategy))
        assert config.finetune.strategy == strategy


def test_gradual_schedule_is_parsed_into_epoch_to_layers():
    config = _service().build_config(
        **_ui(strategy="gradual", gradual_schedule="0:0, 3:2, 6:5")
    )
    assert config.finetune.gradual_schedule == {"0": 0, "3": 2, "6": 5}


def test_gradual_schedule_only_applies_to_the_gradual_strategy():
    config = _service().build_config(**_ui(strategy="partial", gradual_schedule="0:1"))
    assert config.finetune.gradual_schedule == {}


def test_malformed_schedule_is_rejected_with_the_expected_shape():
    expect_error(
        lambda: _service().build_config(**_ui(strategy="gradual", gradual_schedule="nope")),
        ValueError,
        contains="epoch:layers",
    )


def test_malformed_anchor_sizes_are_rejected():
    expect_error(
        lambda: _service().build_config(**_ui(anchor_base_sizes="8, wide, 32")),
        ValueError,
        contains="anchor base sizes",
    )


def test_pretrained_toggle_switches_between_detector_and_backbone_weights():
    service = _service()
    pretrained = service.build_config(**_ui(pretrained=True))
    assert pretrained.model.weights == "DEFAULT"
    assert pretrained.model.weights_backbone is None

    scratch = service.build_config(**_ui(pretrained=False))
    assert scratch.model.weights is None
    assert scratch.model.weights_backbone == "DEFAULT"


def test_optimizer_kwargs_match_the_optimizer():
    """Momentum is meaningless to AdamW and torch rejects the unknown kwarg."""
    service = _service()
    assert service.build_config(**_ui(optimizer="sgd")).optimizer.kwargs == {
        "momentum": 0.9
    }
    assert service.build_config(**_ui(optimizer="adamw")).optimizer.kwargs == {}


def test_detector_kwargs_match_the_architecture_family():
    """box_detections_per_img exists only on the R-CNN family."""
    service = _service()
    rcnn = service.build_config(**_ui(architecture="fasterrcnn_resnet50_fpn_v2"))
    assert "box_detections_per_img" in rcnn.model.kwargs

    retina = service.build_config(
        **_ui(architecture="retinanet_resnet50_fpn_v2", anchors_enabled=False)
    )
    assert "box_detections_per_img" not in retina.model.kwargs


def test_retinanet_gets_three_scales_per_octave():
    config = _service().build_config(**_ui(architecture="retinanet_resnet50_fpn_v2"))
    assert config.model.anchors.scales_per_octave == 3


def test_augmentation_never_offers_zoom_out_and_keeps_sanitize_permissive():
    """Both would delete or shrink exactly the objects AP_small measures."""
    config = _service().build_config(**_ui())
    names = [op.name for op in config.data.augmentation.train]
    assert "random_zoom_out" not in names
    assert names[-2:] == ["to_dtype", "to_pure_tensor"]

    sanitize = next(op for op in config.data.augmentation.train if op.name == "sanitize_bounding_boxes")
    assert sanitize.params["min_size"] == 1.0

    jitter = next(op for op in config.data.augmentation.train if op.name == "scale_jitter")
    assert jitter.params["scale_range"][0] >= 1.0  # never shrinks


def test_invalid_config_surfaces_as_a_config_error():
    expect_error(
        lambda: _service().build_config(**_ui(min_size=900, max_size=100)),
        ConfigError,
        contains="max_size",
    )


# ------------------------------------------------------------------ validation


def test_describe_reports_the_plan_without_training():
    text = _service().describe(**_ui())
    assert "Ready to train" in text
    assert "partial" in text
    assert "Anchors" in text
    assert "train, val" in text


def test_describe_returns_the_error_instead_of_raising():
    """A misconfiguration must show up in the UI panel, not crash the page."""
    text = _service().describe(**_ui(anchor_base_sizes="not-a-list"))
    assert "Configuration error" in text


def test_describe_warns_when_selecting_ap_small_with_default_anchors():
    """The exact trap this project exists to avoid."""
    text = _service().describe(**_ui(anchors_enabled=False, monitor="AP_small"))
    assert "invisible to the model" in text


def test_describe_warns_about_a_diverging_learning_rate_and_missing_warmup():
    text = _service().describe(**_ui(learning_rate=0.5, warmup_enabled=False))
    assert "high for SGD" in text
    assert "Warmup is off" in text


def test_describe_warns_when_no_milestone_falls_inside_the_run():
    text = _service().describe(**_ui(epochs=5, milestones="16, 22"))
    assert "never decays" in text


def test_describe_flags_a_missing_validation_split():
    text = _service().describe(**_ui(val_images="", val_annotations=""))
    assert "nothing is scored" in text


# --------------------------------------------------------------------- export


def test_exported_config_reloads_and_matches():
    """The UI must produce a config the CLI can rerun verbatim."""
    from smalldet.config import load_config

    service = _service()
    path = scratch_dir("ft_export", clean=True) / "config.yaml"
    service.export_config(path, **_ui())

    assert path.is_file()
    reloaded = load_config(path)
    assert reloaded == service.build_config(**_ui())


# ---------------------------------------------------------------------- splits


def test_split_partitions_images_without_leaking_between_splits():
    paths = synthetic_dataset()
    directory = scratch_dir("split_leak", clean=True)
    written = split_coco(paths["train"], directory, [0.6, 0.2, 0.2], seed=1)

    assert set(written) == {"train", "val", "test"}
    seen = {}
    for name, path in written.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        ids = {record["id"] for record in document["images"]}
        for other, other_ids in seen.items():
            assert not (ids & other_ids), f"{name} and {other} share images"
        seen[name] = ids

        # Every annotation must belong to an image in the same file.
        for annotation in document["annotations"]:
            assert annotation["image_id"] in ids

    source = json.loads(paths["train"].read_text(encoding="utf-8"))
    assert sum(len(ids) for ids in seen.values()) == len(source["images"])


def test_split_copies_the_category_list_verbatim_into_every_file():
    """Otherwise each split would infer its own labels and they would disagree."""
    paths = synthetic_dataset()
    directory = scratch_dir("split_categories", clean=True)
    written = split_coco(paths["train"], directory, [0.5, 0.25, 0.25], seed=2)

    source = json.loads(paths["train"].read_text(encoding="utf-8"))["categories"]
    for path in written.values():
        assert json.loads(path.read_text(encoding="utf-8"))["categories"] == source


def test_a_zero_ratio_skips_that_split():
    paths = synthetic_dataset()
    directory = scratch_dir("split_two_way", clean=True)
    written = split_coco(paths["train"], directory, [0.8, 0.2, 0.0], seed=3)
    assert set(written) == {"train", "val"}


def test_split_rejects_nonsense_ratios():
    paths = synthetic_dataset()
    expect_error(
        lambda: split_coco(paths["train"], scratch_dir("split_bad"), [0, 0, 0]),
        ValueError,
        contains="greater than zero",
    )


def test_split_summary_is_readable():
    paths = synthetic_dataset()
    written = split_coco(
        paths["train"], scratch_dir("split_summary", clean=True), [0.6, 0.4, 0.0], seed=4
    )
    text = summarize_split(written)
    assert "train:" in text and "images" in text


# -------------------------------------------------------------------- training


def test_run_streams_progress_and_finishes_cleanly():
    """The Start button's function, end to end.

    Guards the status value specifically: reporting "stopped" for a run that
    completed normally would tell the user their training was cut short.
    """
    updates = list(_service().run(**_ui()))

    assert updates, "the generator yielded nothing"
    final = updates[-1]
    assert final.finished is True
    assert final.failed is False
    assert final.status == "finished", f"expected 'finished', got {final.status!r}"
    assert final.checkpoint is not None
    assert final.history and "AP_small" in final.history[0]
    assert "Training finished" in final.log


def test_run_reports_a_configuration_error_instead_of_raising():
    updates = list(_service().run(**_ui(anchor_base_sizes="bad")))
    assert len(updates) == 1
    assert updates[0].failed is True
    assert "Configuration error" in updates[0].log


def test_stop_requested_during_a_run_is_reported_as_stopped():
    """The Stop button's path. Requested mid-run, because run() deliberately
    clears a stale request at start — a stop aimed at a previous run must not
    kill the next one."""
    service = _service()
    updates = []
    for progress in service.run(**_ui(epochs=3)):
        updates.append(progress)
        if len(updates) == 2:
            service.request_stop()
    assert updates[-1].finished is True
    assert updates[-1].status == "stopped"


def test_a_stale_stop_request_does_not_kill_the_next_run():
    service = _service()
    service.request_stop()  # nothing is running; this must be discarded
    final = list(service.run(**_ui()))[-1]
    assert final.status == "finished"


def test_history_markdown_renders_a_table_with_the_sentinel_as_na():
    text = history_markdown(
        [{"epoch": 1, "AP": 0.25, "AP_small": 0.1, "AP_large": -1.0}]
    )
    assert "| epoch |" in text
    assert "0.2500" in text
    assert "n/a" in text


def test_history_markdown_handles_an_empty_history():
    assert "No epoch" in history_markdown([])
