"""Configuration binding, composition, overrides, and validation."""

from __future__ import annotations

from pathlib import Path

from _support import PROJECT_ROOT, expect_error, scratch_dir

from smalldet.config import (
    Config,
    ConfigError,
    apply_overrides,
    config_from_dict,
    deep_merge,
    dump_config,
    load_config,
    load_raw,
)


def test_defaults_target_small_objects():
    """The out-of-the-box config is set up for the metric this project cares about."""
    config = Config()
    assert config.train.checkpoint.monitor == "AP_small"
    assert config.eval.primary_metric == "AP_small"
    assert "small" in config.eval.area_ranges
    assert "medium" in config.eval.area_ranges
    assert config.visualize.highlight_small_objects is True


def test_nested_binding_and_type_coercion():
    config = config_from_dict(
        {
            "model": {"min_size": "512", "anchors": {"enabled": True, "base_sizes": [4, 8, 16, 32, 64]}},
            "optimizer": {"lr": "0.001"},
            "train": {"amp": "false"},
        }
    )
    # Strings coerce, because CLI overrides and YAML 1.1 both deliver them.
    assert config.model.min_size == 512 and isinstance(config.model.min_size, int)
    assert config.optimizer.lr == 0.001 and isinstance(config.optimizer.lr, float)
    assert config.train.amp is False
    assert config.model.anchors.base_sizes == [4, 8, 16, 32, 64]


def test_unknown_key_is_an_error_not_a_silent_default():
    """The single most valuable strictness rule: a typo must not cost a GPU-day."""
    error = expect_error(
        lambda: config_from_dict({"data": {"train_loader": {"bacth_size": 8}}}),
        ConfigError,
        contains="bacth_size",
    )
    # The message must also say what WAS valid, or it is not actionable.
    assert "batch_size" in str(error)


def test_scalar_type_errors_name_their_path():
    error = expect_error(
        lambda: config_from_dict({"model": {"anchors": {"base_sizes": [8, "wide", 32]}}}),
        ConfigError,
    )
    assert "base_sizes[1]" in str(error)


def test_bool_is_not_accepted_where_an_int_is_expected():
    expect_error(
        lambda: config_from_dict({"train": {"epochs": True}}),
        ConfigError,
        contains="boolean",
    )


def test_deep_merge_replaces_lists_wholesale():
    """Lists replace rather than concatenate, so a derived config can shorten
    an augmentation pipeline instead of only ever growing it."""
    merged = deep_merge(
        {"data": {"augmentation": {"train": [{"name": "a"}, {"name": "b"}]}, "root": "."}},
        {"data": {"augmentation": {"train": [{"name": "c"}]}}},
    )
    assert merged["data"]["augmentation"]["train"] == [{"name": "c"}]
    assert merged["data"]["root"] == "."  # untouched keys survive


def test_dotted_overrides_parse_yaml_values():
    raw = apply_overrides(
        {"train": {"epochs": 20}},
        [
            "train.epochs=3",
            "model.weights=null",
            "eval.max_dets=[1,10,300]",
            "model.anchors.enabled=true",
        ],
    )
    assert raw["train"]["epochs"] == 3
    assert raw["model"]["weights"] is None
    assert raw["eval"]["max_dets"] == [1, 10, 300]
    assert raw["model"]["anchors"]["enabled"] is True


def test_malformed_override_is_rejected():
    expect_error(
        lambda: apply_overrides({}, ["train.epochs"]),
        ConfigError,
        contains="dotted.key=value",
    )


def test_shipped_configs_all_load():
    """Every config in configs/ must bind and validate."""
    configs = sorted((PROJECT_ROOT / "configs").glob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        config = load_config(path)
        assert isinstance(config, Config), path.name


def test_base_composition_inherits_and_overrides():
    """predict.yaml inherits its model definition from the training config.

    That inheritance is what stops the architecture, anchors, and class count
    from drifting away from the checkpoint being loaded.
    """
    training = load_config(PROJECT_ROOT / "configs" / "finetune_synthetic.yaml")
    predict = load_config(PROJECT_ROOT / "configs" / "predict.yaml")

    assert predict.model.architecture == training.model.architecture
    assert predict.model.anchors.base_sizes == training.model.anchors.base_sizes
    # ... while the predict-specific settings do differ.
    assert predict.predict.postprocess.nms_iou_threshold == 0.45
    assert predict.model.checkpoint is not None


def test_base_chain_is_transitive():
    """app.yaml -> predict.yaml -> finetune_synthetic.yaml -> base.yaml."""
    app = load_config(PROJECT_ROOT / "configs" / "app.yaml")
    assert app.model.anchors.base_sizes == [8, 16, 32, 64, 128]  # from base
    assert app.data.root == "Dataset/synthetic"  # from finetune_synthetic
    assert app.predict.postprocess.nms_iou_threshold == 0.45  # from predict
    assert app.app.title.startswith("smalldet")  # from app itself


def test_circular_base_is_detected():
    directory = scratch_dir("config_cycle")
    (directory / "a.yaml").write_text("_base_: b.yaml\nname: a\n", encoding="utf-8")
    (directory / "b.yaml").write_text("_base_: a.yaml\nname: b\n", encoding="utf-8")
    expect_error(
        lambda: load_raw(directory / "a.yaml"), ConfigError, contains="circular"
    )


def test_validation_rejects_impossible_combinations():
    cases = [
        ({"model": {"num_classes": 1}}, "background"),
        ({"model": {"min_size": 900, "max_size": 400}}, "max_size"),
        ({"train": {"epochs": 0}}, "epochs"),
        ({"optimizer": {"lr": 0.0}}, "lr"),
        ({"predict": {"tiling": {"enabled": True, "overlap": 1.0}}}, "overlap"),
        ({"app": {"server_port": 99999}}, "port"),
    ]
    for payload, expected in cases:
        expect_error(lambda p=payload: config_from_dict(p), ConfigError, contains=expected)


def test_area_ranges_must_include_all_and_be_ordered():
    expect_error(
        lambda: config_from_dict({"eval": {"area_ranges": {"small": [0.0, 1024.0]}}}),
        ConfigError,
        contains="'all'",
    )
    expect_error(
        lambda: config_from_dict(
            {"eval": {"area_ranges": {"all": [0.0, 1e10], "small": [2048.0, 1024.0]}}}
        ),
        ConfigError,
        contains="must exceed",
    )


def test_checkpoint_monitor_must_be_a_metric_that_exists():
    """Guards the class of bug where a run trains for a day and never saves a
    best checkpoint because the monitor key is never emitted."""
    error = expect_error(
        lambda: config_from_dict({"train": {"checkpoint": {"monitor": "mAP"}}}),
        ConfigError,
        contains="mAP",
    )
    assert "AP_small" in str(error)  # the message lists what IS available


def test_monitor_follows_custom_area_range_labels():
    """Renaming a bucket must make the matching metric name valid."""
    config = config_from_dict(
        {
            "eval": {
                "area_ranges": {
                    "all": [0.0, 1.0e10],
                    "tiny": [0.0, 256.0],
                    "small": [256.0, 1024.0],
                },
                "primary_metric": "AP_tiny",
            },
            "train": {"checkpoint": {"monitor": "AP_tiny"}},
        }
    )
    assert config.train.checkpoint.monitor == "AP_tiny"


def test_round_trip_through_yaml_is_stable():
    original = load_config(PROJECT_ROOT / "configs" / "finetune_synthetic.yaml")
    path = Path(scratch_dir("config_roundtrip")) / "dumped.yaml"
    dump_config(original, path)
    reloaded = load_config(path)
    assert reloaded == original


def test_missing_file_reports_the_path():
    expect_error(
        lambda: load_config(PROJECT_ROOT / "configs" / "does_not_exist.yaml"),
        ConfigError,
        contains="not found",
    )
