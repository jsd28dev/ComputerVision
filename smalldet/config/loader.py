"""Load, compose, override, and validate YAML configuration documents.

Composition works like mmdetection's: a document may declare ``_base_`` (a path
or list of paths, relative to the file itself) and the bases are deep-merged
underneath it. That is what lets ``configs/predict/default.yaml`` inherit the
model definition from the training config it must stay consistent with, rather
than duplicating it and drifting.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .binding import ConfigError, from_dict, to_dict
from .schema import Config

BASE_KEY = "_base_"


# ------------------------------------------------------------------ public API


def load_config(
    path: os.PathLike | str,
    overrides: Optional[Sequence[str]] = None,
    *,
    validate: bool = True,
) -> Config:
    """Read ``path``, resolve ``_base_``, apply ``overrides``, and bind."""
    raw = load_raw(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    config = from_dict(Config, raw, path=Path(path).name)
    if validate:
        validate_config(config)
    return config


def config_from_dict(data: Mapping[str, Any], *, validate: bool = True) -> Config:
    """Bind an in-memory mapping. Used by tests and by the Gradio app."""
    config = from_dict(Config, data)
    if validate:
        validate_config(config)
    return config


def dump_config(config: Config, path: Optional[os.PathLike | str] = None) -> str:
    """Serialise a config back to YAML, for logging next to a checkpoint."""
    text = yaml.safe_dump(to_dict(config), sort_keys=False, default_flow_style=False)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text


# -------------------------------------------------------------- reading + merge


def load_raw(path: os.PathLike | str, _seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Parse a YAML file and splice in its ``_base_`` documents."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ConfigError(f"config file not found: {file_path}")

    seen = list(_seen or [])
    if file_path in seen:
        chain = " -> ".join(p.name for p in [*seen, file_path])
        raise ConfigError(f"circular {BASE_KEY} chain: {chain}")
    seen.append(file_path)

    try:
        document = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file_path}: invalid YAML: {exc}") from exc

    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise ConfigError(f"{file_path}: top level must be a mapping")

    document = dict(document)
    bases = document.pop(BASE_KEY, None)
    if bases is None:
        return document

    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"{file_path}: {BASE_KEY} must be a string or list of strings")

    merged: Dict[str, Any] = {}
    for base in bases:
        merged = deep_merge(merged, load_raw(file_path.parent / base, seen))
    return deep_merge(merged, document)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base``.

    Mappings merge key-by-key; every other type (including lists) replaces
    wholesale. Element-wise list merging would make it impossible to shorten an
    augmentation pipeline in a derived config.
    """
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(
    raw: Mapping[str, Any], overrides: Iterable[str]
) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` strings, as passed by ``--set`` on the CLI.

    Values go through the YAML scalar parser, so ``--set train.epochs=3``,
    ``--set model.weights=null`` and ``--set eval.max_dets=[1,10,300]`` all do
    what they look like they do.
    """
    result = copy.deepcopy(dict(raw))
    for override in overrides:
        if "=" not in override:
            raise ConfigError(
                f"override {override!r} is not of the form dotted.key=value"
            )
        dotted, _, literal = override.partition("=")
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ConfigError(f"override {override!r} has an empty key")
        try:
            value = yaml.safe_load(literal)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {override!r}: invalid value: {exc}") from exc

        cursor: Dict[str, Any] = result
        for key in keys[:-1]:
            node = cursor.get(key)
            if not isinstance(node, dict):
                node = {}
                cursor[key] = node
            cursor = node
        cursor[keys[-1]] = value
    return result


# ------------------------------------------------------------------- validation


def validate_config(config: Config) -> Config:
    """Cross-section checks that the per-field types cannot express."""
    errors: List[str] = []

    if config.model.num_classes is not None and config.model.num_classes < 2:
        errors.append(
            "model.num_classes counts background, so it must be >= 2 "
            f"(got {config.model.num_classes})"
        )
    if config.model.min_size > config.model.max_size:
        errors.append(
            f"model.min_size ({config.model.min_size}) exceeds "
            f"model.max_size ({config.model.max_size})"
        )
    if config.model.anchors.enabled:
        if not config.model.anchors.base_sizes:
            errors.append("model.anchors.base_sizes must not be empty when enabled")
        if config.model.anchors.scales_per_octave < 1:
            errors.append("model.anchors.scales_per_octave must be >= 1")
        if not config.model.anchors.aspect_ratios:
            errors.append("model.anchors.aspect_ratios must not be empty when enabled")

    if config.finetune.trainable_backbone_layers < 0:
        errors.append("finetune.trainable_backbone_layers must be >= 0")
    if config.finetune.backbone_lr_mult < 0:
        errors.append("finetune.backbone_lr_mult must be >= 0")

    if config.optimizer.lr <= 0:
        errors.append(f"optimizer.lr must be > 0 (got {config.optimizer.lr})")

    if config.train.epochs < 1:
        errors.append("train.epochs must be >= 1")
    if config.train.accumulate_steps < 1:
        errors.append("train.accumulate_steps must be >= 1")
    if config.train.checkpoint.mode not in {"max", "min"}:
        errors.append(
            f"train.checkpoint.mode must be 'max' or 'min' "
            f"(got {config.train.checkpoint.mode!r})"
        )

    errors.extend(_validate_area_ranges(config.eval.area_ranges))
    if config.eval.iou_thresholds is not None:
        if not config.eval.iou_thresholds:
            errors.append("eval.iou_thresholds must not be an empty list")
        if any(not 0.0 < t <= 1.0 for t in config.eval.iou_thresholds):
            errors.append("eval.iou_thresholds must all lie in (0, 1]")
    if not config.eval.max_dets or any(d < 1 for d in config.eval.max_dets):
        errors.append("eval.max_dets must be a non-empty list of positive ints")
    if config.eval.auto_area_ranges and len(config.eval.auto_area_percentiles) != 2:
        errors.append(
            "eval.auto_area_percentiles must hold exactly two percentiles "
            "(the small/medium and medium/large cuts)"
        )

    # The checkpoint monitor has to be a key the evaluator will actually emit.
    monitor = config.train.checkpoint.monitor
    monitor_labels = {f"AP_{label}" for label in config.eval.area_ranges}
    monitor_labels.update({"AP", "AP50", "AP75", "AR_1", "AR_10", "AR_100"})
    monitor_labels.update({f"AR_{label}" for label in config.eval.area_ranges})
    if monitor not in monitor_labels:
        errors.append(
            f"train.checkpoint.monitor {monitor!r} is not a metric this "
            f"evaluation produces. Available: {', '.join(sorted(monitor_labels))}"
        )
    if config.eval.primary_metric not in monitor_labels:
        errors.append(
            f"eval.primary_metric {config.eval.primary_metric!r} is not a metric "
            f"this evaluation produces. Available: {', '.join(sorted(monitor_labels))}"
        )

    if not 0.0 <= config.predict.postprocess.score_threshold <= 1.0:
        errors.append("predict.postprocess.score_threshold must lie in [0, 1]")
    nms = config.predict.postprocess.nms_iou_threshold
    if nms is not None and not 0.0 < nms <= 1.0:
        errors.append("predict.postprocess.nms_iou_threshold must lie in (0, 1]")

    tiling = config.predict.tiling
    if tiling.enabled:
        if len(tiling.tile_size) != 2 or any(s < 1 for s in tiling.tile_size):
            errors.append("predict.tiling.tile_size must be [width, height], both >= 1")
        if not 0.0 <= tiling.overlap < 1.0:
            errors.append("predict.tiling.overlap must lie in [0, 1)")

    if not 0.0 <= config.visualize.mask_alpha <= 1.0:
        errors.append("visualize.mask_alpha must lie in [0, 1]")
    if config.visualize.box_width < 1:
        errors.append("visualize.box_width must be >= 1")

    if not 1 <= config.app.server_port <= 65535:
        errors.append(f"app.server_port {config.app.server_port} is not a valid port")

    if errors:
        raise ConfigError(
            "invalid configuration:\n  - " + "\n  - ".join(errors)
        )
    return config


def _validate_area_ranges(area_ranges: Mapping[str, Sequence[float]]) -> List[str]:
    errors: List[str] = []
    if "all" not in area_ranges:
        errors.append(
            "eval.area_ranges must include an 'all' bucket — it is the range "
            "the headline AP is computed over"
        )
    for label, bounds in area_ranges.items():
        if len(bounds) != 2:
            errors.append(
                f"eval.area_ranges.{label} must be [min_area, max_area] "
                f"(got {len(bounds)} value(s))"
            )
            continue
        low, high = bounds
        if low < 0:
            errors.append(f"eval.area_ranges.{label}: min_area must be >= 0")
        if high <= low:
            errors.append(
                f"eval.area_ranges.{label}: max_area ({high}) must exceed "
                f"min_area ({low})"
            )
    return errors
