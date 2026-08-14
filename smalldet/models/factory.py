"""Build a detector from :class:`~smalldet.config.ModelConfig`.

Each torchvision detector family needs different surgery to retarget it at a
custom class set and a custom anchor pyramid, so each family registers one
``configure`` function and the factory dispatches on architecture name. Adding
support for a new family means registering one function, not editing a chain of
``if`` statements.

Two ordering constraints are load-bearing and easy to get wrong:

1. A pretrained checkpoint was trained with its original class count. The model
   must therefore be built at that count and have its head replaced afterwards
   — passing ``num_classes`` alongside ``weights="DEFAULT"`` is rejected by
   torchvision outright.
2. Changing the anchor pyramid can change how many anchors sit at each spatial
   location, which changes the shape of the head's output convolution. Anchors
   and class count are therefore applied together, in one rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torchvision.models import detection as tv_detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.models.detection.rpn import RPNHead

from ..config import AnchorConfig, ModelConfig
from ..registry import Registry
from .anchors import build_anchor_generator

#: family name -> function(model, num_classes, anchor_config) -> None
FAMILIES: Registry[Callable[..., None]] = Registry("detector family")


@dataclass(frozen=True)
class ArchitectureSpec:
    family: str
    #: Whether the family can be retargeted while keeping pretrained detector
    #: weights. When False, ``model.weights`` must be null and the backbone
    #: carries the transfer learning.
    supports_head_surgery: bool = True
    supports_custom_anchors: bool = True
    #: Why custom anchors are unsupported, when they are. Worth carrying
    #: explicitly: "FCOS is anchor-free" tells the reader to stop looking for a
    #: setting, where a generic "not supported" invites them to keep trying.
    anchor_note: str = ""


ARCHITECTURES: Dict[str, ArchitectureSpec] = {
    "fasterrcnn_resnet50_fpn": ArchitectureSpec("faster_rcnn"),
    "fasterrcnn_resnet50_fpn_v2": ArchitectureSpec("faster_rcnn"),
    "fasterrcnn_mobilenet_v3_large_fpn": ArchitectureSpec("faster_rcnn"),
    "fasterrcnn_mobilenet_v3_large_320_fpn": ArchitectureSpec("faster_rcnn"),
    "maskrcnn_resnet50_fpn": ArchitectureSpec("mask_rcnn"),
    "maskrcnn_resnet50_fpn_v2": ArchitectureSpec("mask_rcnn"),
    "retinanet_resnet50_fpn": ArchitectureSpec("retinanet"),
    "retinanet_resnet50_fpn_v2": ArchitectureSpec("retinanet"),
    "fcos_resnet50_fpn": ArchitectureSpec(
        "fcos",
        supports_custom_anchors=False,
        anchor_note=(
            "FCOS is anchor-free: it regresses distances to object edges from "
            "each feature location, so there is no anchor pyramid to tune. For "
            "small objects, raise model.min_size instead, or switch to "
            "fasterrcnn_resnet50_fpn_v2 and lower model.anchors.base_sizes."
        ),
    ),
    "ssd300_vgg16": ArchitectureSpec(
        "ssd",
        supports_head_surgery=False,
        supports_custom_anchors=False,
        anchor_note="SSD's anchor grid is fixed by its architecture.",
    ),
    "ssdlite320_mobilenet_v3_large": ArchitectureSpec(
        "ssd",
        supports_head_surgery=False,
        supports_custom_anchors=False,
        anchor_note="SSDLite's anchor grid is fixed by its architecture.",
    ),
}


# ------------------------------------------------------------- family adapters


@FAMILIES.register("faster_rcnn")
def _configure_faster_rcnn(
    model: nn.Module, num_classes: int, anchors: Optional[AnchorConfig]
) -> None:
    predictor = model.roi_heads.box_predictor
    if predictor.cls_score.out_features != num_classes:
        model.roi_heads.box_predictor = FastRCNNPredictor(
            predictor.cls_score.in_features, num_classes
        )
    if anchors is not None and anchors.enabled:
        _replace_rpn_anchors(model, anchors)


@FAMILIES.register("mask_rcnn")
def _configure_mask_rcnn(
    model: nn.Module, num_classes: int, anchors: Optional[AnchorConfig]
) -> None:
    _configure_faster_rcnn(model, num_classes, anchors)
    mask_predictor = model.roi_heads.mask_predictor
    if mask_predictor.mask_fcn_logits.out_channels != num_classes:
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            mask_predictor.conv5_mask.in_channels,
            mask_predictor.conv5_mask.out_channels,
            num_classes,
        )


@FAMILIES.register("retinanet")
def _configure_retinanet(
    model: nn.Module, num_classes: int, anchors: Optional[AnchorConfig]
) -> None:
    in_channels = model.backbone.out_channels
    norm_layer = _detect_norm_layer(model.head.classification_head.conv)

    if anchors is not None and anchors.enabled:
        model.anchor_generator = build_anchor_generator(
            anchors, num_levels=len(model.anchor_generator.sizes)
        )
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    # RetinaNet's classification head folds num_classes and num_anchors into a
    # single output convolution, so both are rebuilt together.
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer=norm_layer
    )
    if model.head.regression_head.bbox_reg.out_channels != num_anchors * 4:
        model.head.regression_head = RetinaNetRegressionHead(
            in_channels, num_anchors, norm_layer=norm_layer
        )


@FAMILIES.register("fcos")
def _configure_fcos(
    model: nn.Module, num_classes: int, anchors: Optional[AnchorConfig]
) -> None:
    if anchors is not None and anchors.enabled:
        raise ValueError(
            "FCOS is anchor-free, so model.anchors has no effect on it. Set "
            "model.anchors.enabled: false, or choose an anchor-based "
            "architecture such as fasterrcnn_resnet50_fpn_v2."
        )
    head = model.head.classification_head
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    model.head.classification_head = FCOSClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
        num_convs=_count_convs(head.conv),
        norm_layer=_detect_norm_layer(head.conv),
    )
    model.head.classification_head.num_classes = num_classes


@FAMILIES.register("ssd")
def _configure_ssd(
    model: nn.Module, num_classes: int, anchors: Optional[AnchorConfig]
) -> None:
    # SSD's head is a per-level module list whose channel counts are tied to the
    # backbone's output shapes; rebuilding it correctly means reconstructing the
    # model. The supported path is to build it at the right class count from the
    # start, which build_model does whenever weights is null.
    raise ValueError(
        "SSD cannot be retargeted after construction. Set model.weights: null "
        "and model.weights_backbone: DEFAULT so the detector is built directly "
        "at the required class count."
    )


# ----------------------------------------------------------------- the factory


def build_model(
    config: ModelConfig,
    num_classes: Optional[int] = None,
    *,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Construct, retarget, and optionally restore a detector.

    ``num_classes`` includes background. It comes from the dataset unless
    ``model.num_classes`` overrides it.
    """
    resolved_classes = config.num_classes or num_classes
    if resolved_classes is None:
        raise ValueError(
            "num_classes is unknown: set model.num_classes, or pass a dataset "
            "so it can be inferred from the annotation file"
        )
    if resolved_classes < 2:
        raise ValueError(
            f"num_classes counts background, so it must be >= 2 (got {resolved_classes})"
        )

    spec = _lookup(config.architecture)
    builder = getattr(tv_detection, config.architecture)

    kwargs: Dict[str, Any] = {
        "min_size": config.min_size,
        "max_size": config.max_size,
        **config.kwargs,
    }
    if config.trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = config.trainable_backbone_layers

    if config.anchors.enabled and not spec.supports_custom_anchors:
        raise ValueError(
            f"{config.architecture} does not support a custom anchor pyramid, so "
            f"model.anchors.enabled must be false. {spec.anchor_note}".strip()
        )

    if config.weights:
        if not spec.supports_head_surgery:
            raise ValueError(
                f"{config.architecture} cannot keep pretrained detector weights "
                "while changing the class count. Set model.weights: null and use "
                "model.weights_backbone instead."
            )
        # Built at the checkpoint's own class count; the head is replaced below.
        model = builder(weights=config.weights, **kwargs)
        FAMILIES.get(spec.family)(model, resolved_classes, config.anchors)
    else:
        # No detector weights to preserve, so the class count can be passed
        # straight to the constructor and only the anchors need applying.
        model = builder(
            weights=None,
            weights_backbone=config.weights_backbone,
            num_classes=resolved_classes,
            **kwargs,
        )
        if config.anchors.enabled:
            FAMILIES.get(spec.family)(model, resolved_classes, config.anchors)

    if config.checkpoint:
        load_checkpoint(model, config.checkpoint, map_location=map_location)

    return model


def load_checkpoint(
    model: nn.Module,
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Restore weights written by :class:`smalldet.engine.Trainer`."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint {path} does not fit this model. The usual cause is a "
            f"mismatch between the checkpoint's class count or anchor pyramid and "
            f"the current config.\n{exc}"
        ) from exc
    return model


def available_architectures() -> list[str]:
    return sorted(ARCHITECTURES)


def _lookup(architecture: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(
            f"unknown architecture {architecture!r}; available: "
            f"{', '.join(available_architectures())}"
        ) from None


# ---------------------------------------------------------------------- detail


def _replace_rpn_anchors(model: nn.Module, anchors: AnchorConfig) -> None:
    """Swap the RPN's anchor pyramid, rebuilding its head if the count changes."""
    model.rpn.anchor_generator = build_anchor_generator(
        anchors, num_levels=len(model.rpn.anchor_generator.sizes)
    )
    new_count = model.rpn.anchor_generator.num_anchors_per_location()[0]
    old_count = model.rpn.head.cls_logits.out_channels
    if new_count != old_count:
        # The RPN head's 1x1 output convolutions are sized by anchor count, so a
        # different count invalidates the pretrained head. conv_depth is read
        # from the existing head so v1 (1 conv) and v2 (2 convs) both survive.
        model.rpn.head = RPNHead(
            model.backbone.out_channels,
            new_count,
            conv_depth=_count_convs(model.rpn.head.conv),
        )


def _count_convs(module: nn.Module) -> int:
    return max(1, sum(1 for m in module.modules() if isinstance(m, nn.Conv2d)))


def _detect_norm_layer(module: nn.Module) -> Optional[Callable[..., nn.Module]]:
    """Mirror the normalization the pretrained head used.

    The v2 detectors use GroupNorm in their heads while v1 uses none; rebuilding
    a head without checking would silently change the architecture.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.GroupNorm):
            return partial(nn.GroupNorm, submodule.num_groups)
    return None
