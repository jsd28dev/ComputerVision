"""Finetuning strategies — which parameters train, and at what learning rate.

Implemented as a strategy pattern behind a registry, so ``finetune.strategy``
in YAML selects the behaviour and the trainer never branches on it. Each
strategy answers three questions:

* ``prepare``      — which parameters have ``requires_grad`` set, at the start
* ``on_epoch_start`` — whether that set changes as training progresses
* ``param_groups``  — how the trainable parameters are bucketed for the optimizer

The choice matters most on exactly the datasets this project targets. A few
hundred images of small parts will overfit a fully-unfrozen ResNet-50 long
before the head has converged, while freezing everything caps the ceiling once
there is enough data (arXiv:1601.05150). ``partial`` is the default because it
is the reliable middle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from torch import nn

from ..config import FinetuneConfig
from ..registry import Registry

STRATEGIES: Registry["FinetuneStrategy"] = Registry("finetune strategy")

#: ResNet stage names, ordered from the output end inward. This is the order
#: torchvision's own `_resnet_fpn_extractor` unfreezes them in.
_RESNET_STAGES: Tuple[str, ...] = ("layer4", "layer3", "layer2", "layer1", "conv1")


class FinetuneStrategy(ABC):
    """Base class for a finetuning policy."""

    name: str = "base"

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config

    @abstractmethod
    def prepare(self, model: nn.Module) -> None:
        """Set ``requires_grad`` across the model for epoch 0."""

    def on_epoch_start(self, model: nn.Module, epoch: int) -> bool:
        """Optionally change the trainable set. Returns True if it changed.

        A True return tells the trainer to rebuild the optimizer, because a
        parameter that was frozen when the optimizer was constructed will never
        be updated no matter what ``requires_grad`` says afterwards.
        """
        return False

    def param_groups(
        self, model: nn.Module, base_lr: float, weight_decay: float
    ) -> List[Dict[str, Any]]:
        """Bucket trainable parameters into optimizer groups.

        The backbone carries pretrained features that are already close to
        useful; the head is random. Stepping them at the same rate lets the
        head's early gradients drag the backbone away from a good initialisation.
        """
        backbone, head = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            (backbone if name.startswith("backbone.") else head).append(parameter)

        groups: List[Dict[str, Any]] = []
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": base_lr * self.config.backbone_lr_mult,
                    "weight_decay": weight_decay,
                    "name": "backbone",
                }
            )
        if head:
            groups.append(
                {
                    "params": head,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "head",
                }
            )
        if not groups:
            raise ValueError(
                "no trainable parameters: every parameter is frozen. Check "
                "finetune.strategy and finetune.freeze_patterns."
            )
        return groups

    def apply_patterns(self, model: nn.Module) -> None:
        """Apply the regex escape hatches, freeze first then unfreeze.

        Unfreeze wins, so ``freeze_patterns: ['backbone.*']`` plus
        ``unfreeze_patterns: ['backbone.fpn.*']`` reads the way it looks.
        """
        for pattern in self.config.freeze_patterns:
            regex = re.compile(pattern)
            for name, parameter in model.named_parameters():
                if regex.search(name):
                    parameter.requires_grad_(False)
        for pattern in self.config.unfreeze_patterns:
            regex = re.compile(pattern)
            for name, parameter in model.named_parameters():
                if regex.search(name):
                    parameter.requires_grad_(True)

    def summary(self, model: nn.Module) -> str:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        share = trainable / total if total else 0.0
        return (
            f"finetune strategy '{self.name}': {trainable:,}/{total:,} parameters "
            f"trainable ({share:.1%})"
        )


@STRATEGIES.register("full")
class FullFinetune(FinetuneStrategy):
    """Everything trains, with a lower LR on the backbone."""

    name = "full"

    def prepare(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        self.apply_patterns(model)


@STRATEGIES.register("head_only")
class HeadOnlyFinetune(FinetuneStrategy):
    """Transfer learning: the backbone is a fixed feature extractor.

    Cheapest and least prone to overfitting, which makes it the right first
    baseline on a few-hundred-image dataset — and the right thing to move away
    from as soon as validation AP_small stops improving.
    """

    name = "head_only"

    def prepare(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if hasattr(model, "backbone"):
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)
        self.apply_patterns(model)


@STRATEGIES.register("partial")
class PartialFinetune(FinetuneStrategy):
    """Freeze early backbone stages, train the later ones plus the head.

    Early convolutions learn edges and textures that transfer across domains;
    later stages encode semantics that usually do not. Retraining only the
    latter adapts the model where it matters and leaves the cheap-to-learn
    filters alone.
    """

    name = "partial"

    def prepare(self, model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        set_trainable_backbone_layers(model, self.config.trainable_backbone_layers)
        self.apply_patterns(model)


@STRATEGIES.register("gradual")
class GradualUnfreeze(FinetuneStrategy):
    """Start mostly frozen, unfreeze deeper stages on a schedule.

    A practical middle path when it is not obvious upfront how much data is
    "enough": let the head adapt against stable features first, then hand it
    more of the backbone once the loss has settled.
    """

    name = "gradual"

    def __init__(self, config: FinetuneConfig) -> None:
        super().__init__(config)
        if not config.gradual_schedule:
            raise ValueError(
                "finetune.strategy 'gradual' needs finetune.gradual_schedule, "
                "e.g. {0: 0, 3: 2, 6: 5} — epoch to trainable backbone layers"
            )
        self._schedule = {
            int(epoch): int(layers)
            for epoch, layers in config.gradual_schedule.items()
        }
        self._current: Optional[int] = None

    def prepare(self, model: nn.Module) -> None:
        first = min(self._schedule)
        self._apply(model, self._schedule[first])

    def on_epoch_start(self, model: nn.Module, epoch: int) -> bool:
        # The most recent schedule entry at or before this epoch.
        applicable = [e for e in self._schedule if e <= epoch]
        if not applicable:
            return False
        layers = self._schedule[max(applicable)]
        if layers == self._current:
            return False
        self._apply(model, layers)
        return True

    def _apply(self, model: nn.Module, layers: int) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        set_trainable_backbone_layers(model, layers)
        self.apply_patterns(model)
        self._current = layers


# ---------------------------------------------------------------------- public


def build_strategy(config: FinetuneConfig) -> FinetuneStrategy:
    return STRATEGIES.get(config.strategy)(config)


def set_trainable_backbone_layers(model: nn.Module, trainable_layers: int) -> None:
    """Freeze all but the last ``trainable_layers`` stages of the backbone body.

    The FPN is deliberately left trainable: it is randomly initialised relative
    to the classification backbone and is where multi-scale features — the ones
    small objects depend on — are actually formed. torchvision's own builders
    make the same choice.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError(
            "this model has no `backbone` attribute, so backbone stages cannot "
            "be frozen. Use finetune.freeze_patterns instead."
        )
    body = getattr(backbone, "body", backbone)
    stage_names = _stage_names(body)

    if trainable_layers < 0 or trainable_layers > len(stage_names):
        raise ValueError(
            f"finetune.trainable_backbone_layers must lie in [0, {len(stage_names)}] "
            f"for this backbone (got {trainable_layers})"
        )

    trainable = set(stage_names[:trainable_layers])
    if trainable_layers == len(stage_names):
        # The stem's BatchNorm sits outside the named stages on ResNet.
        trainable.add("bn1")

    for name, parameter in body.named_parameters():
        parameter.requires_grad_(any(name.startswith(stage) for stage in trainable))


def _stage_names(body: nn.Module) -> List[str]:
    """Backbone stages ordered from the output end inward."""
    children = [name for name, _ in body.named_children()]
    resnet_stages = [stage for stage in _RESNET_STAGES if stage in children]
    if len(resnet_stages) == len(_RESNET_STAGES):
        return list(_RESNET_STAGES)
    # Generic fallback (mobilenet's `features`, custom trunks): treat named
    # children as stages, deepest first.
    return list(reversed(children))


def trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
