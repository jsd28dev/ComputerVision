"""Optimizers, LR schedules, and warmup.

SGD with momentum is the default because it is what CNN-based detectors are
tuned around; AdamW is registered for transformer-style heads. The warmup is
not optional decoration — a freshly initialised prediction head produces large
gradients for the first few hundred steps, and without warmup those propagate
straight into a pretrained backbone and take the loss to NaN.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..config import OptimizerConfig, SchedulerConfig
from ..registry import Registry

OPTIMIZERS: Registry[Optimizer] = Registry("optimizer")
SCHEDULERS: Registry[LRScheduler] = Registry("scheduler")


# ------------------------------------------------------------------ optimizers


@OPTIMIZERS.register("sgd")
def _sgd(params: Any, lr: float, weight_decay: float, **kwargs: Any) -> Optimizer:
    kwargs.setdefault("momentum", 0.9)
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZERS.register("adamw")
def _adamw(params: Any, lr: float, weight_decay: float, **kwargs: Any) -> Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZERS.register("adam")
def _adam(params: Any, lr: float, weight_decay: float, **kwargs: Any) -> Optimizer:
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZERS.register("rmsprop")
def _rmsprop(params: Any, lr: float, weight_decay: float, **kwargs: Any) -> Optimizer:
    return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)


# ------------------------------------------------------------------ schedulers


@SCHEDULERS.register("multistep")
def _multistep(optimizer: Optimizer, **kwargs: Any) -> LRScheduler:
    kwargs.setdefault("milestones", [16, 22])
    kwargs.setdefault("gamma", 0.1)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)


@SCHEDULERS.register("step")
def _step(optimizer: Optimizer, **kwargs: Any) -> LRScheduler:
    kwargs.setdefault("step_size", 3)
    kwargs.setdefault("gamma", 0.1)
    return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)


@SCHEDULERS.register("cosine")
def _cosine(optimizer: Optimizer, **kwargs: Any) -> LRScheduler:
    kwargs.setdefault("T_max", 20)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)


@SCHEDULERS.register("plateau")
def _plateau(optimizer: Optimizer, **kwargs: Any) -> Any:
    """Stepped with the monitored metric rather than blindly per epoch."""
    kwargs.setdefault("mode", "max")
    kwargs.setdefault("patience", 2)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


@SCHEDULERS.register("none")
def _none(optimizer: Optimizer, **kwargs: Any) -> LRScheduler:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


# ---------------------------------------------------------------------- public


def build_optimizer(
    config: OptimizerConfig, param_groups: Sequence[Dict[str, Any]]
) -> Optimizer:
    """Build the optimizer over pre-bucketed parameter groups.

    Groups already carry their own ``lr``; the config ``lr`` is the base the
    strategy scaled them from, and is passed through as the optimizer default.
    """
    groups = [dict(group) for group in param_groups]
    for group in groups:
        # `name` is ours, for logging. torch rejects unknown keys.
        group.pop("name", None)
    return OPTIMIZERS.get(config.name)(
        groups, lr=config.lr, weight_decay=config.weight_decay, **config.kwargs
    )


def build_scheduler(
    config: SchedulerConfig, optimizer: Optimizer, *, epochs: Optional[int] = None
) -> Any:
    """Build the epoch-level LR schedule."""
    kwargs = dict(config.kwargs)
    if config.name.lower() == "cosine" and epochs is not None:
        kwargs.setdefault("T_max", epochs)
    return SCHEDULERS.get(config.name)(optimizer, **kwargs)


def build_warmup(
    config: SchedulerConfig, optimizer: Optimizer, steps_per_epoch: int
) -> Optional[LRScheduler]:
    """Per-iteration linear warmup, or None when disabled.

    Stepped inside the batch loop, not once per epoch — the whole point is to
    ramp the LR over the first few hundred *iterations*.
    """
    warmup = config.warmup
    if not warmup.enabled or steps_per_epoch < 1:
        return None
    total_iters = min(warmup.iters, max(1, steps_per_epoch * warmup.epochs - 1))
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup.start_factor,
        total_iters=total_iters,
    )


def is_plateau_scheduler(scheduler: Any) -> bool:
    """ReduceLROnPlateau takes a metric in ``step()``; everything else does not."""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def current_learning_rates(optimizer: Optimizer) -> List[float]:
    return [group["lr"] for group in optimizer.param_groups]
