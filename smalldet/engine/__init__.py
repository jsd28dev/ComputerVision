"""Training: strategies, optimizers, callbacks, and the loop that uses them."""

from __future__ import annotations

from .hooks import (
    CALLBACKS,
    Callback,
    CallbackList,
    CheckpointSaver,
    ConsoleLogger,
    CsvLogger,
    EarlyStopping,
    HistoryRecorder,
    TensorBoardLogger,
    TrainerState,
)
from .optim import (
    OPTIMIZERS,
    SCHEDULERS,
    build_optimizer,
    build_scheduler,
    build_warmup,
    current_learning_rates,
)
from .strategies import (
    STRATEGIES,
    FinetuneStrategy,
    FullFinetune,
    GradualUnfreeze,
    HeadOnlyFinetune,
    PartialFinetune,
    build_strategy,
    count_trainable,
    set_trainable_backbone_layers,
    trainable_parameter_names,
)
from .trainer import Trainer

__all__ = [
    "CALLBACKS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "STRATEGIES",
    "Callback",
    "CallbackList",
    "CheckpointSaver",
    "ConsoleLogger",
    "CsvLogger",
    "EarlyStopping",
    "FinetuneStrategy",
    "FullFinetune",
    "GradualUnfreeze",
    "HeadOnlyFinetune",
    "HistoryRecorder",
    "PartialFinetune",
    "TensorBoardLogger",
    "Trainer",
    "TrainerState",
    "build_optimizer",
    "build_scheduler",
    "build_strategy",
    "build_warmup",
    "count_trainable",
    "current_learning_rates",
    "set_trainable_backbone_layers",
    "trainable_parameter_names",
]
