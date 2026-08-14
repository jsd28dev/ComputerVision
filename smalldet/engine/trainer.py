"""The training loop.

Everything configurable lives in the config; everything reusable lives in a
callback or a strategy. What remains here is the algorithm: forward, sum the
loss terms, backward, step, evaluate, notify.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.loaders import move_to_device
from ..evaluation.coco_eval import EvalResult, GroundTruth
from ..evaluation.runner import evaluate_model
from ..runtime import describe_device, resolve_device
from .hooks import (
    CALLBACKS,
    Callback,
    CallbackList,
    CheckpointSaver,
    EarlyStopping,
    HistoryRecorder,
    TrainerState,
)
from .optim import (
    build_optimizer,
    build_scheduler,
    build_warmup,
    current_learning_rates,
    is_plateau_scheduler,
)
from .strategies import FinetuneStrategy, build_strategy


class Trainer:
    """Finetunes a detector according to a :class:`Config`."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        ground_truth: Optional[GroundTruth] = None,
        eval_config: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        device: Optional[torch.device | str] = None,
        strategy: Optional[FinetuneStrategy] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ground_truth = ground_truth
        self.eval_config = eval_config or config.eval

        self.device = resolve_device(
            str(device) if device is not None else config.train.device
        )
        self.model.to(self.device)

        self.strategy = strategy or build_strategy(config.finetune)
        self.strategy.prepare(self.model)

        self.state = TrainerState(
            epochs=config.train.epochs,
            output_dir=Path(config.train.output_dir),
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(
            config.scheduler, self.optimizer, epochs=config.train.epochs
        )
        self.warmup = build_warmup(
            config.scheduler, self.optimizer, len(train_loader)
        )

        # GradScaler only does anything on CUDA; enabling it elsewhere is a
        # no-op that still costs a warning per step.
        self.amp_enabled = bool(config.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.callbacks = CallbackList(
            callbacks if callbacks is not None else self._default_callbacks()
        )
        self.history = HistoryRecorder()
        self.callbacks.append(self.history)

    # -------------------------------------------------------------- lifecycle

    def fit(self) -> TrainerState:
        """Run the full schedule and return the final state."""
        print(f"device: {describe_device(self.device)}")
        print(self.strategy.summary(self.model))

        self.state.learning_rates = current_learning_rates(self.optimizer)
        self.callbacks.on_train_begin(self, self.state)

        for epoch in range(self.config.train.epochs):
            self.state.epoch = epoch

            # A strategy may hand over more of the backbone partway through. The
            # optimizer has to be rebuilt when it does: parameters absent from
            # its param_groups are never updated, whatever requires_grad says.
            if self.strategy.on_epoch_start(self.model, epoch):
                self.optimizer = self._build_optimizer()
                self.scheduler = build_scheduler(
                    self.config.scheduler,
                    self.optimizer,
                    epochs=self.config.train.epochs,
                )
                print(f"  {self.strategy.summary(self.model)}")

            self.state.learning_rates = current_learning_rates(self.optimizer)
            self.callbacks.on_epoch_begin(self, self.state)

            self.train_one_epoch()

            self.state.epoch_metrics = {}
            if self._should_evaluate(epoch):
                result = self.evaluate()
                self.state.epoch_metrics = dict(result.metrics)

            self.callbacks.on_epoch_end(self, self.state)
            self._step_scheduler()

            if self.state.should_stop:
                break

        self.callbacks.on_train_end(self, self.state)
        return self.state

    def train_one_epoch(self) -> Dict[str, float]:
        """One pass over the training loader."""
        self.model.train()
        accumulate = max(1, self.config.train.accumulate_steps)
        totals: Dict[str, float] = {}
        batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        for index, (images, targets) in enumerate(self.train_loader):
            if (
                self.config.train.max_train_batches is not None
                and index >= self.config.train.max_train_batches
            ):
                break
            # Checked per batch, not only per epoch, so a Stop button in the UI
            # takes effect within seconds rather than at the next epoch boundary.
            if self.state.should_stop:
                break

            images, targets = move_to_device(images, targets, self.device)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                # In train mode a torchvision detector returns its loss dict
                # rather than predictions.
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"loss became {loss_value} at step {self.state.global_step}. "
                    "The usual cause is too high an effective learning rate at "
                    "the start of training: lengthen scheduler.warmup.iters or "
                    "lower optimizer.lr before changing anything else.\n"
                    f"  terms: { {k: float(v) for k, v in loss_dict.items()} }"
                )

            self.scaler.scale(loss / accumulate).backward()

            if (index + 1) % accumulate == 0:
                if self.config.train.grad_clip is not None:
                    # Gradients must be unscaled before their norm is meaningful.
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.train.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Warmup is per-iteration by design; stepping it per epoch would
                # leave the LR at its start_factor for a whole epoch.
                if self.warmup is not None and self._in_warmup():
                    self.warmup.step()

            self.state.global_step += 1
            self.state.batch_losses = {
                name: float(value.detach()) for name, value in loss_dict.items()
            }
            self.state.batch_losses["total"] = loss_value
            for name, value in self.state.batch_losses.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1

            self.callbacks.on_batch_end(self, self.state)

        return {name: value / max(1, batches) for name, value in totals.items()}

    @torch.inference_mode()
    def evaluate(self) -> EvalResult:
        """Score the validation split with COCO metrics."""
        if self.val_loader is None or self.ground_truth is None:
            raise ValueError(
                "evaluation needs both a val_loader and a ground_truth; "
                "construct the Trainer with them or set train.eval_interval: 0"
            )
        return evaluate_model(
            self.model,
            self.val_loader,
            self.ground_truth,
            self.eval_config,
            device=self.device,
            max_batches=self.config.train.max_eval_batches,
        )

    # ------------------------------------------------------------------ detail

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = self.strategy.param_groups(
            self.model, self.config.optimizer.lr, self.config.optimizer.weight_decay
        )
        return build_optimizer(self.config.optimizer, groups)

    def _default_callbacks(self) -> List[Callback]:
        callbacks: List[Callback] = []
        for name in self.config.train.callbacks:
            factory = CALLBACKS.get(name)
            callbacks.append(
                factory(self.config.train.log_interval)
                if name == "console"
                else factory()
            )
        callbacks.append(
            CheckpointSaver(
                self.config.train.checkpoint,
                extra={"config_name": self.config.name},
            )
        )
        callbacks.append(
            EarlyStopping(
                self.config.train.early_stopping,
                monitor=self.config.train.checkpoint.monitor,
                mode=self.config.train.checkpoint.mode,
            )
        )
        return callbacks

    def _should_evaluate(self, epoch: int) -> bool:
        interval = self.config.train.eval_interval
        if interval < 1 or self.val_loader is None or self.ground_truth is None:
            return False
        is_last = epoch == self.config.train.epochs - 1
        return is_last or (epoch + 1) % interval == 0

    def _in_warmup(self) -> bool:
        return self.state.epoch < self.config.scheduler.warmup.epochs

    def _step_scheduler(self) -> None:
        if is_plateau_scheduler(self.scheduler):
            monitor = self.config.train.checkpoint.monitor
            if monitor in self.state.epoch_metrics:
                self.scheduler.step(self.state.epoch_metrics[monitor])
            return
        self.scheduler.step()
