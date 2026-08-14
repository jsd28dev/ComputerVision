"""Training callbacks (observer pattern).

The trainer emits events; callbacks decide what to do with them. Keeping
logging, checkpointing, and early stopping out of the loop means the loop reads
as the algorithm it is, and means a new concern (a TensorBoard writer, a Slack
ping) is a new class rather than another branch in ``train``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..config import CheckpointConfig, EarlyStoppingConfig
from ..evaluation.coco_eval import UNDEFINED
from ..registry import Registry

CALLBACKS: Registry["Callback"] = Registry("callback")


@dataclass
class TrainerState:
    """Mutable state shared between the trainer and its callbacks."""

    epoch: int = 0
    global_step: int = 0
    epochs: int = 0
    #: Losses for the batch just finished, and metrics for the epoch just scored.
    batch_losses: Dict[str, float] = field(default_factory=dict)
    epoch_metrics: Dict[str, float] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    should_stop: bool = False
    output_dir: Path = Path("outputs")
    started_at: float = field(default_factory=time.time)


class Callback:
    """No-op base; override only the events that matter."""

    def on_train_begin(self, trainer: Any, state: TrainerState) -> None: ...

    def on_epoch_begin(self, trainer: Any, state: TrainerState) -> None: ...

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None: ...

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None: ...

    def on_train_end(self, trainer: Any, state: TrainerState) -> None: ...


class CallbackList(Callback):
    """Fans one event out to many callbacks, in registration order."""

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = list(callbacks or [])

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def _dispatch(self, event: str, trainer: Any, state: TrainerState) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(trainer, state)

    def on_train_begin(self, trainer: Any, state: TrainerState) -> None:
        self._dispatch("on_train_begin", trainer, state)

    def on_epoch_begin(self, trainer: Any, state: TrainerState) -> None:
        self._dispatch("on_epoch_begin", trainer, state)

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None:
        self._dispatch("on_batch_end", trainer, state)

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        self._dispatch("on_epoch_end", trainer, state)

    def on_train_end(self, trainer: Any, state: TrainerState) -> None:
        self._dispatch("on_train_end", trainer, state)


@CALLBACKS.register("console")
class ConsoleLogger(Callback):
    """Per-interval loss lines and an end-of-epoch metric summary.

    Each loss term is printed separately rather than only their sum: if
    ``loss_box_reg`` falls while ``loss_objectness`` stays flat, the RPN is not
    finding the objects at all, which on a small-object dataset usually means
    the anchor pyramid is too coarse.
    """

    def __init__(self, interval: int = 20) -> None:
        self.interval = max(1, interval)
        self._epoch_started = 0.0

    def on_epoch_begin(self, trainer: Any, state: TrainerState) -> None:
        self._epoch_started = time.time()
        lrs = ", ".join(f"{lr:.2e}" for lr in state.learning_rates)
        print(f"\nepoch {state.epoch + 1}/{state.epochs}  lr=[{lrs}]")

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None:
        if state.global_step % self.interval:
            return
        terms = "  ".join(
            f"{name.replace('loss_', '')}={value:.4f}"
            for name, value in sorted(state.batch_losses.items())
            if name != "total"
        )
        total = state.batch_losses.get("total", float("nan"))
        print(f"  step {state.global_step:>6}  loss={total:.4f}  {terms}")

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        elapsed = time.time() - self._epoch_started
        if state.epoch_metrics:
            headline = "  ".join(
                f"{key}={_fmt(value)}"
                for key, value in state.epoch_metrics.items()
                if key in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")
            )
            print(f"  [{elapsed:.1f}s] {headline}")
        else:
            print(f"  [{elapsed:.1f}s] no evaluation this epoch")


@CALLBACKS.register("csv")
class CsvLogger(Callback):
    """Append one row per epoch, for plotting after the fact."""

    def __init__(self, filename: str = "metrics.csv") -> None:
        self.filename = filename
        self._path: Optional[Path] = None
        self._fieldnames: List[str] = []

    def on_train_begin(self, trainer: Any, state: TrainerState) -> None:
        state.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = state.output_dir / self.filename

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        if self._path is None or not state.epoch_metrics:
            return
        row = {"epoch": state.epoch + 1, **state.epoch_metrics}
        write_header = not self._path.exists()
        if not self._fieldnames:
            self._fieldnames = list(row)
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)


@CALLBACKS.register("tensorboard")
class TensorBoardLogger(Callback):
    """Scalar logging to TensorBoard, if it is installed."""

    def __init__(self, subdir: str = "tensorboard") -> None:
        self.subdir = subdir
        self._writer: Any = None

    def on_train_begin(self, trainer: Any, state: TrainerState) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "note: tensorboard is not installed, so the tensorboard "
                "callback is inactive. `pip install tensorboard` to enable it."
            )
            return
        self._writer = SummaryWriter(str(state.output_dir / self.subdir))

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None:
        if self._writer is None:
            return
        for name, value in state.batch_losses.items():
            self._writer.add_scalar(f"loss/{name}", value, state.global_step)

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        if self._writer is None:
            return
        for name, value in state.epoch_metrics.items():
            if value > UNDEFINED:
                self._writer.add_scalar(f"val/{name}", value, state.epoch)

    def on_train_end(self, trainer: Any, state: TrainerState) -> None:
        if self._writer is not None:
            self._writer.close()


class CheckpointSaver(Callback):
    """Saves the last epoch, and the best one by the monitored metric.

    ``-1`` is COCO's "no ground truth in this bucket" sentinel, not a score, so
    it can never win a comparison — otherwise a run whose ``AP_small`` bucket is
    empty would save epoch 1 forever and report it as the best model.
    """

    def __init__(self, config: CheckpointConfig, extra: Optional[Dict[str, Any]] = None):
        self.config = config
        self.extra = extra or {}
        self.best_path: Optional[Path] = None
        self.last_path: Optional[Path] = None
        # Owned here rather than read back from TrainerState, mirroring
        # EarlyStopping. Reading it from the state made the callback's memory
        # the caller's responsibility: anything that passed a fresh state would
        # silently reset "best" and overwrite a better checkpoint with a worse
        # one. The state is still updated, since reporting reads it.
        self._best: Optional[float] = None

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        directory = Path(self.config.dir)
        directory.mkdir(parents=True, exist_ok=True)

        if self.config.save_last:
            self.last_path = directory / "last.pt"
            self._save(trainer, state, self.last_path)

        if not self.config.save_best or self.config.monitor not in state.epoch_metrics:
            return
        value = state.epoch_metrics[self.config.monitor]
        if value <= UNDEFINED or value != value:  # sentinel or NaN
            return
        if self._improved(value, self._best):
            self._best = value
            state.best_metric = value
            state.best_epoch = state.epoch
            self.best_path = directory / "best.pt"
            self._save(trainer, state, self.best_path)
            print(
                f"  saved best checkpoint: {self.config.monitor}={value:.4f} "
                f"(epoch {state.epoch + 1})"
            )

    def _improved(self, value: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return value > best if self.config.mode == "max" else value < best

    def _save(self, trainer: Any, state: TrainerState, path: Path) -> None:
        payload = {
            "model": trainer.model.state_dict(),
            "epoch": state.epoch,
            "metrics": state.epoch_metrics,
            "monitor": self.config.monitor,
            **self.extra,
        }
        torch.save(payload, path)


class EarlyStopping(Callback):
    """Stop when the monitored metric has not improved for ``patience`` epochs."""

    def __init__(self, config: EarlyStoppingConfig, monitor: str, mode: str = "max"):
        self.config = config
        self.monitor = monitor
        self.mode = mode
        self._best: Optional[float] = None
        self._waited = 0

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        if not self.config.enabled or self.monitor not in state.epoch_metrics:
            return
        value = state.epoch_metrics[self.monitor]
        if value <= UNDEFINED:
            return
        if self._best is None or self._is_better(value):
            self._best = value
            self._waited = 0
            return
        self._waited += 1
        if self._waited >= self.config.patience:
            state.should_stop = True
            print(
                f"  early stopping: {self.monitor} has not improved by "
                f"{self.config.min_delta} in {self.config.patience} epochs"
            )

    def _is_better(self, value: float) -> bool:
        assert self._best is not None
        if self.mode == "max":
            return value > self._best + self.config.min_delta
        return value < self._best - self.config.min_delta


class HistoryRecorder(Callback):
    """Keeps every epoch's metrics in memory, and writes them out at the end."""

    def __init__(self, filename: str = "history.json") -> None:
        self.filename = filename
        self.history: List[Dict[str, Any]] = []

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        if state.epoch_metrics:
            self.history.append({"epoch": state.epoch + 1, **state.epoch_metrics})

    def on_train_end(self, trainer: Any, state: TrainerState) -> None:
        if not self.history:
            return
        state.output_dir.mkdir(parents=True, exist_ok=True)
        (state.output_dir / self.filename).write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )


def _fmt(value: float) -> str:
    return "n/a" if value <= UNDEFINED else f"{value:.4f}"
