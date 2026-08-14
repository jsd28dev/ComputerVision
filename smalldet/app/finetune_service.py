"""The finetuning service behind the UI's second page.

Like :mod:`smalldet.app.service`, this imports no Gradio: it turns a flat bag of
UI values into a validated :class:`Config`, describes what that config will
actually do, and runs training while streaming progress. That separation is
what makes the whole finetuning page testable without a browser.

The design decision that matters here: the UI does not have its own notion of
what a hyper-parameter is. It edits a config, and the config is the single
source of truth — the same one the CLI would load from YAML. Anything trained
from the UI can therefore be reproduced exactly by exporting the config it
built, which the page offers as a download.
"""

from __future__ import annotations

import io
import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..config import Config, ConfigError, config_from_dict, deep_merge, dump_config
from ..config.binding import to_dict
from ..engine.hooks import Callback, TrainerState
from ..evaluation.coco_eval import UNDEFINED
from ..models import available_architectures
from ..registry import RegistryError

#: Offered in the UI. Ordered from least to most of the model retrained, which
#: is also the order to try them in on a small dataset.
STRATEGY_CHOICES: List[Tuple[str, str]] = [
    (
        "head_only",
        "Transfer learning — freeze the whole backbone, train only the new "
        "prediction head. Cheapest and least prone to overfitting; the right "
        "baseline for tens to low hundreds of images.",
    ),
    (
        "partial",
        "Partial finetuning — freeze early backbone stages, train the later "
        "ones plus the head. Usually the sweet spot: adapts high-level "
        "features while preserving cheap-to-learn edge and texture filters.",
    ),
    (
        "gradual",
        "Gradual unfreezing — start mostly frozen and hand over deeper stages "
        "on a schedule. A middle path when it is unclear how much data is "
        "'enough'.",
    ),
    (
        "full",
        "Full finetuning — every parameter trains, with a lower learning rate "
        "on the backbone. Highest ceiling, needs the most data.",
    ),
]

OPTIMIZER_CHOICES = ["sgd", "adamw", "adam", "rmsprop"]
SCHEDULER_CHOICES = ["multistep", "step", "cosine", "plateau", "none"]
MONITOR_CHOICES = ["AP_small", "AP_medium", "AP", "AP50", "AP75", "AP_large"]


@dataclass
class TrainingProgress:
    """One streamed update from a running job."""

    status: str = "idle"
    log: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    checkpoint: Optional[str] = None
    finished: bool = False
    failed: bool = False


class _StreamCallback(Callback):
    """Pushes trainer events onto a queue for the UI generator to drain."""

    def __init__(self, events: "queue.Queue[Tuple[str, Any]]") -> None:
        self.events = events

    def on_train_begin(self, trainer: Any, state: TrainerState) -> None:
        trainable = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in trainer.model.parameters())
        self.events.put(
            (
                "log",
                f"device: {trainer.device}\n"
                f"trainable parameters: {trainable:,} / {total:,} "
                f"({trainable / max(total, 1):.1%})",
            )
        )

    def on_epoch_begin(self, trainer: Any, state: TrainerState) -> None:
        lrs = ", ".join(f"{lr:.2e}" for lr in state.learning_rates)
        self.events.put(
            ("log", f"\nepoch {state.epoch + 1}/{state.epochs}  lr=[{lrs}]")
        )

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None:
        interval = max(1, trainer.config.train.log_interval)
        if state.global_step % interval:
            return
        terms = "  ".join(
            f"{name.replace('loss_', '')}={value:.4f}"
            for name, value in sorted(state.batch_losses.items())
            if name != "total"
        )
        self.events.put(
            (
                "log",
                f"  step {state.global_step:>5}  "
                f"loss={state.batch_losses.get('total', float('nan')):.4f}  {terms}",
            )
        )

    def on_epoch_end(self, trainer: Any, state: TrainerState) -> None:
        if not state.epoch_metrics:
            self.events.put(("log", "  (no evaluation this epoch)"))
            return
        headline = "  ".join(
            f"{key}={_fmt(state.epoch_metrics[key])}"
            for key in ("AP", "AP50", "AP_small", "AP_medium", "AP_large")
            if key in state.epoch_metrics
        )
        self.events.put(("log", f"  {headline}"))
        self.events.put(
            ("epoch", {"epoch": state.epoch + 1, **state.epoch_metrics})
        )


class _StopCallback(Callback):
    """Lets the UI's Stop button end the run at the next batch."""

    def __init__(self, flag: threading.Event) -> None:
        self.flag = flag

    def on_batch_end(self, trainer: Any, state: TrainerState) -> None:
        if self.flag.is_set():
            state.should_stop = True

    def on_epoch_begin(self, trainer: Any, state: TrainerState) -> None:
        if self.flag.is_set():
            state.should_stop = True


class FinetuneService:
    """Builds configs from UI values and runs training against them."""

    def __init__(self, base_config: Optional[Config] = None) -> None:
        self.base_config = base_config or Config()
        self._stop_flag = threading.Event()
        self._running = False

    # -------------------------------------------------------------- choices

    @staticmethod
    def architectures() -> List[str]:
        return available_architectures()

    @staticmethod
    def strategies() -> List[str]:
        return [name for name, _ in STRATEGY_CHOICES]

    @staticmethod
    def strategy_help() -> str:
        return "\n\n".join(
            f"**`{name}`** — {description}" for name, description in STRATEGY_CHOICES
        )

    # --------------------------------------------------------------- config

    def build_config(
        self,
        *,
        data_root: str,
        train_images: str,
        train_annotations: str,
        val_images: str,
        val_annotations: str,
        test_images: str = "",
        test_annotations: str = "",
        architecture: str = "fasterrcnn_resnet50_fpn_v2",
        pretrained: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
        anchors_enabled: bool = True,
        anchor_base_sizes: str = "8, 16, 32, 64, 128",
        detections_per_image: int = 300,
        strategy: str = "partial",
        trainable_backbone_layers: int = 3,
        backbone_lr_mult: float = 0.1,
        gradual_schedule: str = "0:0, 3:2, 6:5",
        optimizer: str = "sgd",
        learning_rate: float = 0.005,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        scheduler: str = "multistep",
        milestones: str = "16, 22",
        gamma: float = 0.1,
        warmup_enabled: bool = True,
        warmup_iters: int = 500,
        epochs: int = 20,
        batch_size: int = 2,
        num_workers: int = 0,
        seed: int = 0,
        amp: bool = False,
        grad_clip: Optional[float] = None,
        accumulate_steps: int = 1,
        horizontal_flip: bool = True,
        photometric: bool = True,
        scale_jitter: bool = True,
        monitor: str = "AP_small",
        auto_area_ranges: bool = False,
        early_stopping: bool = False,
        patience: int = 5,
        output_dir: str = "outputs/ui-run",
        device: str = "auto",
    ) -> Config:
        """Assemble a validated Config from the page's widget values."""
        overrides: Dict[str, Any] = {
            "name": "ui-finetune",
            "data": {
                "root": data_root or ".",
                "train": {"images": train_images, "annotations": train_annotations},
                "val": {"images": val_images, "annotations": val_annotations},
                "test": {"images": test_images, "annotations": test_annotations},
                "train_loader": {
                    "batch_size": int(batch_size),
                    "shuffle": True,
                    "num_workers": int(num_workers),
                },
                "eval_loader": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": int(num_workers),
                },
                "augmentation": {
                    "train": _augmentation(horizontal_flip, photometric, scale_jitter),
                    "eval": [
                        {"name": "to_dtype", "params": {"dtype": "float32", "scale": True}},
                        {"name": "to_pure_tensor"},
                    ],
                },
            },
            "model": {
                "architecture": architecture,
                "weights": "DEFAULT" if pretrained else None,
                "weights_backbone": None if pretrained else "DEFAULT",
                "min_size": int(min_size),
                "max_size": int(max_size),
                "anchors": {
                    "enabled": bool(anchors_enabled),
                    "base_sizes": _int_list(anchor_base_sizes, "anchor base sizes"),
                    "scales_per_octave": 3 if architecture.startswith("retinanet") else 1,
                },
                "kwargs": _detector_kwargs(architecture, int(detections_per_image)),
            },
            "finetune": {
                "strategy": strategy,
                "trainable_backbone_layers": int(trainable_backbone_layers),
                "backbone_lr_mult": float(backbone_lr_mult),
                "gradual_schedule": _schedule(gradual_schedule)
                if strategy == "gradual"
                else {},
            },
            "optimizer": {
                "name": optimizer,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "kwargs": {"momentum": float(momentum)}
                if optimizer in {"sgd", "rmsprop"}
                else {},
            },
            "scheduler": {
                "name": scheduler,
                "kwargs": _scheduler_kwargs(scheduler, milestones, gamma, epochs),
                "warmup": {
                    "enabled": bool(warmup_enabled),
                    "iters": int(warmup_iters),
                    "start_factor": 0.001,
                    "epochs": 1,
                },
            },
            "train": {
                "epochs": int(epochs),
                "device": device,
                "seed": int(seed),
                "amp": bool(amp),
                "grad_clip": float(grad_clip) if grad_clip else None,
                "accumulate_steps": int(accumulate_steps),
                "output_dir": output_dir,
                "callbacks": ["csv"],
                "checkpoint": {
                    "dir": str(Path(output_dir) / "checkpoints"),
                    "monitor": monitor,
                    "mode": "max",
                },
                "early_stopping": {
                    "enabled": bool(early_stopping),
                    "patience": int(patience),
                },
            },
            "eval": {
                "auto_area_ranges": bool(auto_area_ranges),
                "primary_metric": monitor,
            },
        }
        merged = deep_merge(to_dict(self.base_config), overrides)

        # deep_merge recurses into every mapping, which is right for the schema
        # but wrong for the free-form `kwargs` bags: they are argument lists for
        # one specific optimizer/scheduler/architecture, not a set of
        # independent settings. Merging them means switching from `step` to
        # `multistep` keeps a stale `step_size` and MultiStepLR raises on the
        # unexpected argument. These are replaced wholesale instead.
        merged["model"]["kwargs"] = overrides["model"]["kwargs"]
        merged["optimizer"]["kwargs"] = overrides["optimizer"]["kwargs"]
        merged["scheduler"]["kwargs"] = overrides["scheduler"]["kwargs"]
        merged["finetune"]["gradual_schedule"] = overrides["finetune"][
            "gradual_schedule"
        ]
        return config_from_dict(merged)

    def describe(self, **kwargs: Any) -> str:
        """Validate the settings and explain what they will do.

        Runs before training so a misconfiguration costs a second rather than
        an epoch, and so the consequences of the finetuning strategy are stated
        in words rather than left implicit in a radio button.
        """
        try:
            config = self.build_config(**kwargs)
        except (ConfigError, ValueError, RegistryError) as exc:
            return f"### ⚠️ Configuration error\n\n```\n{exc}\n```"

        lines = [
            "### Ready to train",
            "",
            f"- **Architecture** `{config.model.architecture}`, "
            + ("pretrained COCO weights" if config.model.weights else "backbone weights only"),
            f"- **Strategy** `{config.finetune.strategy}` — "
            + _strategy_summary(config),
            f"- **Optimizer** `{config.optimizer.name}` at lr {config.optimizer.lr:g}"
            + (
                f", backbone at {config.optimizer.lr * config.finetune.backbone_lr_mult:g}"
                f" ({config.finetune.backbone_lr_mult:g}×)"
                if config.finetune.strategy != "head_only"
                else ""
            ),
            f"- **Schedule** `{config.scheduler.name}`"
            + (
                f", {config.scheduler.warmup.iters}-iteration warmup"
                if config.scheduler.warmup.enabled
                else ", no warmup"
            ),
            f"- **Budget** {config.train.epochs} epochs, batch size "
            f"{config.data.train_loader.batch_size}"
            + (
                f", grad accumulation ×{config.train.accumulate_steps}"
                f" (effective batch {config.data.train_loader.batch_size * config.train.accumulate_steps})"
                if config.train.accumulate_steps > 1
                else ""
            ),
            f"- **Input resolution** min {config.model.min_size} / max {config.model.max_size} px",
        ]
        if config.model.anchors.enabled:
            lines.append(
                f"- **Anchors** {config.model.anchors.base_sizes} — lowered from "
                "torchvision's 32–512 default, which is what makes objects "
                "under ~32px reachable by the RPN at all"
            )
        lines.append(
            f"- **Selecting on** `{config.train.checkpoint.monitor}`"
            + (
                " — area buckets will be derived from this dataset's own size "
                "distribution"
                if config.eval.auto_area_ranges
                else ""
            )
        )

        warnings = _warnings(config)
        if warnings:
            lines += ["", "### Worth checking", ""]
            lines += [f"- {warning}" for warning in warnings]

        splits = [
            name
            for name in ("train", "val", "test")
            if getattr(config.data, name).is_configured
        ]
        lines += ["", f"**Splits configured:** {', '.join(splits) or 'none'}"]
        if "val" not in splits:
            lines.append(
                "\n⚠️ Without a validation split nothing is scored, so no best "
                "checkpoint can be selected."
            )
        return "\n".join(lines)

    def export_config(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the config this page built, so the run is reproducible by CLI."""
        config = self.build_config(**kwargs)
        target = Path(path)
        dump_config(config, target)
        return target

    # -------------------------------------------------------------- running

    @property
    def running(self) -> bool:
        return self._running

    def request_stop(self) -> str:
        self._stop_flag.set()
        return "Stop requested — training will end after the current batch."

    def run(self, **kwargs: Any) -> Iterator[TrainingProgress]:
        """Train, yielding a :class:`TrainingProgress` as events arrive.

        Training runs on a worker thread so the generator can keep the UI
        responsive; without that, the page would freeze for the whole run.
        """
        try:
            config = self.build_config(**kwargs)
        except (ConfigError, ValueError, RegistryError) as exc:
            yield TrainingProgress(
                status="failed", log=f"Configuration error:\n{exc}", failed=True, finished=True
            )
            return

        self._stop_flag.clear()
        self._running = True
        events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        result: Dict[str, Any] = {}

        def worker() -> None:
            from ..pipeline import run_training

            buffer = io.StringIO()
            try:
                # resolve_eval_config prints its size report; capture it so the
                # area-bucket warning reaches the UI instead of the terminal.
                import contextlib

                with contextlib.redirect_stdout(buffer):
                    state = run_training(
                        config,
                        verbose=True,
                        extra_callbacks=[
                            _StreamCallback(events),
                            _StopCallback(self._stop_flag),
                        ],
                    )
                result["state"] = state
                result["stopped"] = self._stop_flag.is_set()
                events.put(("done", buffer.getvalue()))
            except Exception:
                events.put(("error", buffer.getvalue() + "\n" + traceback.format_exc()))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        log: List[str] = [f"Training `{config.name}` → `{config.train.output_dir}`"]
        history: List[Dict[str, Any]] = []
        yield TrainingProgress(status="starting", log="\n".join(log), history=history)

        while True:
            kind, payload = events.get()
            if kind == "log":
                log.append(str(payload))
                yield TrainingProgress(
                    status="training", log="\n".join(log[-400:]), history=list(history)
                )
            elif kind == "epoch":
                history.append(payload)
                yield TrainingProgress(
                    status="training", log="\n".join(log[-400:]), history=list(history)
                )
            elif kind == "done":
                stopped_early = bool(result.get("stopped"))
                if payload.strip():
                    log.insert(1, payload.strip())
                state = result.get("state")
                checkpoint = Path(config.train.checkpoint.dir) / "best.pt"
                log.append("\nTraining finished.")
                self._running = False
                # Read the flag as it stood when the run ended, not later: a
                # request that arrives after the last batch never took effect,
                # and reporting "stopped" for a run that completed in full tells
                # the user their training was cut short when it was not.
                yield TrainingProgress(
                    status="stopped" if stopped_early else "finished",
                    log="\n".join(log[-400:]),
                    history=list(history),
                    best_metric=getattr(state, "best_metric", None),
                    best_epoch=getattr(state, "best_epoch", None),
                    checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                    finished=True,
                )
                return
            elif kind == "error":
                self._running = False
                yield TrainingProgress(
                    status="failed",
                    log="\n".join(log[-200:]) + "\n\n" + str(payload),
                    history=list(history),
                    finished=True,
                    failed=True,
                )
                return


# ---------------------------------------------------------------- formatting


def history_markdown(history: Sequence[Dict[str, Any]]) -> str:
    """Per-epoch metrics as a Markdown table."""
    if not history:
        return "_No epoch has been scored yet._"
    columns = ["epoch", "AP", "AP50", "AP_small", "AP_medium", "AP_large"]
    present = [c for c in columns if any(c in entry for entry in history)]
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for entry in history:
        cells = []
        for column in present:
            value = entry.get(column)
            cells.append(
                str(value)
                if column == "epoch"
                else (_fmt(value) if isinstance(value, (int, float)) else "–")
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_summary_markdown(written: Dict[str, Path]) -> str:
    from ..data.split import summarize_split

    if not written:
        return "_Nothing was written._"
    return (
        "**Splits written:**\n\n```\n"
        + summarize_split(written)
        + "\n```\n\nPoint the annotation fields above at these files."
    )


# ---------------------------------------------------------------------- detail


def _augmentation(flip: bool, photometric: bool, jitter: bool) -> List[Dict[str, Any]]:
    """Build the train pipeline from three checkboxes.

    RandomZoomOut is deliberately not offered: it pads the frame and shrinks
    every object in it, which is the opposite of what a small-object dataset
    needs. ScaleJitter's range starts at 1.0 for the same reason.
    """
    pipeline: List[Dict[str, Any]] = []
    if flip:
        pipeline.append({"name": "random_horizontal_flip", "params": {"p": 0.5}})
    if photometric:
        pipeline.append({"name": "random_photometric_distort"})
    if jitter:
        pipeline.append(
            {
                "name": "scale_jitter",
                "params": {"target_size": [800, 800], "scale_range": [1.0, 1.8]},
            }
        )
    pipeline.append(
        {"name": "sanitize_bounding_boxes", "params": {"min_size": 1.0, "min_area": 1.0}}
    )
    pipeline.append({"name": "to_dtype", "params": {"dtype": "float32", "scale": True}})
    pipeline.append({"name": "to_pure_tensor"})
    return pipeline


def _detector_kwargs(architecture: str, detections_per_image: int) -> Dict[str, Any]:
    """Only the R-CNN family accepts the box_* knobs."""
    if architecture.startswith(("fasterrcnn", "maskrcnn")):
        return {"box_detections_per_img": detections_per_image}
    return {"detections_per_img": detections_per_image} if architecture.startswith(
        ("retinanet", "fcos")
    ) else {}


def _int_list(text: str, label: str) -> List[int]:
    try:
        return [int(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError:
        raise ValueError(
            f"{label} must be a comma-separated list of integers, got {text!r}"
        ) from None


def _schedule(text: str) -> Dict[str, int]:
    """Parse "0:0, 3:2, 6:5" into {epoch: trainable_backbone_layers}."""
    schedule: Dict[str, int] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"gradual schedule entries must look like 'epoch:layers', got {part!r}"
            )
        epoch, layers = part.split(":", 1)
        schedule[str(int(epoch.strip()))] = int(layers.strip())
    if not schedule:
        raise ValueError(
            "the gradual strategy needs a schedule, e.g. '0:0, 3:2, 6:5'"
        )
    return schedule


def _scheduler_kwargs(
    name: str, milestones: str, gamma: float, epochs: int
) -> Dict[str, Any]:
    if name == "multistep":
        return {"milestones": _int_list(milestones, "milestones"), "gamma": float(gamma)}
    if name == "step":
        return {"step_size": max(1, epochs // 3), "gamma": float(gamma)}
    if name == "cosine":
        return {"T_max": int(epochs)}
    if name == "plateau":
        return {"mode": "max", "patience": 2}
    return {}


def _strategy_summary(config: Config) -> str:
    strategy = config.finetune.strategy
    if strategy == "head_only":
        return "backbone frozen, only the new prediction head trains"
    if strategy == "partial":
        return (
            f"last {config.finetune.trainable_backbone_layers} backbone stage(s) "
            "plus the FPN and head train"
        )
    if strategy == "gradual":
        return f"unfreezing on schedule {dict(config.finetune.gradual_schedule)}"
    return "every parameter trains"


def _warnings(config: Config) -> List[str]:
    """Flag combinations that are legal but usually mistakes."""
    warnings: List[str] = []

    if config.finetune.strategy == "full" and config.train.epochs < 5:
        warnings.append(
            "Full finetuning for fewer than 5 epochs rarely beats `partial` — "
            "the backbone barely moves but is free to overfit."
        )
    if config.finetune.strategy == "head_only" and config.finetune.backbone_lr_mult:
        pass  # harmless: the backbone group is empty anyway
    if not config.model.anchors.enabled and config.train.checkpoint.monitor == "AP_small":
        warnings.append(
            "Selecting on `AP_small` while leaving anchors at torchvision's "
            "32–512 default: objects under ~32px cannot clear the RPN's IoU "
            "threshold against a 32px anchor, so they are invisible to the model."
        )
    if config.model.min_size < 640 and config.train.checkpoint.monitor == "AP_small":
        warnings.append(
            f"`min_size` is {config.model.min_size}px. The detector resizes every "
            "input to this, so a low value shrinks small objects further."
        )
    if config.optimizer.name == "sgd" and config.optimizer.lr > 0.02:
        warnings.append(
            f"lr {config.optimizer.lr:g} is high for SGD on a detector; losses "
            "commonly diverge above ~0.02 without a long warmup."
        )
    if config.optimizer.name in {"adam", "adamw"} and config.optimizer.lr > 0.001:
        warnings.append(
            f"lr {config.optimizer.lr:g} is high for {config.optimizer.name}; "
            "1e-4 is the usual starting point."
        )
    if not config.scheduler.warmup.enabled:
        warnings.append(
            "Warmup is off. A freshly initialised head produces large early "
            "gradients that can take the loss to NaN."
        )
    if config.scheduler.name == "multistep":
        milestones = config.scheduler.kwargs.get("milestones") or []
        if milestones and min(milestones) >= config.train.epochs:
            warnings.append(
                f"No LR milestone falls inside {config.train.epochs} epochs "
                f"({milestones}), so the learning rate never decays."
            )
    if config.data.train_loader.batch_size > 8:
        warnings.append(
            "Detection models are memory-heavy; batch sizes above 8 per device "
            "commonly run out of memory."
        )
    return warnings


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "–"
    return "n/a" if value <= UNDEFINED else f"{value:.4f}"
