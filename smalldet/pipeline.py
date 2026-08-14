"""Assembly: turn a :class:`Config` into the objects that do the work.

The CLI, the tests, and the Gradio app all enter here rather than each wiring
datasets to loaders to models themselves. That is what makes "the config is the
experiment" true in practice — there is exactly one place where a config
becomes a running system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config, EvalConfig, dump_config
from .data import (
    CocoDetectionDataset,
    build_dataloader,
    build_dataset,
    build_transform,
)
from .engine import Trainer, TrainerState
from .evaluation import (
    EvalResult,
    GroundTruth,
    evaluate_model,
    resolve_eval_config,
    save_report,
)
from .inference import Predictor
from .models import build_model
from .runtime import resolve_device, seeded_generator, set_seed
from .visualization import Renderer


@dataclass
class Assembly:
    """Everything a run needs, built once and shared."""

    config: Config
    datasets: Dict[str, CocoDetectionDataset]
    loaders: Dict[str, DataLoader]
    class_names: List[str]
    num_classes: int

    def ground_truth(self, split: str = "val") -> GroundTruth:
        return GroundTruth.from_dataset(self.datasets[split])


def build_assembly(
    config: Config, splits: Sequence[str] = ("train", "val")
) -> Assembly:
    """Build the datasets and loaders named in ``config`` for ``splits``.

    Splits that are not configured are skipped rather than raising, so a
    predict-only config with no ``train`` section still assembles.
    """
    set_seed(config.train.seed)
    generator = seeded_generator(config.train.seed)

    transforms = {
        "train": build_transform(config.data.augmentation.train),
        "eval": build_transform(config.data.augmentation.eval),
    }

    datasets: Dict[str, CocoDetectionDataset] = {}
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        split_config = getattr(config.data, split, None)
        if split_config is None or not split_config.is_configured:
            continue
        is_train = split == "train"
        dataset = build_dataset(
            config.data, split, transforms["train" if is_train else "eval"]
        )
        datasets[split] = dataset
        loaders[split] = build_dataloader(
            dataset,
            config.data.train_loader if is_train else config.data.eval_loader,
            generator=generator if is_train else None,
        )

    if not datasets:
        raise ValueError(
            "no dataset split is configured. Set data.train.images / "
            "data.train.annotations (and the same for val) in the config."
        )

    reference = datasets.get("train") or next(iter(datasets.values()))
    class_names = list(reference.class_names)

    # Every split must agree on the label mapping, or a class index means one
    # thing during training and another during evaluation.
    for split, dataset in datasets.items():
        if dataset.class_names != class_names:
            raise ValueError(
                f"data.{split} declares different categories than the reference "
                f"split. All splits must share one `categories` array.\n"
                f"  reference: {class_names}\n  {split}: {dataset.class_names}"
            )

    return Assembly(
        config=config,
        datasets=datasets,
        loaders=loaders,
        class_names=class_names,
        num_classes=len(class_names),
    )


# --------------------------------------------------------------------- running


def run_training(
    config: Config,
    *,
    verbose: bool = True,
    extra_callbacks: Optional[Sequence[Any]] = None,
) -> TrainerState:
    """Finetune according to ``config`` and return the final trainer state.

    ``extra_callbacks`` are appended to the defaults rather than replacing them,
    which is how the Gradio finetuning page streams progress without losing
    checkpointing or early stopping.
    """
    assembly = build_assembly(config, ("train", "val"))
    if "train" not in assembly.loaders:
        raise ValueError("training needs data.train.images and data.train.annotations")

    model = build_model(config.model, assembly.num_classes)

    ground_truth: Optional[GroundTruth] = None
    eval_config: EvalConfig = config.eval
    if "val" in assembly.datasets:
        ground_truth = assembly.ground_truth("val")
        eval_config = resolve_eval_config(config, ground_truth, verbose=verbose)

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The exact config that produced a checkpoint, next to the checkpoint.
    dump_config(config, output_dir / "config.resolved.yaml")

    trainer = Trainer(
        config,
        model,
        assembly.loaders["train"],
        val_loader=assembly.loaders.get("val"),
        ground_truth=ground_truth,
        eval_config=eval_config,
    )
    for callback in extra_callbacks or ():
        trainer.callbacks.append(callback)
    state = trainer.fit()

    if verbose and state.best_metric is not None:
        print(
            f"\nbest {config.train.checkpoint.monitor}={state.best_metric:.4f} "
            f"at epoch {(state.best_epoch or 0) + 1}"
        )
    return state


def run_evaluation(
    config: Config,
    *,
    split: str = "val",
    checkpoint: Optional[str] = None,
    verbose: bool = True,
    report_dir: Optional[str | Path] = None,
) -> EvalResult:
    """Score a checkpoint on one split."""
    assembly = build_assembly(config, (split,))
    ground_truth = assembly.ground_truth(split)
    eval_config = resolve_eval_config(config, ground_truth, verbose=verbose)

    model_config = config.model
    if checkpoint:
        import dataclasses

        model_config = dataclasses.replace(model_config, checkpoint=checkpoint)
    model = build_model(model_config, assembly.num_classes)

    device = resolve_device(config.predict.device)
    result = evaluate_model(
        model,
        assembly.loaders[split],
        ground_truth,
        eval_config,
        device=device,
        max_batches=config.train.max_eval_batches,
    )

    if verbose:
        print(result.table())
    if report_dir is not None:
        save_report(result, report_dir, name=f"evaluation_{split}")
    return result


def build_predictor(
    config: Config,
    *,
    checkpoint: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Predictor:
    """Build a Predictor, taking class names from the dataset when available."""
    names = list(class_names) if class_names else None
    if names is None:
        for split in ("val", "train", "test"):
            split_config = getattr(config.data, split, None)
            if split_config is not None and split_config.is_configured:
                try:
                    names = list(build_dataset(config.data, split).class_names)
                    break
                except (FileNotFoundError, ValueError):
                    continue

    return Predictor.from_config(config, class_names=names, checkpoint=checkpoint)


def build_renderer(config: Config, class_names: Sequence[str]) -> Renderer:
    return Renderer(config.visualize, class_names)
