"""The training loop, callbacks, and the config-to-running-system assembly."""

from __future__ import annotations

import json

import torch

from _support import expect_error, scratch_dir, tiny_config

from smalldet.config import CheckpointConfig, EarlyStoppingConfig
from smalldet.engine import Trainer
from smalldet.engine.hooks import (
    CheckpointSaver,
    EarlyStopping,
    HistoryRecorder,
    TrainerState,
)
from smalldet.evaluation import GroundTruth
from smalldet.models import build_model
from smalldet.pipeline import build_assembly, run_training


# ------------------------------------------------------------------- assembly


def test_assembly_builds_datasets_loaders_and_class_names_from_config():
    assembly = build_assembly(tiny_config())
    assert set(assembly.loaders) == {"train", "val"}
    assert assembly.class_names[0] == "__background__"
    assert assembly.num_classes == len(assembly.class_names) == 4

    images, targets = next(iter(assembly.loaders["train"]))
    assert len(images) == len(targets)


def test_assembly_rejects_splits_that_disagree_on_categories():
    """A class index must mean the same thing in training and evaluation."""
    import json as json_module

    from smalldet.data.synthetic import generate_dataset

    root = scratch_dir("mismatched_splits")
    generate_dataset(root, num_images=2, image_size=(64, 64), seed=3)
    path = root / "annotations_val.json"
    document = json_module.loads(path.read_text(encoding="utf-8"))
    document["categories"] = document["categories"][:2]  # drop a class
    path.write_text(json_module.dumps(document), encoding="utf-8")

    config = tiny_config(data={"root": str(root)})
    expect_error(
        lambda: build_assembly(config), ValueError, contains="different categories"
    )


def test_unconfigured_split_is_reported_clearly():
    config = tiny_config(data={"train": {"images": "", "annotations": ""}})
    error = expect_error(lambda: build_assembly(config, ("train",)), ValueError)
    assert "data" in str(error)


# ---------------------------------------------------------------- the loop


def _trainer(config=None, **overrides):
    config = config or tiny_config(**overrides)
    assembly = build_assembly(config)
    model = build_model(config.model, assembly.num_classes)
    return (
        Trainer(
            config,
            model,
            assembly.loaders["train"],
            val_loader=assembly.loaders["val"],
            ground_truth=GroundTruth.from_dataset(assembly.datasets["val"]),
            callbacks=[],
        ),
        config,
    )


def test_one_epoch_runs_and_produces_metrics():
    trainer, config = _trainer()
    state = trainer.fit()

    assert state.global_step > 0
    assert "AP" in state.epoch_metrics
    assert "AP_small" in state.epoch_metrics
    # An untrained model scores badly, but the numbers must be real.
    for key in ("AP", "AP_small"):
        assert state.epoch_metrics[key] >= -1.0


def test_train_mode_returns_losses_and_they_are_finite():
    """The contract the whole loop rests on: train mode gives a loss dict, and
    each term must be logged separately so a stuck component is visible."""
    trainer, _ = _trainer()
    losses = trainer.train_one_epoch()

    assert "total" in losses
    for name in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
        assert name in losses, f"{name} was not logged"
        assert losses[name] == losses[name]  # not NaN
    assert losses["total"] > 0


def test_a_diverging_loss_stops_training_with_an_actionable_message():
    """NaN losses are the classic too-high-LR failure. Silently continuing
    wastes the whole run."""
    trainer, _ = _trainer()

    original = trainer.model.forward

    def exploding(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, dict):
            return {key: value * float("inf") for key, value in result.items()}
        return result

    trainer.model.forward = exploding
    error = expect_error(trainer.train_one_epoch, RuntimeError, contains="warmup")
    assert "learning rate" in str(error)


def test_gradient_accumulation_runs():
    trainer, _ = _trainer(train={"accumulate_steps": 2, "max_train_batches": 4})
    losses = trainer.train_one_epoch()
    assert losses["total"] > 0


def test_gradient_clipping_runs():
    trainer, _ = _trainer(train={"grad_clip": 1.0})
    assert trainer.train_one_epoch()["total"] > 0


def test_evaluation_without_a_val_loader_is_refused_clearly():
    config = tiny_config()
    assembly = build_assembly(config)
    trainer = Trainer(
        config,
        build_model(config.model, assembly.num_classes),
        assembly.loaders["train"],
        callbacks=[],
    )
    expect_error(trainer.evaluate, ValueError, contains="val_loader")


def test_gradual_unfreezing_rebuilds_the_optimizer():
    """A parameter absent from optimizer.param_groups is never updated, no
    matter what requires_grad says. Unfreezing without rebuilding is a no-op.
    """
    config = tiny_config(
        finetune={"strategy": "gradual", "gradual_schedule": {"0": 0, "1": 3}},
        train={"epochs": 2, "eval_interval": 0, "max_train_batches": 1},
    )
    assembly = build_assembly(config)
    trainer = Trainer(
        config,
        build_model(config.model, assembly.num_classes),
        assembly.loaders["train"],
        callbacks=[],
    )
    before = sum(len(group["params"]) for group in trainer.optimizer.param_groups)
    trainer.fit()
    after = sum(len(group["params"]) for group in trainer.optimizer.param_groups)
    assert after > before


# ------------------------------------------------------------------ callbacks


def test_checkpoint_saver_writes_last_and_best():
    trainer, config = _trainer()
    directory = scratch_dir("ckpt_cb", clean=True)
    saver = CheckpointSaver(CheckpointConfig(dir=str(directory), monitor="AP_small"))
    state = TrainerState(epoch=0, epoch_metrics={"AP_small": 0.4})

    saver.on_epoch_end(trainer, state)
    assert (directory / "last.pt").is_file()
    assert (directory / "best.pt").is_file()
    assert state.best_metric == 0.4

    # A worse epoch must not overwrite best.
    saver.on_epoch_end(trainer, TrainerState(epoch=1, epoch_metrics={"AP_small": 0.1}))
    payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    assert payload["metrics"]["AP_small"] == 0.4


def test_checkpoint_saver_ignores_the_undefined_sentinel():
    """-1 means "no ground truth in this bucket". Letting it win would pin the
    best checkpoint to epoch 1 forever."""
    trainer, _ = _trainer()
    directory = scratch_dir("ckpt_sentinel", clean=True)
    saver = CheckpointSaver(CheckpointConfig(dir=str(directory), monitor="AP_medium"))

    state = TrainerState(epoch=0, epoch_metrics={"AP_medium": -1.0})
    saver.on_epoch_end(trainer, state)
    assert state.best_metric is None
    assert not (directory / "best.pt").exists()

    saver.on_epoch_end(trainer, TrainerState(epoch=1, epoch_metrics={"AP_medium": 0.2}))
    assert (directory / "best.pt").is_file()


def test_early_stopping_fires_after_patience_is_exhausted():
    stopper = EarlyStopping(
        EarlyStoppingConfig(enabled=True, patience=2), monitor="AP_small"
    )
    state = TrainerState()

    for value in (0.5, 0.4, 0.45):
        state.epoch_metrics = {"AP_small": value}
        stopper.on_epoch_end(None, state)
    assert state.should_stop is True


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(
        EarlyStoppingConfig(enabled=True, patience=2), monitor="AP_small"
    )
    state = TrainerState()
    for value in (0.5, 0.4, 0.6, 0.5):
        state.epoch_metrics = {"AP_small": value}
        stopper.on_epoch_end(None, state)
    assert state.should_stop is False


def test_history_is_written_as_json():
    recorder = HistoryRecorder()
    state = TrainerState(output_dir=scratch_dir("history", clean=True))
    for epoch, value in enumerate([0.1, 0.2]):
        state.epoch = epoch
        state.epoch_metrics = {"AP": value}
        recorder.on_epoch_end(None, state)
    recorder.on_train_end(None, state)

    payload = json.loads((state.output_dir / "history.json").read_text(encoding="utf-8"))
    assert [entry["AP"] for entry in payload] == [0.1, 0.2]


# ----------------------------------------------------------------- end to end


def test_run_training_produces_a_usable_checkpoint_and_resolved_config():
    """The full path a user actually takes: config in, checkpoint out, and the
    checkpoint loads back into a model built from the same config."""
    output = scratch_dir("e2e_train", clean=True)
    config = tiny_config(
        train={
            "epochs": 1,
            "output_dir": str(output),
            "max_train_batches": 2,
            "max_eval_batches": 2,
            "callbacks": [],
            "checkpoint": {"dir": str(output / "checkpoints"), "monitor": "AP"},
        }
    )
    state = run_training(config, verbose=False)

    assert (output / "config.resolved.yaml").is_file()
    assert (output / "checkpoints" / "last.pt").is_file()
    assert "AP" in state.epoch_metrics

    from smalldet.models import load_checkpoint

    restored = build_model(config.model, 4)
    load_checkpoint(restored, str(output / "checkpoints" / "last.pt"))

    restored.eval()
    with torch.inference_mode():
        outputs = restored([torch.rand(3, 160, 160)])
    assert "boxes" in outputs[0]
