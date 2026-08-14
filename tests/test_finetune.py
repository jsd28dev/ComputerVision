"""Finetuning strategies, parameter groups, optimizers, and schedules."""

from __future__ import annotations

import torch

from _support import expect_error

from smalldet.config import (
    AnchorConfig,
    FinetuneConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    WarmupConfig,
)
from smalldet.engine import (
    STRATEGIES,
    build_optimizer,
    build_scheduler,
    build_strategy,
    build_warmup,
    count_trainable,
    set_trainable_backbone_layers,
)
from smalldet.models import build_model


def _model():
    return build_model(
        ModelConfig(
            architecture="fasterrcnn_resnet50_fpn_v2",
            weights=None,
            weights_backbone=None,
            min_size=128,
            max_size=160,
            anchors=AnchorConfig(enabled=False),
        ),
        4,
    )


def _trainable(model, prefix: str) -> int:
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and name.startswith(prefix)
    )


# ------------------------------------------------------------------ strategies


def test_every_strategy_is_registered():
    assert set(STRATEGIES.names()) == {"full", "partial", "head_only", "gradual"}


def test_head_only_freezes_the_whole_backbone():
    """Transfer learning: the backbone becomes a fixed feature extractor."""
    model = _model()
    build_strategy(FinetuneConfig(strategy="head_only")).prepare(model)

    assert _trainable(model, "backbone.") == 0
    assert _trainable(model, "roi_heads.") > 0
    assert _trainable(model, "rpn.") > 0


def test_partial_freezes_early_stages_and_trains_later_ones():
    """Early convolutions learn transferable edges and textures; later stages
    encode semantics that usually do not transfer."""
    model = _model()
    build_strategy(
        FinetuneConfig(strategy="partial", trainable_backbone_layers=2)
    ).prepare(model)

    body = model.backbone.body
    frozen = {"conv1", "layer1", "layer2"}
    unfrozen = {"layer3", "layer4"}
    for name, parameter in body.named_parameters():
        stage = name.split(".")[0]
        if stage in frozen:
            assert not parameter.requires_grad, name
        elif stage in unfrozen:
            assert parameter.requires_grad, name


def test_partial_leaves_the_fpn_trainable():
    """The FPN is where multi-scale features are formed — the ones small
    objects depend on — and torchvision's own builders keep it trainable."""
    model = _model()
    build_strategy(
        FinetuneConfig(strategy="partial", trainable_backbone_layers=1)
    ).prepare(model)
    assert _trainable(model, "backbone.fpn") > 0


def test_zero_trainable_layers_freezes_the_body_but_not_the_fpn():
    model = _model()
    set_trainable_backbone_layers(model, 0)
    assert sum(p.numel() for p in model.backbone.body.parameters() if p.requires_grad) == 0


def test_full_trains_everything():
    model = _model()
    build_strategy(FinetuneConfig(strategy="full")).prepare(model)
    trainable, total = count_trainable(model)
    assert trainable == total


def test_strategies_are_ordered_by_how_much_they_train():
    """head_only < partial < full. If this ordering ever inverts, one of the
    strategies is silently not doing what its name says."""
    counts = {}
    for name, layers in (("head_only", 0), ("partial", 3), ("full", 5)):
        model = _model()
        build_strategy(
            FinetuneConfig(strategy=name, trainable_backbone_layers=layers)
        ).prepare(model)
        counts[name] = count_trainable(model)[0]
    assert counts["head_only"] < counts["partial"] < counts["full"]


def test_out_of_range_trainable_layers_is_rejected():
    model = _model()
    expect_error(
        lambda: set_trainable_backbone_layers(model, 9), ValueError, contains="[0, 5]"
    )


# --------------------------------------------------------------------- patterns


def test_freeze_and_unfreeze_patterns_compose_with_unfreeze_winning():
    model = _model()
    build_strategy(
        FinetuneConfig(
            strategy="full",
            freeze_patterns=[r"^backbone\."],
            unfreeze_patterns=[r"^backbone\.fpn\."],
        )
    ).prepare(model)

    assert _trainable(model, "backbone.body") == 0
    assert _trainable(model, "backbone.fpn") > 0


# ---------------------------------------------------------------- param groups


def test_backbone_gets_a_lower_learning_rate_than_the_head():
    """A randomly initialised head's early gradients would otherwise drag a
    good pretrained backbone away from its initialisation."""
    model = _model()
    config = FinetuneConfig(strategy="full", backbone_lr_mult=0.1)
    strategy = build_strategy(config)
    strategy.prepare(model)

    groups = strategy.param_groups(model, base_lr=0.01, weight_decay=0.0005)
    by_name = {group["name"]: group for group in groups}
    assert by_name["backbone"]["lr"] == 0.001
    assert by_name["head"]["lr"] == 0.01


def test_param_groups_exclude_frozen_parameters():
    """Passing frozen parameters to the optimizer wastes state and can mask a
    freezing bug."""
    model = _model()
    strategy = build_strategy(FinetuneConfig(strategy="head_only"))
    strategy.prepare(model)

    groups = strategy.param_groups(model, 0.01, 0.0005)
    assert all(
        all(p.requires_grad for p in group["params"]) for group in groups
    )
    assert "backbone" not in {group["name"] for group in groups}


def test_a_fully_frozen_model_is_reported_rather_than_silently_training_nothing():
    model = _model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    strategy = build_strategy(FinetuneConfig(strategy="full"))
    expect_error(
        lambda: strategy.param_groups(model, 0.01, 0.0),
        ValueError,
        contains="no trainable parameters",
    )


# ---------------------------------------------------------------------- gradual


def test_gradual_unfreezing_hands_over_more_of_the_backbone_over_time():
    model = _model()
    strategy = build_strategy(
        FinetuneConfig(strategy="gradual", gradual_schedule={"0": 0, "2": 2, "4": 5})
    )
    strategy.prepare(model)
    at_start = count_trainable(model)[0]

    assert strategy.on_epoch_start(model, 1) is False  # no schedule entry yet
    assert strategy.on_epoch_start(model, 2) is True
    at_middle = count_trainable(model)[0]

    assert strategy.on_epoch_start(model, 5) is True  # last entry still applies
    at_end = count_trainable(model)[0]

    assert at_start < at_middle < at_end


def test_gradual_requires_a_schedule():
    expect_error(
        lambda: build_strategy(FinetuneConfig(strategy="gradual")),
        ValueError,
        contains="gradual_schedule",
    )


def test_unknown_strategy_lists_the_valid_ones():
    error = expect_error(
        lambda: build_strategy(FinetuneConfig(strategy="freeze_everything")),
        Exception,
        contains="freeze_everything",
    )
    assert "partial" in str(error)


# ------------------------------------------------------- optimizers/schedulers


def test_optimizer_is_built_over_the_strategy_groups():
    model = _model()
    strategy = build_strategy(FinetuneConfig(strategy="full", backbone_lr_mult=0.1))
    strategy.prepare(model)
    optimizer = build_optimizer(
        OptimizerConfig(name="sgd", lr=0.01, kwargs={"momentum": 0.9}),
        strategy.param_groups(model, 0.01, 0.0005),
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert [group["lr"] for group in optimizer.param_groups] == [0.001, 0.01]


def test_adamw_is_available_for_transformer_style_heads():
    model = _model()
    strategy = build_strategy(FinetuneConfig(strategy="head_only"))
    strategy.prepare(model)
    optimizer = build_optimizer(
        OptimizerConfig(name="adamw", lr=1e-4, kwargs={}),
        strategy.param_groups(model, 1e-4, 0.01),
    )
    assert isinstance(optimizer, torch.optim.AdamW)


def test_warmup_ramps_the_learning_rate_from_near_zero():
    """Without warmup, a fresh head's early gradients take the loss to NaN."""
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    warmup = build_warmup(
        SchedulerConfig(warmup=WarmupConfig(enabled=True, iters=10, start_factor=0.001)),
        optimizer,
        steps_per_epoch=100,
    )
    assert warmup is not None
    start = optimizer.param_groups[0]["lr"]
    assert start < 0.01 / 100  # begins essentially at zero

    for _ in range(10):
        warmup.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-9  # reaches the base LR


def test_warmup_can_be_disabled():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    assert (
        build_warmup(
            SchedulerConfig(warmup=WarmupConfig(enabled=False)), optimizer, 10
        )
        is None
    )


def test_schedulers_decay_as_configured():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = build_scheduler(
        SchedulerConfig(name="multistep", kwargs={"milestones": [2], "gamma": 0.1}),
        optimizer,
    )
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.1) < 1e-9
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-9


def test_cosine_schedule_takes_its_horizon_from_the_epoch_count():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = build_scheduler(
        SchedulerConfig(name="cosine", kwargs={}), optimizer, epochs=7
    )
    assert scheduler.T_max == 7
