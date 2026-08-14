"""Anchor pyramids sized for small objects.

torchvision's default RPN pyramid is ``((32,), (64,), (128,), (256,), (512,))``
— the smallest anchor is 32px on a side. An anchor is labelled positive when
its IoU with a ground-truth box clears ~0.7, and a 12px object can never reach
that against a 32px anchor no matter where it sits. Such objects are therefore
never sampled as RPN positives, produce no proposals, and are invisible to the
second stage. Lowering the base sizes is the single highest-leverage change
available for ``AP_small``, and costs nothing at inference time.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from torchvision.models.detection.anchor_utils import AnchorGenerator

from ..config import AnchorConfig


def pyramid_sizes(
    base_sizes: Sequence[int], scales_per_octave: int = 1
) -> Tuple[Tuple[int, ...], ...]:
    """Expand one base size per level into ``scales_per_octave`` anchors each.

    With ``scales_per_octave=3`` this reproduces RetinaNet's convention of
    ``2^0, 2^(1/3), 2^(2/3)`` multiples, which fills the gaps between pyramid
    levels — worth having when object sizes cluster between two levels.
    """
    if not base_sizes:
        raise ValueError("base_sizes must not be empty")
    if scales_per_octave < 1:
        raise ValueError(f"scales_per_octave must be >= 1 (got {scales_per_octave})")
    if any(size < 1 for size in base_sizes):
        raise ValueError(f"anchor base sizes must be >= 1 (got {list(base_sizes)})")

    return tuple(
        tuple(
            int(round(base * 2 ** (octave / scales_per_octave)))
            for octave in range(scales_per_octave)
        )
        for base in base_sizes
    )


def build_anchor_generator(
    config: AnchorConfig, num_levels: int | None = None
) -> AnchorGenerator:
    """Build an :class:`AnchorGenerator` from config.

    ``num_levels`` is the number of feature maps the backbone emits. It is
    passed in from the live model rather than assumed, because an FPN with a
    ``LastLevelMaxPool`` produces five maps while a mobilenet FPN produces
    fewer, and a mismatch here surfaces as an opaque shape error deep in the RPN.
    """
    sizes = pyramid_sizes(config.base_sizes, config.scales_per_octave)
    if num_levels is not None and len(sizes) != num_levels:
        raise ValueError(
            f"model.anchors.base_sizes has {len(sizes)} entries but this backbone "
            f"produces {num_levels} feature map(s). Provide one base size per level."
        )
    aspect_ratios = (tuple(float(r) for r in config.aspect_ratios),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


def anchors_per_location(config: AnchorConfig) -> int:
    """How many anchors the generator will place at each spatial position."""
    return config.scales_per_octave * len(config.aspect_ratios)


def describe(generator: AnchorGenerator) -> List[str]:
    """Human-readable pyramid, for the training log."""
    return [
        f"level {level}: sizes={list(sizes)} ratios={list(ratios)}"
        for level, (sizes, ratios) in enumerate(
            zip(generator.sizes, generator.aspect_ratios)
        )
    ]
