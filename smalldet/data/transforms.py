"""YAML-driven augmentation pipelines built from ``torchvision.transforms.v2``.

A pipeline is a list of ``{name, params}`` entries in the config, so changing
augmentation never means editing Python:

    augmentation:
      train:
        - {name: random_horizontal_flip, params: {p: 0.5}}
        - {name: random_photometric_distort}
        - {name: sanitize_bounding_boxes, params: {min_size: 1.0}}
        - {name: to_dtype, params: {dtype: float32, scale: true}}
        - {name: to_pure_tensor}

Two rules the registry enforces for small-object work, because getting them
wrong is silent rather than loud:

* ``sanitize_bounding_boxes`` defaults to ``min_size=1.0``. torchvision's own
  default is also 1.0, but a copy-pasted recipe often sets it to 3 or 10, which
  deletes precisely the objects ``AP_small`` measures.
* ``random_zoom_out`` is registered but flagged: it shrinks every object in the
  frame, which is the opposite of what a small-object dataset needs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch
from torchvision.transforms import v2 as T

from ..config import AugmentationConfig, TransformOp
from ..registry import Registry

TRANSFORMS: Registry[Callable[..., Any]] = Registry("transform")

_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


def _dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[str(name).lower().replace("torch.", "")]
    except KeyError:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {', '.join(sorted(_DTYPES))}"
        ) from None


# ------------------------------------------------------------------ geometric


@TRANSFORMS.register("random_horizontal_flip")
def _random_horizontal_flip(p: float = 0.5) -> Any:
    return T.RandomHorizontalFlip(p=p)


@TRANSFORMS.register("random_vertical_flip")
def _random_vertical_flip(p: float = 0.5) -> Any:
    """Safe for top-down industrial capture; wrong for anything gravity-aware."""
    return T.RandomVerticalFlip(p=p)


@TRANSFORMS.register("random_rotation")
def _random_rotation(degrees: float | Sequence[float] = 10.0, expand: bool = False) -> Any:
    return T.RandomRotation(degrees=degrees, expand=expand)


@TRANSFORMS.register("random_affine")
def _random_affine(
    degrees: float | Sequence[float] = 0.0,
    translate: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
    shear: float | Sequence[float] | None = None,
) -> Any:
    return T.RandomAffine(
        degrees=degrees,
        translate=tuple(translate) if translate else None,
        scale=tuple(scale) if scale else None,
        shear=shear,
    )


@TRANSFORMS.register("resize")
def _resize(size: int | Sequence[int] = 800, max_size: int | None = None) -> Any:
    return T.Resize(size=size if isinstance(size, int) else list(size), max_size=max_size)


@TRANSFORMS.register("random_shortest_size")
def _random_shortest_size(
    min_size: int | Sequence[int] = (640, 672, 704, 736, 768, 800),
    max_size: int | None = 1333,
) -> Any:
    """Multi-scale training, as used by torchvision's detection reference.

    Sampling a shorter-side length per image is the cheapest way to make a
    detector scale-robust, and biasing the list upward enlarges small objects.
    """
    return T.RandomShortestSize(
        min_size=min_size if isinstance(min_size, int) else list(min_size),
        max_size=max_size,
    )


@TRANSFORMS.register("scale_jitter")
def _scale_jitter(
    target_size: Sequence[int] = (800, 800),
    scale_range: Sequence[float] = (0.8, 2.0),
) -> Any:
    """Large-scale jitter. Keep the lower bound near 1.0 for small objects —
    scaling below 1.0 shrinks objects that are already only a few pixels."""
    return T.ScaleJitter(
        target_size=tuple(int(value) for value in target_size),
        scale_range=tuple(float(value) for value in scale_range),
    )


@TRANSFORMS.register("random_iou_crop")
def _random_iou_crop() -> Any:
    """Must be followed by ``sanitize_bounding_boxes`` — it can leave boxes
    partially outside the crop."""
    return T.RandomIoUCrop()


@TRANSFORMS.register("random_zoom_out")
def _random_zoom_out(
    fill: float = 0.0,
    side_range: Sequence[float] = (1.0, 4.0),
    p: float = 0.5,
) -> Any:
    """Pads the frame, shrinking every object in it. Included for completeness;
    it works against ``AP_small`` on a dataset that is already small-object."""
    return T.RandomZoomOut(
        fill=fill, side_range=tuple(float(value) for value in side_range), p=p
    )


# --------------------------------------------------------------- photometric


@TRANSFORMS.register("random_photometric_distort")
def _random_photometric_distort(
    brightness: Sequence[float] = (0.875, 1.125),
    contrast: Sequence[float] = (0.5, 1.5),
    saturation: Sequence[float] = (0.5, 1.5),
    hue: Sequence[float] = (-0.05, 0.05),
    p: float = 0.5,
) -> Any:
    return T.RandomPhotometricDistort(
        brightness=tuple(brightness),
        contrast=tuple(contrast),
        saturation=tuple(saturation),
        hue=tuple(hue),
        p=p,
    )


@TRANSFORMS.register("color_jitter")
def _color_jitter(
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> Any:
    return T.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
    )


@TRANSFORMS.register("gaussian_blur")
def _gaussian_blur(
    kernel_size: int | Sequence[int] = 3, sigma: Sequence[float] = (0.1, 2.0)
) -> Any:
    return T.GaussianBlur(kernel_size=kernel_size, sigma=tuple(sigma))


# --------------------------------------------------------------- housekeeping


@TRANSFORMS.register("sanitize_bounding_boxes")
def _sanitize_bounding_boxes(min_size: float = 1.0, min_area: float = 1.0) -> Any:
    return T.SanitizeBoundingBoxes(min_size=min_size, min_area=min_area)


@TRANSFORMS.register("clamp_bounding_boxes")
def _clamp_bounding_boxes() -> Any:
    return T.ClampBoundingBoxes()


@TRANSFORMS.register("to_dtype")
def _to_dtype(dtype: str = "float32", scale: bool = True) -> Any:
    return T.ToDtype(_dtype(dtype), scale=scale)


@TRANSFORMS.register("normalize")
def _normalize(
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Any:
    """Rarely needed: torchvision detection models normalize internally via
    ``model.transform``. Normalizing twice quietly degrades every metric."""
    return T.Normalize(mean=list(mean), std=list(std))


@TRANSFORMS.register("to_pure_tensor")
def _to_pure_tensor() -> Any:
    """Unwraps tv_tensors back to plain tensors once augmentation is done."""
    return T.ToPureTensor()


# --------------------------------------------------------------------- public

#: Appended when a pipeline does not end in a dtype conversion. Detection
#: models need float input; forgetting this fails deep inside the backbone with
#: a dtype error that says nothing about the pipeline.
_DEFAULT_TAIL: List[TransformOp] = [
    TransformOp(name="to_dtype", params={"dtype": "float32", "scale": True}),
    TransformOp(name="to_pure_tensor"),
]


def build_transform(ops: Sequence[TransformOp], *, ensure_tail: bool = True) -> Any:
    """Compose a ``transforms.v2`` pipeline from config entries."""
    entries = list(ops)
    if ensure_tail and not any(op.name == "to_dtype" for op in entries):
        entries.extend(_DEFAULT_TAIL)

    built = []
    for index, op in enumerate(entries):
        factory = TRANSFORMS.get(op.name)
        try:
            built.append(factory(**op.params))
        except TypeError as exc:
            raise ValueError(
                f"augmentation step {index} ({op.name!r}): {exc}"
            ) from exc
    return T.Compose(built)


def build_transforms(config: AugmentationConfig) -> Dict[str, Any]:
    """Build the train and eval pipelines named in an augmentation config."""
    return {
        "train": build_transform(config.train),
        "eval": build_transform(config.eval),
    }
