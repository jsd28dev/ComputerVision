"""The inference entrypoint shared by the CLI, the batch runner, and the UI.

One class owns "config in, detections out" so the Gradio app and a scripted
batch run cannot drift apart in how they threshold, tile, or name classes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
from PIL import Image as PILImage
from torch import nn

from ..config import Config, PostprocessConfig, PredictConfig
from ..models import build_model, load_checkpoint
from ..runtime import resolve_device
from .postprocess import apply_postprocess
from .tiling import crop_tiles, merge_tile_predictions, tiles_for

ImageLike = Union[str, Path, "PILImage.Image", torch.Tensor, Any]


@dataclass
class PredictionResult:
    """Detections for one image, plus the context needed to render them."""

    boxes: torch.Tensor  # (N, 4) XYXY in original image coordinates
    scores: torch.Tensor  # (N,)
    labels: torch.Tensor  # (N,) contiguous class indices
    class_names: List[str]
    #: The uint8 (3, H, W) image the boxes belong to, kept for drawing.
    image: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    elapsed_ms: float = 0.0
    num_tiles: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def areas(self) -> torch.Tensor:
        return (self.boxes[:, 2] - self.boxes[:, 0]) * (
            self.boxes[:, 3] - self.boxes[:, 1]
        )

    def names(self) -> List[str]:
        return [
            self.class_names[int(label)]
            if 0 <= int(label) < len(self.class_names)
            else f"class_{int(label)}"
            for label in self.labels
        ]

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name in self.names():
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def size_histogram(self, small: float = 32.0**2, medium: float = 96.0**2) -> Dict[str, int]:
        """How the detections fall into the COCO size buckets.

        The same partition the AP_small / AP_medium metrics use, so a glance at
        the UI tells you which bucket a run is actually exercising.
        """
        areas = self.areas
        return {
            "small": int((areas < small).sum()),
            "medium": int(((areas >= small) & (areas < medium)).sum()),
            "large": int((areas >= medium).sum()),
        }

    def to_records(self) -> List[Dict[str, Any]]:
        records = []
        for box, score, label, name in zip(
            self.boxes.tolist(), self.scores.tolist(), self.labels.tolist(), self.names()
        ):
            x1, y1, x2, y2 = box
            records.append(
                {
                    "label": int(label),
                    "class_name": name,
                    "score": round(float(score), 4),
                    "box_xyxy": [round(v, 2) for v in box],
                    "area": round((x2 - x1) * (y2 - y1), 1),
                }
            )
        return records


class Predictor:
    """Runs a detector over images, applying config-driven postprocessing."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Sequence[str],
        config: Optional[PredictConfig] = None,
        *,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.config = config or PredictConfig()
        self.device = resolve_device(
            str(device) if device is not None else self.config.device
        )
        self.model = model.to(self.device).eval()
        self.class_names = list(class_names)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        class_names: Optional[Sequence[str]] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "Predictor":
        """Build model and predictor from a whole config document.

        ``class_names`` normally comes from the dataset. When it is absent the
        names are synthesised so the UI still has something to label boxes with.
        """
        names = list(class_names or config.data.class_names)
        if names and names[0] != "__background__":
            names = ["__background__", *names]

        num_classes = config.model.num_classes or (len(names) if names else None)
        if num_classes is None:
            raise ValueError(
                "cannot determine the class count: set model.num_classes, or "
                "data.class_names, or pass class_names explicitly"
            )
        if not names:
            names = ["__background__"] + [
                f"class_{index}" for index in range(1, num_classes)
            ]

        model = build_model(config.model, num_classes)
        path = checkpoint or config.model.checkpoint
        if path:
            load_checkpoint(model, path, map_location="cpu")

        return cls(
            model,
            names,
            config.predict,
            device=device or config.predict.device,
        )

    # ---------------------------------------------------------------- predict

    @torch.inference_mode()
    def predict(
        self,
        image: ImageLike,
        *,
        postprocess: Optional[PostprocessConfig] = None,
        tiling: Optional[bool] = None,
    ) -> PredictionResult:
        """Detect objects in one image.

        ``postprocess`` and ``tiling`` override the config for this call, which
        is what lets the Gradio sliders take effect without rebuilding anything.
        """
        started = time.perf_counter()
        uint8_image = to_uint8_tensor(image)
        settings = postprocess or self.config.postprocess
        use_tiling = self.config.tiling.enabled if tiling is None else bool(tiling)

        if use_tiling:
            prediction, num_tiles = self._predict_tiled(uint8_image)
        else:
            prediction = self._forward([uint8_image])[0]
            num_tiles = 1

        prediction = apply_postprocess(prediction, settings)
        elapsed = (time.perf_counter() - started) * 1000.0

        return PredictionResult(
            boxes=prediction["boxes"].cpu(),
            scores=prediction["scores"].cpu(),
            labels=prediction["labels"].cpu(),
            masks=prediction["masks"].cpu() if "masks" in prediction else None,
            class_names=self.class_names,
            image=uint8_image,
            elapsed_ms=elapsed,
            num_tiles=num_tiles,
        )

    def predict_many(
        self,
        images: Sequence[ImageLike],
        *,
        postprocess: Optional[PostprocessConfig] = None,
        tiling: Optional[bool] = None,
    ) -> List[PredictionResult]:
        return [
            self.predict(image, postprocess=postprocess, tiling=tiling)
            for image in images
        ]

    # ----------------------------------------------------------------- detail

    def _forward(self, images: Sequence[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Run the model on uint8 images, in ``predict.batch_size`` chunks."""
        outputs: List[Dict[str, torch.Tensor]] = []
        chunk = max(1, self.config.batch_size)
        for start in range(0, len(images), chunk):
            batch = [
                # Detection models expect float in [0, 1]; they normalize
                # internally, so no mean/std belongs here.
                image.to(self.device).float().div(255.0)
                for image in images[start : start + chunk]
            ]
            outputs.extend(
                {key: value.detach() for key, value in output.items()}
                for output in self.model(batch)
            )
        return outputs

    def _predict_tiled(self, image: torch.Tensor) -> tuple[Dict[str, torch.Tensor], int]:
        tiles = tiles_for(image, self.config.tiling)
        crops = crop_tiles(image, tiles)
        predictions = self._forward(crops)
        merged = merge_tile_predictions(
            predictions, tiles, self.config.tiling.merge_nms_iou
        )
        return merged, len(tiles)


# ------------------------------------------------------------------ conversion


def to_uint8_tensor(image: ImageLike) -> torch.Tensor:
    """Normalise any supported input into a uint8 ``(3, H, W)`` tensor.

    uint8 is the canonical form here because it is what the visualization
    utilities require; the float conversion happens once, at the model boundary.
    """
    if isinstance(image, (str, Path)):
        with PILImage.open(image) as handle:
            return _from_pil(handle)

    if isinstance(image, PILImage.Image):
        return _from_pil(image)

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                f"expected a (C, H, W) image tensor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        if tensor.is_floating_point():
            # A float image is [0, 1] by convention, but a [0, 255] float array
            # is common enough coming out of NumPy that guessing wrong here
            # would produce an all-white render.
            scale = 255.0 if float(tensor.max()) <= 1.0 else 1.0
            tensor = (tensor * scale).clamp(0, 255).to(torch.uint8)
        else:
            tensor = tensor.to(torch.uint8)
        return _to_rgb(tensor)

    # numpy array or anything else array-like (this is what Gradio hands over)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        raise TypeError(f"unsupported image type: {type(image)!r}") from None

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image array, got shape {array.shape}")
    tensor = torch.from_numpy(array.copy())
    if tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.is_floating_point():
        scale = 255.0 if float(tensor.max()) <= 1.0 else 1.0
        tensor = (tensor * scale).clamp(0, 255)
    return _to_rgb(tensor.to(torch.uint8))


def _from_pil(handle: "PILImage.Image") -> torch.Tensor:
    from torchvision.transforms.v2 import functional as F

    return F.pil_to_tensor(handle.convert("RGB"))


def _to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[0]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(3, 1, 1)
    if channels == 4:
        return tensor[:3]  # drop alpha
    raise ValueError(f"expected 1, 3, or 4 channels, got {channels}")
