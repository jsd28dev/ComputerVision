"""Draw boxes, masks, and ground-truth comparisons, all driven by config.

Every knob (line width, whether scores are shown, colours, mask alpha, the
small-object highlight) comes from :class:`VisualizationConfig`, so a figure in
a report and a frame in the Gradio UI are rendered by the same code with the
same settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from PIL import Image as PILImage
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from ..config import VisualizationConfig
from .palette import build_palette


class Renderer:
    """Renders detections onto uint8 images."""

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or VisualizationConfig()
        self.class_names = list(class_names or [])
        self.palette = build_palette(
            max(len(self.class_names), 1),
            self.config.palette,
            self.class_names,
            self.config.class_colors,
        )

    # -------------------------------------------------------------- rendering

    def draw(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        *,
        color: Optional[str] = None,
    ) -> torch.Tensor:
        """Overlay detections on a uint8 ``(3, H, W)`` image."""
        canvas = _as_uint8(image)

        if masks is not None and self.config.draw_masks and masks.numel():
            canvas = self._draw_masks(canvas, masks, labels, color)

        if boxes is None or boxes.numel() == 0:
            return canvas

        boxes = boxes.detach().cpu().float()
        colors = self._box_colors(boxes, labels, color)
        text = self._box_labels(labels, scores)

        # torchvision rejects font_size unless font is also set, and there is no
        # sensible default TTF path across platforms, so the size only applies
        # when a font file was configured.
        font_kwargs: Dict[str, Any] = {}
        if self.config.font:
            font_kwargs = {"font": self.config.font, "font_size": self.config.font_size}

        return draw_bounding_boxes(
            canvas,
            boxes=boxes,
            labels=text,
            colors=colors,
            width=self.config.box_width,
            fill=self.config.fill_boxes,
            **font_kwargs,
        )

    def draw_result(self, result: Any) -> torch.Tensor:
        """Render a :class:`~smalldet.inference.PredictionResult`."""
        if result.image is None:
            raise ValueError(
                "this PredictionResult carries no image; predict() keeps one so "
                "the boxes can be drawn without reloading the file"
            )
        return self.draw(
            result.image,
            result.boxes,
            result.labels,
            result.scores,
            self._binarize(result.masks),
        )

    def draw_ground_truth(
        self, image: torch.Tensor, target: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Render annotations, in the dedicated ground-truth colour."""
        return self.draw(
            image,
            target["boxes"],
            target.get("labels"),
            None,
            self._binarize(target.get("masks")),
            color=self.config.ground_truth_color,
        )

    def compare(
        self,
        image: torch.Tensor,
        target: Mapping[str, torch.Tensor],
        prediction: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Ground truth beside predictions, on one canvas.

        Scalar metrics tell you a run got worse; this tells you whether it
        started missing objects, started hallucinating them, or shifted boxes.
        """
        left = self.draw_ground_truth(image, target)
        right = self.draw(
            image,
            prediction["boxes"],
            prediction.get("labels"),
            prediction.get("scores"),
            self._binarize(prediction.get("masks")),
        )
        if not self.config.side_by_side:
            return right
        gap = torch.zeros(
            (3, left.shape[1], 4), dtype=torch.uint8, device=left.device
        )
        return torch.cat([left, gap, right], dim=2)

    # ------------------------------------------------------------------ output

    @staticmethod
    def to_pil(image: torch.Tensor) -> "PILImage.Image":
        return PILImage.fromarray(
            _as_uint8(image).permute(1, 2, 0).cpu().numpy()
        )

    def save(self, image: torch.Tensor, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_pil(image).save(target)
        return target

    # ------------------------------------------------------------------ detail

    def _box_colors(
        self,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor],
        color: Optional[str],
    ) -> List[str]:
        override = color or self.config.prediction_color
        if override:
            colors = [override] * len(boxes)
        elif labels is None:
            colors = [self.palette[0]] * len(boxes)
        else:
            colors = [
                self.palette[int(label) % len(self.palette)] for label in labels
            ]

        if self.config.highlight_small_objects:
            # Recolouring the small ones makes it immediately visible whether a
            # regression is concentrated in the bucket AP_small measures.
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            colors = [
                self.config.small_object_color
                if float(area) < self.config.small_area_threshold
                else existing
                for area, existing in zip(areas, colors)
            ]
        return colors

    def _box_labels(
        self, labels: Optional[torch.Tensor], scores: Optional[torch.Tensor]
    ) -> Optional[List[str]]:
        if not self.config.show_labels:
            return None
        if labels is None:
            return None
        names = [
            self.class_names[int(label)]
            if 0 <= int(label) < len(self.class_names)
            else f"class_{int(label)}"
            for label in labels
        ]
        if scores is None or not self.config.show_scores:
            return names
        return [
            f"{name} {self.config.score_format.format(float(score))}"
            for name, score in zip(names, scores)
        ]

    def _draw_masks(
        self,
        canvas: torch.Tensor,
        masks: torch.Tensor,
        labels: Optional[torch.Tensor],
        color: Optional[str],
    ) -> torch.Tensor:
        override = color or self.config.prediction_color
        if override:
            colors: Any = [override] * masks.shape[0]
        elif labels is not None:
            colors = [self.palette[int(label) % len(self.palette)] for label in labels]
        else:
            colors = None
        return draw_segmentation_masks(
            canvas, masks, alpha=self.config.mask_alpha, colors=colors
        )

    def _binarize(self, masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Mask R-CNN emits soft (N, 1, H, W) masks; the drawing utility needs
        hard (N, H, W) boolean ones."""
        if masks is None or masks.numel() == 0:
            return None
        masks = masks.detach().cpu()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.dtype != torch.bool:
            masks = masks > self.config.mask_threshold
        return masks


def _as_uint8(image: torch.Tensor) -> torch.Tensor:
    """draw_bounding_boxes needs uint8 in [0, 255], not the model's float input."""
    tensor = image.detach().cpu()
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.is_floating_point():
        scale = 255.0 if float(tensor.max()) <= 1.0 else 1.0
        return (tensor * scale).clamp(0, 255).to(torch.uint8)
    return tensor.to(torch.uint8)
