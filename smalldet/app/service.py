"""The detection service behind the UI, with no Gradio import anywhere.

Keeping the logic here means it can be unit-tested without a browser or a web
server, and it means the UI layer is only wiring. Every method takes plain
Python and returns plain Python (PIL images, strings, dicts).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image as PILImage

from ..config import Config, PostprocessConfig
from ..inference import Predictor
from ..visualization import Renderer


class DetectionService:
    """Runs one image through the predictor and formats the result for display."""

    def __init__(
        self,
        predictor: Predictor,
        renderer: Renderer,
        config: Optional[Config] = None,
    ) -> None:
        self.predictor = predictor
        self.renderer = renderer
        self.config = config or Config()

    # ------------------------------------------------------------------ detect

    def detect(
        self,
        image: Any,
        score_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        use_tiling: Optional[bool] = None,
        highlight_small: Optional[bool] = None,
    ) -> Tuple[Optional["PILImage.Image"], str, List[Dict[str, Any]]]:
        """Detect and render. Returns ``(image, markdown summary, records)``.

        Every override is optional and falls back to the config, so the same
        method serves the interactive UI and a pinned, non-interactive demo.
        """
        if image is None:
            return None, "Upload an image to run detection.", []

        settings = self._postprocess(score_threshold, max_detections)
        result = self.predictor.predict(image, postprocess=settings, tiling=use_tiling)

        renderer = self.renderer
        if highlight_small is not None and (
            highlight_small != self.renderer.config.highlight_small_objects
        ):
            renderer = Renderer(
                dataclasses.replace(
                    self.renderer.config, highlight_small_objects=highlight_small
                ),
                self.renderer.class_names,
            )

        rendered = renderer.to_pil(renderer.draw_result(result))
        return rendered, self.summarize(result, settings), result.to_records()

    def _postprocess(
        self, score_threshold: Optional[float], max_detections: Optional[int]
    ) -> PostprocessConfig:
        settings = self.predictor.config.postprocess
        changes: Dict[str, Any] = {}
        if score_threshold is not None:
            changes["score_threshold"] = float(score_threshold)
        if max_detections is not None:
            changes["max_detections"] = int(max_detections)
        return dataclasses.replace(settings, **changes) if changes else settings

    # ----------------------------------------------------------------- format

    def summarize(self, result: Any, settings: PostprocessConfig) -> str:
        """A compact Markdown report of what was found.

        The size histogram uses the same small/medium/large partition as
        ``AP_small``/``AP_medium``, so what the UI shows and what the metrics
        measure are the same thing.
        """
        if len(result) == 0:
            return (
                f"**No detections** above a score of {settings.score_threshold:.2f}.\n\n"
                "Lower the confidence threshold, or enable tiled inference if the "
                "objects are small relative to the frame."
            )

        buckets = result.size_histogram()
        counts = result.count_by_class()
        lines = [
            f"**{len(result)} detection(s)** in {result.elapsed_ms:.0f} ms"
            + (f" across {result.num_tiles} tiles" if result.num_tiles > 1 else ""),
            "",
            "| class | count |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {name} | {count} |" for name, count in counts.items())
        lines.extend(
            [
                "",
                "**By object size** (the buckets AP_small / AP_medium score):",
                "",
                "| bucket | area | count |",
                "| --- | --- | ---: |",
                f"| small | < 32² px | {buckets['small']} |",
                f"| medium | 32²–96² px | {buckets['medium']} |",
                f"| large | ≥ 96² px | {buckets['large']} |",
                "",
                f"Confidence ≥ {settings.score_threshold:.2f} · "
                f"score range {float(result.scores.min()):.2f}–"
                f"{float(result.scores.max()):.2f}",
            ]
        )
        return "\n".join(lines)

    def model_summary(self) -> str:
        """Static description of what is loaded, shown in the UI sidebar."""
        model_config = self.config.model
        anchors = model_config.anchors
        lines = [
            f"**Architecture** `{model_config.architecture}`",
            f"**Classes** {len(self.predictor.class_names) - 1} "
            f"({', '.join(self.predictor.class_names[1:]) or 'none'})",
            f"**Input resolution** min {model_config.min_size} / max {model_config.max_size} px",
            f"**Device** `{self.predictor.device}`",
        ]
        if anchors.enabled:
            lines.append(
                f"**Anchors** base sizes {anchors.base_sizes}, "
                f"{anchors.scales_per_octave} scale(s)/octave — tuned down from "
                "torchvision's 32–512 default for small objects"
            )
        if self.predictor.config.tiling.enabled:
            tiling = self.predictor.config.tiling
            lines.append(
                f"**Tiled inference** {tiling.tile_size[0]}×{tiling.tile_size[1]} "
                f"at {tiling.overlap:.0%} overlap"
            )
        if model_config.checkpoint:
            lines.append(f"**Checkpoint** `{model_config.checkpoint}`")
        else:
            lines.append(
                "**Checkpoint** none — running on "
                + ("pretrained COCO weights" if model_config.weights else "random weights")
            )
        return "\n\n".join(lines)


def resolve_examples(config: Config) -> List[str]:
    """Expand ``app.examples`` into existing image paths.

    Entries may be files, directories, or globs. Anything that does not resolve
    is dropped rather than raising — a missing example should not stop the app
    from starting.
    """
    resolved: List[str] = []
    for entry in config.app.examples:
        path = Path(entry)
        if path.is_file():
            resolved.append(str(path))
        elif path.is_dir():
            resolved.extend(
                str(child)
                for child in sorted(path.iterdir())
                if child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            )
        else:
            parent = path.parent if path.parent != Path("") else Path(".")
            if parent.is_dir():
                resolved.extend(str(match) for match in sorted(parent.glob(path.name)))
    return resolved[:12]
