"""Sliced (tiled) inference for small objects in large frames.

A detector resizes its input to ``min_size`` internally. Feed it a 4000x3000
industrial frame with ``min_size=800`` and every object shrinks by 3.75x — a
20px part becomes 5px and is gone. Running the detector over overlapping crops
at native resolution keeps objects at their true scale, at the cost of one
forward pass per tile.

The overlap is what makes it correct rather than merely faster-looking: an
object straddling a tile boundary is truncated in both tiles, so tiles are
strided by ``(1 - overlap) * tile_size`` to guarantee any object smaller than
the overlap band lies wholly inside at least one tile. Duplicates from the
shared bands are then resolved by NMS.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
from torchvision.ops import batched_nms

from ..config import TilingConfig

Tile = Tuple[int, int, int, int]  # x0, y0, x1, y1


def generate_tiles(
    width: int,
    height: int,
    tile_size: Sequence[int],
    overlap: float = 0.2,
    include_full_image: bool = True,
) -> List[Tile]:
    """Cover a ``width`` x ``height`` frame with overlapping tiles.

    The last tile in each direction is pulled back flush with the edge instead
    of being padded, so no tile is partly empty and no detection is made
    against padding.
    """
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    if tile_w < 1 or tile_h < 1:
        raise ValueError(f"tile_size must be positive (got {list(tile_size)})")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must lie in [0, 1) (got {overlap})")

    tiles: List[Tile] = []
    if tile_w >= width and tile_h >= height:
        # The frame already fits in one tile; tiling would be a no-op.
        return [(0, 0, width, height)]

    stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))))

    for y0 in _starts(height, tile_h, stride_y):
        for x0 in _starts(width, tile_w, stride_x):
            tiles.append((x0, y0, min(x0 + tile_w, width), min(y0 + tile_h, height)))

    if include_full_image:
        # Catches objects larger than one tile, which no crop can contain.
        tiles.append((0, 0, width, height))
    return tiles


def _starts(extent: int, tile: int, stride: int) -> List[int]:
    if tile >= extent:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] + tile < extent:
        starts.append(extent - tile)
    return starts


def crop_tiles(image: torch.Tensor, tiles: Sequence[Tile]) -> List[torch.Tensor]:
    """Cut ``image`` (C, H, W) into the given tiles."""
    return [image[:, y0:y1, x0:x1] for (x0, y0, x1, y1) in tiles]


def merge_tile_predictions(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    tiles: Sequence[Tile],
    iou_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Shift tile-local boxes back to frame coordinates and de-duplicate.

    NMS is class-aware: two different parts genuinely overlapping in the same
    place is a real (if rare) configuration, and suppressing across classes
    would silently drop one of them.
    """
    if len(predictions) != len(tiles):
        raise ValueError(
            f"got {len(predictions)} prediction(s) for {len(tiles)} tile(s)"
        )

    boxes: List[torch.Tensor] = []
    scores: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for prediction, (x0, y0, _, _) in zip(predictions, tiles):
        tile_boxes = prediction["boxes"]
        if tile_boxes.numel() == 0:
            continue
        offset = torch.tensor(
            [x0, y0, x0, y0], dtype=tile_boxes.dtype, device=tile_boxes.device
        )
        boxes.append(tile_boxes + offset)
        scores.append(prediction["scores"])
        labels.append(prediction["labels"])

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros((0,), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }

    all_boxes = torch.cat(boxes)
    all_scores = torch.cat(scores)
    all_labels = torch.cat(labels)

    keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold)
    return {
        "boxes": all_boxes[keep],
        "scores": all_scores[keep],
        "labels": all_labels[keep],
    }


def tiles_for(image: torch.Tensor, config: TilingConfig) -> List[Tile]:
    """Tiles for one image, per config."""
    _, height, width = image.shape
    return generate_tiles(
        width, height, config.tile_size, config.overlap, config.include_full_image
    )
