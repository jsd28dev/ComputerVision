"""Stable per-class colours.

Colour is assigned by class *index*, not by order of appearance, so the same
class is the same colour in every image of a run and across runs. A palette
that shuffles per image makes side-by-side comparison useless.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

#: Used when matplotlib is unavailable. Qualitative, reasonably distinguishable
#: at the 1-2px line widths small-object boxes are drawn with.
FALLBACK_PALETTE: Sequence[str] = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffe119", "#00ced1",
)


def build_palette(
    num_classes: int,
    colormap: str = "tab20",
    class_names: Optional[Sequence[str]] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """One hex colour per class index, with config overrides applied by name."""
    colors = _from_colormap(num_classes, colormap) or [
        FALLBACK_PALETTE[index % len(FALLBACK_PALETTE)] for index in range(num_classes)
    ]

    if overrides and class_names:
        for name, color in overrides.items():
            if name in class_names:
                colors[class_names.index(name)] = color
    return colors


def _from_colormap(num_classes: int, colormap: str) -> Optional[List[str]]:
    try:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex
    except ImportError:
        return None
    try:
        cmap = colormaps[colormap]
    except KeyError:
        raise ValueError(
            f"unknown visualize.palette {colormap!r}; use a matplotlib colormap "
            "name such as tab10, tab20, or Set3"
        ) from None

    # Qualitative colormaps hold a fixed set of colours and should be sampled by
    # index; continuous ones are sampled evenly across their range.
    if getattr(cmap, "colors", None) is not None:
        entries = list(cmap.colors)
        return [to_hex(entries[index % len(entries)]) for index in range(num_classes)]
    span = max(num_classes - 1, 1)
    return [to_hex(cmap(index / span)) for index in range(num_classes)]
