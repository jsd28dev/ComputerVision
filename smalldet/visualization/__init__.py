"""Rendering detections and plotting diagnostics."""

from __future__ import annotations

from .palette import FALLBACK_PALETTE, build_palette
from .renderer import Renderer

__all__ = [
    "FALLBACK_PALETTE",
    "Renderer",
    "build_palette",
    "plot_area_histogram",
    "plot_history",
    "plot_per_class",
    "plot_pr_curves",
]


def __getattr__(name: str):
    """Import the plotting helpers lazily.

    ``plots`` pulls in matplotlib, which costs about a second and is not needed
    to render boxes. Deferring it keeps ``import smalldet.visualization`` cheap
    for the inference path.
    """
    if name in {"plot_area_histogram", "plot_history", "plot_per_class", "plot_pr_curves"}:
        from . import plots

        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
