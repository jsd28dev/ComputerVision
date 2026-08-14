"""The Gradio application: a framework-free service plus its UI wiring."""

from __future__ import annotations

from .finetune_service import FinetuneService, TrainingProgress, history_markdown
from .service import DetectionService, resolve_examples

__all__ = [
    "DetectionService",
    "ELEM_IDS",
    "FinetuneService",
    "TrainingProgress",
    "build_interface",
    "history_markdown",
    "launch_app",
    "resolve_examples",
]


def __getattr__(name: str):
    """Defer the Gradio import until the UI is actually asked for.

    ``smalldet.app.DetectionService`` must work in an environment without
    gradio installed — that is the whole point of the service/UI split.
    """
    if name in {"ELEM_IDS", "build_interface", "launch_app"}:
        from . import gradio_app

        return getattr(gradio_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
