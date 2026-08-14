"""smalldet — a configuration-driven small-object detection toolkit.

Everything the project does (finetuning, prediction, visualization, and the
Gradio UI) is described by one YAML document validated into the typed
:class:`smalldet.config.Config` tree. Python code never reads YAML directly and
never hard-codes a hyper-parameter; it asks the config object.

The metrics that matter here are ``AP_small`` and ``AP_medium`` at
IoU=[0.50:0.95], so the defaults throughout — anchor sizes, input resolution,
detections per image, augmentation choices, checkpoint monitor — are tuned for
objects that occupy a few dozen pixels rather than a third of the frame.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
