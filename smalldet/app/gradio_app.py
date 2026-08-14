"""The Gradio UI: a Detect page and a Finetune page.

This module is wiring only — every decision lives in the config or in the two
services (:class:`~smalldet.app.service.DetectionService` and
:class:`~smalldet.app.finetune_service.FinetuneService`), neither of which
imports Gradio. Gradio itself is imported lazily so the rest of the package
stays usable without it.

Every interactive component carries an explicit ``elem_id``. Gradio's generated
ids change between releases and between layout edits, so the end-to-end tests
select on these instead; they are part of this module's contract, not an
implementation detail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..inference import Predictor
from ..visualization import Renderer
from .finetune_service import (
    MONITOR_CHOICES,
    OPTIMIZER_CHOICES,
    SCHEDULER_CHOICES,
    FinetuneService,
    history_markdown,
    split_summary_markdown,
)
from .service import DetectionService, resolve_examples

#: Selectors shared with tests/e2e/test_gradio_app.py.
ELEM_IDS = {
    # Detect page
    "input_image": "sd-input-image",
    "score_threshold": "sd-score-threshold",
    "max_detections": "sd-max-detections",
    "use_tiling": "sd-use-tiling",
    "highlight_small": "sd-highlight-small",
    "detect_button": "sd-detect-button",
    "clear_button": "sd-clear-button",
    "output_image": "sd-output-image",
    "summary": "sd-summary",
    "detections_json": "sd-detections-json",
    "model_summary": "sd-model-summary",
    # Finetune page. gr.Tab does not forward elem_id to a stable DOM node, so
    # the page is located by its controls rather than by a tab id.
    "ft_data_root": "sd-ft-data-root",
    "ft_train_images": "sd-ft-train-images",
    "ft_train_annotations": "sd-ft-train-annotations",
    "ft_val_images": "sd-ft-val-images",
    "ft_val_annotations": "sd-ft-val-annotations",
    "ft_test_images": "sd-ft-test-images",
    "ft_test_annotations": "sd-ft-test-annotations",
    "ft_split_source": "sd-ft-split-source",
    "ft_split_ratios": "sd-ft-split-ratios",
    "ft_split_button": "sd-ft-split-button",
    "ft_split_output": "sd-ft-split-output",
    "ft_architecture": "sd-ft-architecture",
    "ft_pretrained": "sd-ft-pretrained",
    "ft_min_size": "sd-ft-min-size",
    "ft_max_size": "sd-ft-max-size",
    "ft_anchors_enabled": "sd-ft-anchors-enabled",
    "ft_anchor_sizes": "sd-ft-anchor-sizes",
    "ft_strategy": "sd-ft-strategy",
    "ft_trainable_layers": "sd-ft-trainable-layers",
    "ft_backbone_lr_mult": "sd-ft-backbone-lr-mult",
    "ft_gradual_schedule": "sd-ft-gradual-schedule",
    "ft_optimizer": "sd-ft-optimizer",
    "ft_learning_rate": "sd-ft-learning-rate",
    "ft_weight_decay": "sd-ft-weight-decay",
    "ft_momentum": "sd-ft-momentum",
    "ft_scheduler": "sd-ft-scheduler",
    "ft_milestones": "sd-ft-milestones",
    "ft_warmup_enabled": "sd-ft-warmup-enabled",
    "ft_warmup_iters": "sd-ft-warmup-iters",
    "ft_epochs": "sd-ft-epochs",
    "ft_batch_size": "sd-ft-batch-size",
    "ft_accumulate": "sd-ft-accumulate",
    "ft_monitor": "sd-ft-monitor",
    "ft_auto_areas": "sd-ft-auto-areas",
    "ft_early_stopping": "sd-ft-early-stopping",
    "ft_output_dir": "sd-ft-output-dir",
    "ft_validate_button": "sd-ft-validate-button",
    "ft_start_button": "sd-ft-start-button",
    "ft_stop_button": "sd-ft-stop-button",
    "ft_export_button": "sd-ft-export-button",
    "ft_preview": "sd-ft-preview",
    "ft_log": "sd-ft-log",
    "ft_metrics": "sd-ft-metrics",
    "ft_status": "sd-ft-status",
    "ft_config_file": "sd-ft-config-file",
}


def _require_gradio() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised only without gradio
        raise ImportError(
            "the Gradio app needs the `gradio` package. Install it with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return gr


def build_interface(
    config: Config,
    predictor: Predictor,
    renderer: Optional[Renderer] = None,
    service: Optional[DetectionService] = None,
    finetune_service: Optional[FinetuneService] = None,
) -> Any:
    """Build (but do not launch) the two-page Blocks app."""
    gr = _require_gradio()

    renderer = renderer or Renderer(config.visualize, predictor.class_names)
    service = service or DetectionService(predictor, renderer, config)
    trainer_service = finetune_service or FinetuneService(config)
    app_config = config.app

    with gr.Blocks(title=app_config.title, analytics_enabled=False) as demo:
        gr.Markdown(f"# {app_config.title}")
        if app_config.description:
            gr.Markdown(app_config.description)

        with gr.Tabs():
            with gr.Tab("Detect", id="detect"):
                _detect_page(gr, config, predictor, service)
            with gr.Tab("Finetune", id="finetune"):
                _finetune_page(gr, config, trainer_service)

    # Bound so tests can drive the app without a browser.
    demo.smalldet_service = service  # type: ignore[attr-defined]
    demo.smalldet_finetune_service = trainer_service  # type: ignore[attr-defined]
    return demo


# ------------------------------------------------------------------ detect page


def _detect_page(gr: Any, config: Config, predictor: Predictor, service: DetectionService) -> None:
    app_config = config.app
    postprocess = predictor.config.postprocess

    with gr.Row():
        with gr.Column(scale=3):
            input_image = gr.Image(
                label="Input image", type="numpy", elem_id=ELEM_IDS["input_image"]
            )
            with gr.Row():
                detect_button = gr.Button(
                    "Detect", variant="primary", elem_id=ELEM_IDS["detect_button"]
                )
                clear_button = gr.Button("Clear", elem_id=ELEM_IDS["clear_button"])

        with gr.Column(scale=2):
            # When threshold control is off the sliders still exist (the callback
            # signature must not change) but are locked to the config values, so
            # a demo reports reproducible numbers.
            interactive = app_config.allow_threshold_control
            score_threshold = gr.Slider(
                0.0, 1.0, value=postprocess.score_threshold, step=0.01,
                label="Confidence threshold",
                info="Detections below this score are hidden.",
                interactive=interactive,
                elem_id=ELEM_IDS["score_threshold"],
            )
            max_detections = gr.Slider(
                1, 500, value=postprocess.max_detections or 100, step=1,
                label="Max detections",
                info="Dense small-object frames need this well above the usual "
                "default of 100.",
                interactive=interactive,
                elem_id=ELEM_IDS["max_detections"],
            )
            use_tiling = gr.Checkbox(
                value=predictor.config.tiling.enabled,
                label="Tiled inference",
                info="Run the detector over overlapping crops at native "
                "resolution. Slower, but recovers small objects the internal "
                "resize would otherwise shrink away.",
                interactive=app_config.show_tiling_control,
                elem_id=ELEM_IDS["use_tiling"],
            )
            highlight_small = gr.Checkbox(
                value=config.visualize.highlight_small_objects,
                label="Highlight small objects",
                info=f"Outline objects under "
                f"{config.visualize.small_area_threshold:.0f} px² separately.",
                elem_id=ELEM_IDS["highlight_small"],
            )
            gr.Markdown(service.model_summary(), elem_id=ELEM_IDS["model_summary"])

    with gr.Row():
        output_image = gr.Image(
            label="Detections", type="pil", elem_id=ELEM_IDS["output_image"]
        )
        summary = gr.Markdown(
            "Upload an image to run detection.", elem_id=ELEM_IDS["summary"]
        )

    detections_json = gr.JSON(
        label="Detections (JSON)",
        visible=app_config.show_json_output,
        elem_id=ELEM_IDS["detections_json"],
    )

    examples = resolve_examples(config)
    if examples:
        gr.Examples(examples=examples, inputs=input_image, label="Examples")

    inputs = [input_image, score_threshold, max_detections, use_tiling, highlight_small]
    outputs = [output_image, summary, detections_json]

    detect_button.click(
        fn=service.detect, inputs=inputs, outputs=outputs, api_name="detect"
    )
    clear_button.click(
        fn=lambda: (None, None, "Upload an image to run detection.", []),
        inputs=None,
        outputs=[input_image, *outputs],
    )


# ---------------------------------------------------------------- finetune page


def _finetune_page(gr: Any, config: Config, service: FinetuneService) -> None:
    """Dataset splits, model, finetuning strategy, and hyper-parameters.

    The page edits a config rather than holding its own state, so anything
    trained here is reproducible from the CLI with the exported YAML.
    """
    gr.Markdown(
        "Finetune a detector on your own COCO-format dataset. Every control "
        "below maps to one field of the same YAML config the CLI uses — press "
        "**Export config** to get a file you can rerun with "
        "`python -m smalldet.cli train`."
    )

    with gr.Accordion("1 · Dataset and splits", open=True):
        data_root = gr.Textbox(
            value=config.data.root,
            label="Dataset root",
            info="Every path below is relative to this directory.",
            elem_id=ELEM_IDS["ft_data_root"],
        )
        with gr.Row():
            train_images = gr.Textbox(
                value=config.data.train.images or "images",
                label="Train images", elem_id=ELEM_IDS["ft_train_images"],
            )
            train_annotations = gr.Textbox(
                value=config.data.train.annotations or "annotations_train.json",
                label="Train annotations (COCO JSON)",
                elem_id=ELEM_IDS["ft_train_annotations"],
            )
        with gr.Row():
            val_images = gr.Textbox(
                value=config.data.val.images or "images",
                label="Validation images", elem_id=ELEM_IDS["ft_val_images"],
            )
            val_annotations = gr.Textbox(
                value=config.data.val.annotations or "annotations_val.json",
                label="Validation annotations",
                info="Required — without it nothing is scored and no best "
                "checkpoint can be chosen.",
                elem_id=ELEM_IDS["ft_val_annotations"],
            )
        with gr.Row():
            test_images = gr.Textbox(
                value=config.data.test.images, label="Test images (optional)",
                elem_id=ELEM_IDS["ft_test_images"],
            )
            test_annotations = gr.Textbox(
                value=config.data.test.annotations,
                label="Test annotations (optional)",
                info="Held out entirely during training; score it afterwards "
                "with `smalldet.cli evaluate --split test`.",
                elem_id=ELEM_IDS["ft_test_annotations"],
            )

        gr.Markdown(
            "**Only have one annotation file?** Split it here. Splitting is "
            "done per *image*, never per annotation — putting two objects from "
            "the same image on opposite sides of a split leaks the exact pixels "
            "you are testing on into training."
        )
        with gr.Row():
            split_source = gr.Textbox(
                label="Annotation file to split",
                placeholder="annotations.json",
                elem_id=ELEM_IDS["ft_split_source"],
            )
            split_ratios = gr.Textbox(
                value="0.7, 0.15, 0.15",
                label="train / val / test ratios",
                info="Use 0 for a split you do not want.",
                elem_id=ELEM_IDS["ft_split_ratios"],
            )
            split_button = gr.Button("Split dataset", elem_id=ELEM_IDS["ft_split_button"])
        split_output = gr.Markdown(elem_id=ELEM_IDS["ft_split_output"])

    with gr.Accordion("2 · Model", open=True):
        with gr.Row():
            architecture = gr.Dropdown(
                choices=service.architectures(),
                value=config.model.architecture,
                label="Architecture",
                elem_id=ELEM_IDS["ft_architecture"],
            )
            pretrained = gr.Checkbox(
                value=bool(config.model.weights),
                label="Start from pretrained COCO weights",
                info="The head is replaced for your class count afterwards — "
                "the only order torchvision allows.",
                elem_id=ELEM_IDS["ft_pretrained"],
            )
        with gr.Row():
            min_size = gr.Number(
                value=config.model.min_size, precision=0, label="min_size (px)",
                info="The detector resizes every input to this. Raising it is "
                "the second-biggest lever on AP_small.",
                elem_id=ELEM_IDS["ft_min_size"],
            )
            max_size = gr.Number(
                value=config.model.max_size, precision=0, label="max_size (px)",
                elem_id=ELEM_IDS["ft_max_size"],
            )
        with gr.Row():
            anchors_enabled = gr.Checkbox(
                value=config.model.anchors.enabled,
                label="Custom anchor pyramid",
                info="torchvision's default starts at 32px. An object smaller "
                "than the smallest anchor can never clear the RPN's IoU "
                "threshold, so it is invisible to the model.",
                elem_id=ELEM_IDS["ft_anchors_enabled"],
            )
            anchor_sizes = gr.Textbox(
                value=", ".join(str(s) for s in config.model.anchors.base_sizes),
                label="Anchor base sizes (one per FPN level)",
                elem_id=ELEM_IDS["ft_anchor_sizes"],
            )
        detections_per_image = gr.Slider(
            10, 1000, value=300, step=10,
            label="Max detections per image (training/eval)",
            info="The default of 100 silently truncates dense frames.",
        )

    with gr.Accordion("3 · Finetuning strategy", open=True):
        gr.Markdown(service.strategy_help())
        with gr.Row():
            strategy = gr.Radio(
                choices=service.strategies(),
                value=config.finetune.strategy,
                label="Strategy",
                elem_id=ELEM_IDS["ft_strategy"],
            )
        with gr.Row():
            trainable_layers = gr.Slider(
                0, 5, value=config.finetune.trainable_backbone_layers, step=1,
                label="Trainable backbone stages (partial)",
                info="0 = frozen backbone, 5 = fully trainable. The FPN stays "
                "trainable either way.",
                elem_id=ELEM_IDS["ft_trainable_layers"],
            )
            backbone_lr_mult = gr.Number(
                value=config.finetune.backbone_lr_mult,
                label="Backbone LR multiplier",
                info="Pretrained features need smaller steps than a random head.",
                elem_id=ELEM_IDS["ft_backbone_lr_mult"],
            )
        gradual_schedule = gr.Textbox(
            value="0:0, 3:2, 6:5",
            label="Gradual unfreezing schedule (epoch:stages)",
            info="Only used by the `gradual` strategy.",
            elem_id=ELEM_IDS["ft_gradual_schedule"],
        )

    with gr.Accordion("4 · Hyper-parameters", open=True):
        with gr.Row():
            optimizer = gr.Dropdown(
                choices=OPTIMIZER_CHOICES, value=config.optimizer.name,
                label="Optimizer", elem_id=ELEM_IDS["ft_optimizer"],
            )
            learning_rate = gr.Number(
                value=config.optimizer.lr, label="Learning rate (head)",
                elem_id=ELEM_IDS["ft_learning_rate"],
            )
            weight_decay = gr.Number(
                value=config.optimizer.weight_decay, label="Weight decay",
                elem_id=ELEM_IDS["ft_weight_decay"],
            )
            momentum = gr.Number(
                value=0.9, label="Momentum (SGD/RMSprop)",
                elem_id=ELEM_IDS["ft_momentum"],
            )
        with gr.Row():
            scheduler = gr.Dropdown(
                choices=SCHEDULER_CHOICES, value=config.scheduler.name,
                label="LR schedule", elem_id=ELEM_IDS["ft_scheduler"],
            )
            milestones = gr.Textbox(
                value="16, 22", label="Milestones (multistep)",
                elem_id=ELEM_IDS["ft_milestones"],
            )
            warmup_enabled = gr.Checkbox(
                value=config.scheduler.warmup.enabled, label="Linear warmup",
                info="Stepped per iteration. Without it a fresh head's early "
                "gradients can take the loss to NaN.",
                elem_id=ELEM_IDS["ft_warmup_enabled"],
            )
            warmup_iters = gr.Number(
                value=config.scheduler.warmup.iters, precision=0,
                label="Warmup iterations", elem_id=ELEM_IDS["ft_warmup_iters"],
            )
        with gr.Row():
            epochs = gr.Slider(
                1, 200, value=config.train.epochs, step=1, label="Epochs",
                elem_id=ELEM_IDS["ft_epochs"],
            )
            batch_size = gr.Slider(
                1, 16, value=config.data.train_loader.batch_size, step=1,
                label="Batch size",
                info="Detection models are memory-heavy; 2–8 per device is typical.",
                elem_id=ELEM_IDS["ft_batch_size"],
            )
            accumulate = gr.Slider(
                1, 16, value=config.train.accumulate_steps, step=1,
                label="Gradient accumulation",
                info="Simulates a larger batch when memory is the limit.",
                elem_id=ELEM_IDS["ft_accumulate"],
            )
        with gr.Row():
            num_workers = gr.Slider(0, 8, value=0, step=1, label="Dataloader workers")
            seed = gr.Number(value=config.train.seed, precision=0, label="Seed")
            amp = gr.Checkbox(value=config.train.amp, label="Mixed precision (CUDA only)")
            grad_clip = gr.Number(value=0, label="Gradient clip (0 = off)")
        with gr.Row():
            horizontal_flip = gr.Checkbox(value=True, label="Augment: horizontal flip")
            photometric = gr.Checkbox(value=True, label="Augment: photometric distort")
            scale_jitter = gr.Checkbox(
                value=True, label="Augment: scale jitter (1.0–1.8×)",
                info="Biased upward on purpose — scaling below 1.0 shrinks "
                "objects that are already only a few pixels.",
            )

    with gr.Accordion("5 · Evaluation and checkpointing", open=True):
        with gr.Row():
            monitor = gr.Dropdown(
                choices=MONITOR_CHOICES, value=config.train.checkpoint.monitor,
                label="Select best checkpoint on",
                info="AP_small by default: a checkpoint that wins on overall AP "
                "by improving large objects is the wrong one here.",
                elem_id=ELEM_IDS["ft_monitor"],
            )
            auto_areas = gr.Checkbox(
                value=config.eval.auto_area_ranges,
                label="Derive area buckets from this dataset",
                info="COCO's 32²/96² cut-offs are tuned to COCO's resolution. On "
                "other data a bucket can end up empty, making AP_medium report -1.",
                elem_id=ELEM_IDS["ft_auto_areas"],
            )
        with gr.Row():
            early_stopping = gr.Checkbox(
                value=config.train.early_stopping.enabled, label="Early stopping",
                elem_id=ELEM_IDS["ft_early_stopping"],
            )
            patience = gr.Slider(
                1, 20, value=config.train.early_stopping.patience, step=1,
                label="Patience (epochs)",
            )
            output_dir = gr.Textbox(
                value="outputs/ui-run", label="Output directory",
                elem_id=ELEM_IDS["ft_output_dir"],
            )
            device = gr.Dropdown(
                choices=["auto", "cpu", "cuda"], value="auto", label="Device"
            )

    with gr.Row():
        validate_button = gr.Button(
            "Validate settings", elem_id=ELEM_IDS["ft_validate_button"]
        )
        start_button = gr.Button(
            "Start finetuning", variant="primary", elem_id=ELEM_IDS["ft_start_button"]
        )
        stop_button = gr.Button("Stop", elem_id=ELEM_IDS["ft_stop_button"])
        export_button = gr.Button("Export config", elem_id=ELEM_IDS["ft_export_button"])

    preview = gr.Markdown(
        "Press **Validate settings** to check the configuration before training.",
        elem_id=ELEM_IDS["ft_preview"],
    )
    status = gr.Markdown("_Idle._", elem_id=ELEM_IDS["ft_status"])
    with gr.Row():
        metrics = gr.Markdown(
            "_No epoch has been scored yet._", elem_id=ELEM_IDS["ft_metrics"]
        )
    log = gr.Textbox(
        label="Training log", lines=18, max_lines=18, interactive=False,
        elem_id=ELEM_IDS["ft_log"],
    )
    config_file = gr.File(label="Exported config", elem_id=ELEM_IDS["ft_config_file"])

    # The single ordered list of widgets whose values become a Config. Keeping
    # it in one place is what stops the three callbacks below from drifting.
    controls = [
        data_root, train_images, train_annotations, val_images, val_annotations,
        test_images, test_annotations,
        architecture, pretrained, min_size, max_size, anchors_enabled, anchor_sizes,
        detections_per_image,
        strategy, trainable_layers, backbone_lr_mult, gradual_schedule,
        optimizer, learning_rate, weight_decay, momentum,
        scheduler, milestones, warmup_enabled, warmup_iters,
        epochs, batch_size, accumulate, num_workers, seed, amp, grad_clip,
        horizontal_flip, photometric, scale_jitter,
        monitor, auto_areas, early_stopping, patience, output_dir, device,
    ]

    def _kwargs(*values: Any) -> Dict[str, Any]:
        names = [
            "data_root", "train_images", "train_annotations", "val_images",
            "val_annotations", "test_images", "test_annotations",
            "architecture", "pretrained", "min_size", "max_size",
            "anchors_enabled", "anchor_base_sizes", "detections_per_image",
            "strategy", "trainable_backbone_layers", "backbone_lr_mult",
            "gradual_schedule",
            "optimizer", "learning_rate", "weight_decay", "momentum",
            "scheduler", "milestones", "warmup_enabled", "warmup_iters",
            "epochs", "batch_size", "accumulate_steps", "num_workers", "seed",
            "amp", "grad_clip",
            "horizontal_flip", "photometric", "scale_jitter",
            "monitor", "auto_area_ranges", "early_stopping", "patience",
            "output_dir", "device",
        ]
        return dict(zip(names, values))

    def on_validate(*values: Any) -> str:
        return service.describe(**_kwargs(*values))

    def on_split(root: str, source: str, ratios: str) -> str:
        from ..data.split import split_coco

        if not source.strip():
            return "Enter the annotation file to split."
        try:
            parts = [float(p) for p in ratios.split(",")]
            written = split_coco(Path(root) / source, Path(root), parts)
        except Exception as exc:  # surfaced to the user, not the console
            return f"### ⚠️ Split failed\n\n```\n{exc}\n```"
        return split_summary_markdown(written)

    def on_export(*values: Any):
        target = Path(values[-2] or "outputs/ui-run") / "config.yaml"
        try:
            path = service.export_config(target, **_kwargs(*values))
        except Exception as exc:
            return None, f"### ⚠️ Export failed\n\n```\n{exc}\n```"
        return str(path), (
            f"Config written to `{path}`. Rerun this exact experiment with:\n\n"
            f"```\npython -m smalldet.cli train {path}\n```"
        )

    def on_start(*values: Any):
        for progress in service.run(**_kwargs(*values)):
            badge = {
                "starting": "⏳ Starting…",
                "training": "🔵 Training…",
                "finished": "✅ Finished",
                "stopped": "⏹️ Stopped",
                "failed": "❌ Failed",
            }.get(progress.status, progress.status)
            if progress.finished and not progress.failed:
                if progress.best_metric is not None:
                    badge += (
                        f" — best {progress.best_metric:.4f} at epoch "
                        f"{(progress.best_epoch or 0) + 1}"
                    )
                if progress.checkpoint:
                    badge += f"\n\nBest checkpoint: `{progress.checkpoint}`"
            yield badge, progress.log, history_markdown(progress.history)

    validate_button.click(
        fn=on_validate, inputs=controls, outputs=preview, api_name="validate_finetune"
    )
    split_button.click(
        fn=on_split,
        inputs=[data_root, split_source, split_ratios],
        outputs=split_output,
        api_name="split_dataset",
    )
    export_button.click(
        fn=on_export, inputs=controls, outputs=[config_file, preview],
        api_name="export_config",
    )
    start_button.click(
        fn=on_start, inputs=controls, outputs=[status, log, metrics],
        api_name="finetune",
    )
    stop_button.click(
        fn=lambda: f"⏹️ {service.request_stop()}", inputs=None, outputs=status,
        api_name="stop_finetune",
    )


# ----------------------------------------------------------------------- launch


def launch_app(
    config: Config,
    predictor: Optional[Predictor] = None,
    *,
    checkpoint: Optional[str] = None,
    share: Optional[bool] = None,
    server_port: Optional[int] = None,
    prevent_thread_lock: bool = False,
) -> Any:
    """Build and serve the app."""
    if predictor is None:
        from ..pipeline import build_predictor

        predictor = build_predictor(config, checkpoint=checkpoint)

    demo = build_interface(config, predictor)
    demo.queue(default_concurrency_limit=config.app.concurrency_limit)
    demo.launch(
        server_name=config.app.server_name,
        server_port=server_port or config.app.server_port,
        share=config.app.share if share is None else share,
        show_error=True,
        prevent_thread_lock=prevent_thread_lock,
    )
    return demo
