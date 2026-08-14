"""The typed configuration tree.

One YAML document describes a whole experiment. The sections are deliberately
independent so a single file can drive finetuning, batch prediction, and the
Gradio app without any of them reaching into each other's settings:

    data:       datasets, augmentation, dataloaders
    model:      architecture, pretrained weights, anchors, input resolution
    finetune:   which parameters train, and at what relative learning rate
    optimizer:  optimizer name and hyper-parameters
    scheduler:  LR schedule and warmup
    train:      loop mechanics, checkpointing, early stopping
    eval:       COCO metric parameters, including the small/medium area cuts
    predict:    inference-time postprocessing and tiled inference
    visualize:  how boxes, masks, and comparison grids are drawn
    app:        Gradio server and UI options

Defaults throughout target small objects. Where a default differs from the
torchvision norm, the reason is in a comment next to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# COCO's own small/medium cut-offs, in pixels^2: 32^2 and 96^2.
COCO_SMALL_AREA = 32.0**2
COCO_MEDIUM_AREA = 96.0**2
_UNBOUNDED_AREA = 1e10


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class TransformOp:
    """One entry in an augmentation pipeline: a registered name plus kwargs."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentationConfig:
    """Ordered transform pipelines, built by ``smalldet.data.transforms``.

    The eval pipeline must stay geometry-preserving — anything that moves or
    rescales pixels there invalidates the comparison against ground truth.
    """

    train: List[TransformOp] = field(default_factory=list)
    eval: List[TransformOp] = field(default_factory=list)


@dataclass(frozen=True)
class SplitConfig:
    """One dataset split: an image directory and a COCO annotation file."""

    images: str = ""
    annotations: str = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.images and self.annotations)


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 2
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False


@dataclass(frozen=True)
class DataConfig:
    #: Prefix joined onto every relative ``images``/``annotations`` path below.
    root: str = "."
    train: SplitConfig = field(default_factory=SplitConfig)
    val: SplitConfig = field(default_factory=SplitConfig)
    test: SplitConfig = field(default_factory=SplitConfig)

    #: Optional override for the class names in the annotation file. Index 0 is
    #: implicitly background and must not appear here.
    class_names: List[str] = field(default_factory=list)

    #: Images with no annotations. Keeping them teaches the model what an empty
    #: frame looks like, which matters when small objects are sparse.
    keep_empty_images: bool = True

    #: Boxes thinner than this (in pixels, before augmentation) are dropped as
    #: degenerate. Keep it at 1.0 for small-object data — a higher threshold
    #: silently deletes exactly the objects being optimised for.
    min_box_size: float = 1.0

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    train_loader: LoaderConfig = field(
        default_factory=lambda: LoaderConfig(batch_size=2, shuffle=True)
    )
    eval_loader: LoaderConfig = field(
        default_factory=lambda: LoaderConfig(batch_size=1, shuffle=False)
    )


# -------------------------------------------------------------------------- model


@dataclass(frozen=True)
class AnchorConfig:
    """Anchor pyramid override.

    torchvision's default RPN anchors start at 32px on the finest FPN level,
    which is already larger than many of the objects this project targets — an
    object smaller than the smallest anchor can never reach a high enough IoU
    to be sampled as a positive, so it is invisible to the RPN. Dropping the
    base sizes is the single highest-leverage change for ``AP_small``.
    """

    enabled: bool = False
    #: One base size per feature-pyramid level, finest level first.
    base_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    #: Anchors per octave. 1 for Faster R-CNN's RPN, 3 for RetinaNet/FCOS heads.
    scales_per_octave: int = 1
    aspect_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "fasterrcnn_resnet50_fpn_v2"

    #: Pretrained detector weights ("DEFAULT", a specific enum name, or null).
    #: With weights set, the model is built at its original class count and the
    #: prediction head is then replaced — the only order that works.
    weights: Optional[str] = "DEFAULT"
    #: Backbone-only weights, used when ``weights`` is null.
    weights_backbone: Optional[str] = None

    #: Including background. Left null, it is inferred from the dataset.
    num_classes: Optional[int] = None
    trainable_backbone_layers: Optional[int] = None

    #: The detector's internal resize. Raising ``min_size`` above the 800px
    #: default gives small objects more pixels to be detected from, and is the
    #: second-biggest lever on ``AP_small`` after anchor sizes.
    min_size: int = 800
    max_size: int = 1333

    anchors: AnchorConfig = field(default_factory=AnchorConfig)

    #: Passed straight to the torchvision constructor. Family-specific, e.g.
    #: ``box_detections_per_img``, ``rpn_pre_nms_top_n_train``,
    #: ``box_score_thresh`` for the R-CNN family.
    kwargs: Dict[str, Any] = field(default_factory=dict)

    #: Path to a checkpoint written by the trainer, loaded after construction.
    checkpoint: Optional[str] = None


# ----------------------------------------------------------------------- training


@dataclass(frozen=True)
class FinetuneConfig:
    """Which parameters train. See ``smalldet.engine.strategies``."""

    #: One of: full, partial, head_only, gradual.
    strategy: str = "partial"

    #: For ``partial``: how many backbone stages stay trainable, counting from
    #: the output end (0 = frozen backbone, 5 = fully trainable for ResNet).
    trainable_backbone_layers: int = 3

    #: Backbone LR as a fraction of the head LR. Pretrained features need
    #: smaller steps than a randomly-initialised head.
    backbone_lr_mult: float = 0.1

    #: Regex escape hatches applied to parameter names after the strategy runs.
    #: ``unfreeze_patterns`` is applied last and wins.
    freeze_patterns: List[str] = field(default_factory=list)
    unfreeze_patterns: List[str] = field(default_factory=list)

    #: For ``gradual``: {epoch: trainable_backbone_layers}, applied at the start
    #: of each listed epoch. Keys are strings because YAML mappings are strings.
    gradual_schedule: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.005
    weight_decay: float = 0.0005
    kwargs: Dict[str, Any] = field(default_factory=lambda: {"momentum": 0.9})


@dataclass(frozen=True)
class WarmupConfig:
    """Linear LR warmup, stepped per iteration.

    A freshly-initialised prediction head produces large gradients in the first
    few hundred steps; without warmup those propagate into the pretrained
    backbone and the loss diverges.
    """

    enabled: bool = True
    iters: int = 500
    start_factor: float = 0.001
    #: Warmup runs during the first N epochs (normally 1).
    epochs: int = 1


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "multistep"
    kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"milestones": [16, 22], "gamma": 0.1}
    )
    warmup: WarmupConfig = field(default_factory=WarmupConfig)


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str = "outputs/checkpoints"
    #: Any key from the evaluation summary. ``AP_small`` rather than ``AP``
    #: because a checkpoint that wins on overall AP by improving large objects
    #: is the wrong checkpoint for this project.
    monitor: str = "AP_small"
    mode: str = "max"
    save_best: bool = True
    save_last: bool = True


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    #: "auto" resolves to cuda when available, else cpu. Never probed at import.
    device: str = "auto"
    seed: int = 0
    amp: bool = False
    grad_clip: Optional[float] = None
    accumulate_steps: int = 1
    log_interval: int = 20
    eval_interval: int = 1
    output_dir: str = "outputs"

    #: Caps for smoke tests and CI; null means "the whole split".
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    #: Registered callback names, e.g. ["console", "csv", "tensorboard"].
    callbacks: List[str] = field(default_factory=lambda: ["console", "csv"])


# --------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvalConfig:
    iou_type: str = "bbox"

    #: Null means the COCO default, 0.50:0.05:0.95.
    iou_thresholds: Optional[List[float]] = None
    max_dets: List[int] = field(default_factory=lambda: [1, 10, 100])

    #: Area buckets in pixels^2. The "small"/"medium"/"large" labels are what
    #: produce AP_small and AP_medium; "all" must always be present.
    area_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "all": [0.0, _UNBOUNDED_AREA],
            "small": [0.0, COCO_SMALL_AREA],
            "medium": [COCO_SMALL_AREA, COCO_MEDIUM_AREA],
            "large": [COCO_MEDIUM_AREA, _UNBOUNDED_AREA],
        }
    )

    #: Replace the buckets above with percentile cuts of this dataset's own box
    #: areas. COCO's 32^2/96^2 thresholds are calibrated to COCO's resolution;
    #: on a 4000px industrial frame every object can land in "small", which
    #: makes AP_small and AP identical and AP_medium undefined.
    auto_area_ranges: bool = False
    auto_area_percentiles: List[float] = field(default_factory=lambda: [33.3, 66.6])

    per_class: bool = True
    #: Key used to rank checkpoints and to headline reports.
    primary_metric: str = "AP_small"


# ---------------------------------------------------------------------- inference


@dataclass(frozen=True)
class PostprocessConfig:
    score_threshold: float = 0.5
    #: Extra NMS pass on top of the model's own. Null skips it.
    nms_iou_threshold: Optional[float] = None
    class_agnostic_nms: bool = False
    max_detections: Optional[int] = None
    #: Drop boxes below this area in pixels^2. Leave at 0 for small objects.
    min_box_area: float = 0.0
    #: Restrict output to these class indices; empty means all.
    allowed_labels: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class TilingConfig:
    """Sliced inference for small objects in high-resolution frames.

    A 4000x3000 frame fed whole to a detector with ``min_size=800`` shrinks a
    20px object to 5px. Running the detector over overlapping crops at native
    resolution and merging with NMS recovers those detections at the cost of
    one forward pass per tile.
    """

    enabled: bool = False
    tile_size: List[int] = field(default_factory=lambda: [512, 512])
    #: Fraction of the tile shared with its neighbour, so objects straddling a
    #: tile edge are still wholly inside some tile.
    overlap: float = 0.2
    #: Also run the full frame, to catch objects larger than one tile.
    include_full_image: bool = True
    merge_nms_iou: float = 0.5


@dataclass(frozen=True)
class PredictConfig:
    device: str = "auto"
    batch_size: int = 1
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    save_json: bool = True
    save_images: bool = True
    output_dir: str = "outputs/predictions"


# ------------------------------------------------------------------ visualization


@dataclass(frozen=True)
class VisualizationConfig:
    box_width: int = 2
    font: Optional[str] = None
    font_size: int = 14

    #: Matplotlib qualitative colormap name used to colour classes.
    palette: str = "tab20"
    #: Explicit per-class colours; keys are class names, values any PIL colour.
    class_colors: Dict[str, str] = field(default_factory=dict)

    show_labels: bool = True
    show_scores: bool = True
    score_format: str = "{:.2f}"
    fill_boxes: bool = False

    draw_masks: bool = True
    mask_alpha: float = 0.6
    mask_threshold: float = 0.5

    ground_truth_color: str = "#00FF00"
    #: Null means "colour predictions per class from the palette".
    prediction_color: Optional[str] = None
    side_by_side: bool = True
    max_images: int = 8
    dpi: int = 120

    #: Outline objects below ``small_area_threshold`` px^2 in a distinct colour.
    #: On a dense frame this is the fastest way to see whether a run regressed
    #: specifically on the objects AP_small measures.
    highlight_small_objects: bool = True
    small_area_threshold: float = COCO_SMALL_AREA
    small_object_color: str = "#FF00FF"


# ---------------------------------------------------------------------------- app


@dataclass(frozen=True)
class AppConfig:
    title: str = "smalldet — small-object detection"
    description: str = ""
    theme: str = "default"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False

    #: Image paths preloaded as clickable examples in the UI.
    examples: List[str] = field(default_factory=list)

    #: Let the user move the score threshold / tiling switches at runtime. With
    #: this off the UI is pinned to the config values, which is what you want
    #: for a demo that must show reproducible numbers.
    allow_threshold_control: bool = True
    show_tiling_control: bool = True
    show_json_output: bool = True
    concurrency_limit: int = 2


# --------------------------------------------------------------------------- root


@dataclass(frozen=True)
class Config:
    """The root document. Every entrypoint takes one of these."""

    name: str = "smalldet"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    visualize: VisualizationConfig = field(default_factory=VisualizationConfig)
    app: AppConfig = field(default_factory=AppConfig)
