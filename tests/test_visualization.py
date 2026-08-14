"""Rendering and diagnostic plots."""

from __future__ import annotations

import torch

from _support import scratch_dir, synthetic_dataset

from smalldet.config import VisualizationConfig
from smalldet.visualization import Renderer, build_palette

CLASS_NAMES = ["__background__", "screw", "washer", "nut"]


def _image(size: int = 96) -> torch.Tensor:
    return torch.full((3, size, size), 90, dtype=torch.uint8)


def _boxes():
    return torch.tensor(
        [
            [5.0, 5.0, 20.0, 20.0],  # 225 px^2  -> small
            [30.0, 30.0, 80.0, 80.0],  # 2500 px^2 -> medium
        ]
    )


# ------------------------------------------------------------------- palette


def test_palette_is_stable_per_class_index():
    """The same class must be the same colour in every image and every run, or
    side-by-side comparison is useless."""
    first = build_palette(4, "tab20", CLASS_NAMES)
    second = build_palette(4, "tab20", CLASS_NAMES)
    assert first == second
    assert len(set(first)) == 4  # and distinguishable


def test_class_colour_overrides_apply_by_name():
    palette = build_palette(4, "tab20", CLASS_NAMES, {"washer": "#123456"})
    assert palette[CLASS_NAMES.index("washer")] == "#123456"


def test_palette_falls_back_without_an_unknown_colormap():
    from _support import expect_error

    expect_error(
        lambda: build_palette(3, "not_a_colormap"), ValueError, contains="colormap"
    )


# ------------------------------------------------------------------ rendering


def test_draw_returns_a_uint8_image_of_the_same_shape():
    renderer = Renderer(VisualizationConfig(), CLASS_NAMES)
    image = _image()
    output = renderer.draw(image, _boxes(), torch.tensor([1, 2]), torch.tensor([0.9, 0.7]))
    assert output.shape == image.shape
    assert output.dtype == torch.uint8
    assert not torch.equal(output, image)  # something was actually drawn


def test_float_images_are_converted_before_drawing():
    """draw_bounding_boxes rejects float input; the renderer must not pass the
    model's normalized tensor straight through."""
    renderer = Renderer(VisualizationConfig(), CLASS_NAMES)
    output = renderer.draw(
        torch.rand(3, 64, 64), torch.tensor([[5.0, 5.0, 30.0, 30.0]]), torch.tensor([1])
    )
    assert output.dtype == torch.uint8


def test_empty_predictions_render_the_untouched_image():
    renderer = Renderer(VisualizationConfig(), CLASS_NAMES)
    image = _image()
    output = renderer.draw(image, torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64))
    assert torch.equal(output, image)


def test_small_objects_are_highlighted_distinctly():
    """The fastest way to see whether a regression is concentrated in the
    bucket AP_small measures."""
    config = VisualizationConfig(
        highlight_small_objects=True, small_area_threshold=1024.0
    )
    renderer = Renderer(config, CLASS_NAMES)
    colors = renderer._box_colors(_boxes(), torch.tensor([1, 2]), None)
    assert colors[0] == config.small_object_color  # 225 px^2
    assert colors[1] != config.small_object_color  # 2500 px^2


def test_highlight_can_be_turned_off():
    config = VisualizationConfig(highlight_small_objects=False)
    renderer = Renderer(config, CLASS_NAMES)
    colors = renderer._box_colors(_boxes(), torch.tensor([1, 2]), None)
    assert config.small_object_color not in colors


def test_labels_follow_the_config():
    renderer = Renderer(VisualizationConfig(show_scores=True), CLASS_NAMES)
    labels = renderer._box_labels(torch.tensor([1, 3]), torch.tensor([0.912, 0.5]))
    assert labels == ["screw 0.91", "nut 0.50"]

    without = Renderer(VisualizationConfig(show_scores=False), CLASS_NAMES)
    assert without._box_labels(torch.tensor([1]), torch.tensor([0.9])) == ["screw"]

    hidden = Renderer(VisualizationConfig(show_labels=False), CLASS_NAMES)
    assert hidden._box_labels(torch.tensor([1]), torch.tensor([0.9])) is None


def test_score_format_is_configurable():
    renderer = Renderer(VisualizationConfig(score_format="{:.0%}"), CLASS_NAMES)
    assert renderer._box_labels(torch.tensor([1]), torch.tensor([0.9])) == ["screw 90%"]


def test_soft_masks_are_thresholded_and_squeezed():
    """Mask R-CNN emits (N, 1, H, W) floats; draw_segmentation_masks needs
    (N, H, W) booleans or nothing appears."""
    renderer = Renderer(VisualizationConfig(mask_threshold=0.5), CLASS_NAMES)
    soft = torch.zeros(2, 1, 32, 32)
    soft[0, 0, :16, :16] = 0.9
    hard = renderer._binarize(soft)
    assert hard.shape == (2, 32, 32)
    assert hard.dtype == torch.bool
    assert int(hard[0].sum()) == 256


def test_masks_are_drawn_onto_the_canvas():
    renderer = Renderer(VisualizationConfig(draw_masks=True, mask_alpha=0.8), CLASS_NAMES)
    masks = torch.zeros(1, 96, 96, dtype=torch.bool)
    masks[0, 10:40, 10:40] = True
    output = renderer.draw(
        _image(), torch.tensor([[10.0, 10.0, 40.0, 40.0]]), torch.tensor([1]), masks=masks
    )
    assert output.dtype == torch.uint8


def test_comparison_places_ground_truth_beside_predictions():
    renderer = Renderer(VisualizationConfig(side_by_side=True), CLASS_NAMES)
    image = _image()
    target = {"boxes": _boxes(), "labels": torch.tensor([1, 2])}
    prediction = {
        "boxes": _boxes(),
        "labels": torch.tensor([1, 2]),
        "scores": torch.tensor([0.8, 0.6]),
    }
    output = renderer.compare(image, target, prediction)
    assert output.shape[2] == image.shape[2] * 2 + 4  # two panels plus a gap
    assert output.shape[1] == image.shape[1]


def test_side_by_side_can_be_disabled():
    renderer = Renderer(VisualizationConfig(side_by_side=False), CLASS_NAMES)
    image = _image()
    target = {"boxes": _boxes(), "labels": torch.tensor([1, 2])}
    output = renderer.compare(image, target, dict(target, scores=torch.tensor([0.8, 0.6])))
    assert output.shape == image.shape


def test_renderer_saves_a_readable_png():
    from PIL import Image

    renderer = Renderer(VisualizationConfig(), CLASS_NAMES)
    output = renderer.draw(_image(), _boxes(), torch.tensor([1, 2]))
    path = renderer.save(output, scratch_dir("render") / "out.png")
    assert path.is_file()
    with Image.open(path) as handle:
        assert handle.size == (96, 96)


def test_box_width_reaches_the_drawing_call():
    """A thin default matters for small objects: a 4px line on a 10px box hides
    the object it is meant to point at."""
    thin = Renderer(VisualizationConfig(box_width=1), CLASS_NAMES).draw(
        _image(), _boxes(), torch.tensor([1, 2])
    )
    thick = Renderer(VisualizationConfig(box_width=5), CLASS_NAMES).draw(
        _image(), _boxes(), torch.tensor([1, 2])
    )
    changed_thin = int((thin != _image()).sum())
    changed_thick = int((thick != _image()).sum())
    assert changed_thick > changed_thin


# ---------------------------------------------------------------------- plots


def test_area_histogram_marks_the_evaluation_cut_offs():
    from smalldet.data.coco import CocoDetectionDataset
    from smalldet.visualization import plot_area_histogram

    paths = synthetic_dataset()
    dataset = CocoDetectionDataset(paths["images"], paths["train"])
    path = plot_area_histogram(
        dataset.box_areas(),
        {"all": [0.0, 1e10], "small": [0.0, 1024.0], "medium": [1024.0, 9216.0]},
        scratch_dir("plots") / "areas.png",
    )
    assert path.is_file() and path.stat().st_size > 0


def test_pr_curves_plot_per_area_bucket():
    from smalldet.evaluation import GroundTruth, evaluate_detections
    from smalldet.visualization import plot_pr_curves

    annotations = [
        {"image_id": 1, "label": 1, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 0},
        {"image_id": 1, "label": 1, "bbox": [50, 50, 110, 110], "area": 3600, "iscrowd": 0},
    ]
    gt = GroundTruth(image_ids=[1], annotations=annotations, class_names=["__background__", "part"])
    detections = [{**a, "score": 0.9} for a in annotations]
    result = evaluate_detections(gt, detections)

    path = plot_pr_curves(result, scratch_dir("plots") / "pr.png")
    assert path.is_file() and path.stat().st_size > 0


def test_history_plot_skips_the_undefined_sentinel():
    """-1 must never be drawn as a data point; it would pull the axis below
    zero and imply a measurement that was never made."""
    from smalldet.visualization import plot_history

    history = [
        {"epoch": 1, "AP": 0.10, "AP_small": 0.05, "AP_medium": -1.0},
        {"epoch": 2, "AP": 0.22, "AP_small": 0.14, "AP_medium": -1.0},
    ]
    figure = plot_history(history)
    axis = figure.axes[0]
    for line in axis.lines:
        assert min(line.get_ydata()) >= 0.0
    assert axis.get_ylim()[0] >= -0.05


def test_per_class_plot_renders():
    from smalldet.evaluation.coco_eval import EvalResult
    from smalldet.visualization import plot_per_class

    result = EvalResult(
        metrics={"AP": 0.4},
        per_class={"screw": {"AP": 0.5}, "washer": {"AP": 0.3}, "nut": {"AP": -1.0}},
    )
    path = plot_per_class(result, scratch_dir("plots") / "per_class.png")
    assert path.is_file()
