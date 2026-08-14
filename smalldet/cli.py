"""Command line entrypoints.

    python -m smalldet.cli train    configs/finetune_smallobj.yaml
    python -m smalldet.cli evaluate configs/finetune_smallobj.yaml --checkpoint outputs/checkpoints/best.pt
    python -m smalldet.cli predict  configs/predict.yaml --images path/to/*.jpg
    python -m smalldet.cli app      configs/app.yaml
    python -m smalldet.cli stats    configs/finetune_smallobj.yaml
    python -m smalldet.cli synth    --root Dataset/synthetic

Every subcommand takes ``--set dotted.key=value`` overrides, so a one-off
experiment does not need a new file:

    python -m smalldet.cli train configs/finetune_smallobj.yaml \\
        --set train.epochs=3 --set model.anchors.base_sizes=[4,8,16,32,64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .config import ConfigError, dump_config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smalldet",
        description="Config-driven small-object detection: finetune, evaluate, "
        "predict, and serve.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser, needs_config: bool = True) -> None:
        if needs_config:
            sub.add_argument("config", help="path to a YAML config")
        sub.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a config value, e.g. --set train.epochs=3",
        )

    train = subparsers.add_parser("train", help="finetune a detector")
    add_common(train)

    evaluate = subparsers.add_parser("evaluate", help="score a checkpoint")
    add_common(evaluate)
    evaluate.add_argument("--split", default="val", choices=["train", "val", "test"])
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument(
        "--report-dir", default=None, help="write metrics JSON here"
    )

    predict = subparsers.add_parser("predict", help="run inference on images")
    add_common(predict)
    predict.add_argument("--images", nargs="+", required=True)
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--output-dir", default=None)

    app = subparsers.add_parser("app", help="serve the Gradio UI")
    add_common(app)
    app.add_argument("--checkpoint", default=None)
    app.add_argument("--port", type=int, default=None)
    app.add_argument("--share", action="store_true")

    stats = subparsers.add_parser(
        "stats", help="report ground-truth object sizes and area-bucket occupancy"
    )
    add_common(stats)
    stats.add_argument("--split", default="train", choices=["train", "val", "test"])
    stats.add_argument("--plot", default=None, help="write a histogram here")

    synth = subparsers.add_parser(
        "synth", help="generate a synthetic small-object dataset"
    )
    add_common(synth, needs_config=False)
    synth.add_argument("--root", required=True)
    synth.add_argument("--num-images", type=int, default=24)
    synth.add_argument("--seed", type=int, default=0)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "synth":
        return _synth(args)

    try:
        config = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handlers = {
        "train": _train,
        "evaluate": _evaluate,
        "predict": _predict,
        "app": _app,
        "stats": _stats,
    }
    return handlers[args.command](config, args)


# --------------------------------------------------------------------- commands


def _train(config, args) -> int:
    from .pipeline import run_training

    state = run_training(config)
    return 0 if state.epoch >= 0 else 1


def _evaluate(config, args) -> int:
    from .pipeline import run_evaluation

    run_evaluation(
        config,
        split=args.split,
        checkpoint=args.checkpoint,
        report_dir=args.report_dir,
    )
    return 0


def _predict(config, args) -> int:
    from .pipeline import build_predictor, build_renderer

    predictor = build_predictor(config, checkpoint=args.checkpoint)
    renderer = build_renderer(config, predictor.class_names)

    output_dir = Path(args.output_dir or config.predict.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in _expand(args.images):
        result = predictor.predict(path)
        print(
            f"{path.name}: {len(result)} detection(s) in {result.elapsed_ms:.0f} ms "
            f"{result.size_histogram()}"
        )
        if config.predict.save_images:
            renderer.save(
                renderer.draw_result(result), output_dir / f"{path.stem}_pred.png"
            )
        records.append({"image": str(path), "detections": result.to_records()})

    if config.predict.save_json:
        (output_dir / "predictions.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
    print(f"wrote {output_dir}")
    return 0


def _app(config, args) -> int:
    from .app import launch_app

    launch_app(
        config,
        checkpoint=args.checkpoint,
        share=True if args.share else None,
        server_port=args.port,
    )
    return 0


def _stats(config, args) -> int:
    from .data import build_dataset, summarize_areas
    from .data.stats import area_ranges_from_percentiles

    dataset = build_dataset(config.data, args.split)
    areas = dataset.box_areas()
    if not areas:
        print(f"error: split {args.split!r} has no annotations", file=sys.stderr)
        return 1

    stats = summarize_areas(areas, config.eval.area_ranges)
    print(f"split: {args.split}  images: {len(dataset)}")
    print(stats.describe())
    print(
        "\nsuggested eval.area_ranges from this split's own percentiles:\n  "
        + json.dumps(
            area_ranges_from_percentiles(areas, config.eval.auto_area_percentiles)
        )
    )

    if args.plot:
        from .visualization import plot_area_histogram

        path = plot_area_histogram(
            areas, config.eval.area_ranges, args.plot, config=config.visualize
        )
        print(f"wrote {path}")
    return 0


def _synth(args) -> int:
    from .data.synthetic import generate_dataset

    written = generate_dataset(
        args.root, num_images=args.num_images, seed=args.seed
    )
    for split, path in written.items():
        print(f"{split}: {path}")
    return 0


def _expand(patterns: Sequence[str]) -> List[Path]:
    """Expand paths, directories, and globs into a flat list of image files."""
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            paths.extend(
                child for child in sorted(path.iterdir()) if child.suffix.lower() in suffixes
            )
        elif path.is_file():
            paths.append(path)
        else:
            parent = path.parent if str(path.parent) else Path(".")
            paths.extend(sorted(parent.glob(path.name)))
    if not paths:
        raise SystemExit(f"no images matched {list(patterns)}")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
