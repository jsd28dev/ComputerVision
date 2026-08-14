"""Generate a synthetic small-object dataset in COCO format.

Real industrial datasets are large, slow, and often not redistributable, which
makes them a poor foundation for a test suite. This generator produces the
property that actually matters here: most objects are a handful of pixels
across, with a deliberate minority large enough to populate the *medium*
bucket. Without that minority, ``AP_medium`` would report COCO's -1 sentinel
and half the metrics this project cares about would be untestable.

Objects are drawn as three visually distinct shapes so a detector can, in
principle, learn to separate the classes — the smoke tests only need the
pipeline to run, but a dataset that is impossible to fit makes it hard to tell
a broken training loop from an unlucky one.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

#: Contiguous label order is 1..3; index 0 is background.
CLASS_NAMES: Tuple[str, ...] = ("screw", "washer", "nut")

#: Side length ranges in pixels. "small" sits below COCO's 32px cut, "medium"
#: straddles it, so both AP_small and AP_medium have ground truth to score.
SMALL_SIZE_RANGE = (6, 16)
MEDIUM_SIZE_RANGE = (34, 70)


def generate_dataset(
    root: str | Path,
    *,
    num_images: int = 12,
    image_size: Tuple[int, int] = (256, 256),
    objects_per_image: Tuple[int, int] = (3, 8),
    medium_fraction: float = 0.25,
    splits: Sequence[str] = ("train", "val"),
    seed: int = 0,
) -> Dict[str, Path]:
    """Write images and one COCO annotation file per split.

    Returns a mapping of split name to its annotation file path. Images for all
    splits share one ``images/`` directory, with the split name in the filename.
    """
    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    image_id = 1
    annotation_id = 1

    for split_index, split in enumerate(splits):
        # Offsetting the seed per split keeps train and val genuinely different
        # while remaining fully reproducible.
        rng = random.Random(seed + 1009 * split_index)
        records: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        for index in range(num_images):
            width, height = image_size
            image = _background(rng, width, height)
            draw = ImageDraw.Draw(image)

            count = rng.randint(*objects_per_image)
            placed: List[Tuple[float, float, float, float]] = []
            for _ in range(count):
                is_medium = rng.random() < medium_fraction
                size_range = MEDIUM_SIZE_RANGE if is_medium else SMALL_SIZE_RANGE
                size = rng.randint(*size_range)
                if size + 2 >= min(width, height):
                    continue
                box = _find_free_box(rng, width, height, size, placed)
                if box is None:
                    continue
                placed.append(box)

                label = rng.randint(1, len(CLASS_NAMES))
                _draw_shape(draw, box, label, rng)

                x1, y1, x2, y2 = box
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            file_name = f"{split}_{index:04d}.png"
            image.save(images_dir / file_name)
            records.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            image_id += 1

        document = {
            "info": {
                "description": "smalldet synthetic small-object dataset",
                "split": split,
            },
            "images": records,
            "annotations": annotations,
            "categories": [
                {"id": index, "name": name, "supercategory": "part"}
                for index, name in enumerate(CLASS_NAMES, start=1)
            ],
        }
        path = root / f"annotations_{split}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        written[split] = path

    return written


# ---------------------------------------------------------------------- detail


def _background(rng: random.Random, width: int, height: int) -> Image.Image:
    """A mildly textured backdrop.

    A flat background would let a detector separate objects on brightness
    alone, which makes a smoke test pass for the wrong reason.
    """
    base = rng.randint(60, 110)
    image = Image.new("RGB", (width, height), (base, base, base + rng.randint(0, 12)))
    draw = ImageDraw.Draw(image)
    for _ in range(width * height // 220):
        x, y = rng.randrange(width), rng.randrange(height)
        shade = base + rng.randint(-25, 25)
        draw.point((x, y), fill=(shade, shade, shade))
    return image


def _find_free_box(
    rng: random.Random,
    width: int,
    height: int,
    size: int,
    placed: Sequence[Tuple[float, float, float, float]],
    attempts: int = 30,
) -> Tuple[float, float, float, float] | None:
    """Place a non-overlapping box, or give up.

    Overlapping ground truth would make the greedy matcher in the evaluator
    ambiguous, and an ambiguous expectation is useless in a test.
    """
    for _ in range(attempts):
        x1 = rng.uniform(1, max(1.0, width - size - 1))
        y1 = rng.uniform(1, max(1.0, height - size - 1))
        candidate = (x1, y1, x1 + size, y1 + size)
        if all(_gap(candidate, other) for other in placed):
            return candidate
    return None


def _gap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    margin: float = 3.0,
) -> bool:
    return (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    label: int,
    rng: random.Random,
) -> None:
    x1, y1, x2, y2 = box
    brightness = rng.randint(170, 245)
    fill = (brightness, brightness, brightness)
    outline = (35, 35, 35)

    if label == 1:  # screw: elongated capsule
        draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline)
        draw.line([(x1 + (x2 - x1) / 2, y1), (x1 + (x2 - x1) / 2, y2)], fill=outline)
    elif label == 2:  # washer: ring
        draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline)
        inset = max(1.0, (x2 - x1) / 4)
        draw.ellipse(
            [x1 + inset, y1 + inset, x2 - inset, y2 - inset], fill=(50, 50, 55)
        )
    else:  # nut: hexagon
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        radius = (x2 - x1) / 2
        points = [
            (
                cx + radius * math.cos(math.pi / 3 * i),
                cy + radius * math.sin(math.pi / 3 * i),
            )
            for i in range(6)
        ]
        draw.polygon(points, fill=fill, outline=outline)
