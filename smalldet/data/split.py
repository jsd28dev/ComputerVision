"""Split one COCO annotation file into train / validation / test files.

Splitting happens at the *image* level, never the annotation level. Putting two
annotations from the same image on opposite sides of the split leaks the exact
pixels being tested on into training, and the resulting metrics are fiction.

The categories array is copied verbatim into every output file. That is what
lets ``build_assembly`` assert all splits agree on the label mapping — a check
that would be meaningless if each split derived its own category list from the
annotations it happened to receive.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

DEFAULT_RATIOS: Tuple[float, float, float] = (0.7, 0.15, 0.15)
SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")


def split_coco(
    annotation_file: str | Path,
    output_dir: str | Path | None = None,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    seed: int = 0,
    prefix: str = "annotations",
    stratify_by_class: bool = True,
) -> Dict[str, Path]:
    """Write ``{prefix}_train.json`` / ``_val.json`` / ``_test.json``.

    Returns a mapping of split name to the file written. A ratio of 0 skips
    that split entirely rather than writing an empty file, so a two-way
    train/val split is expressed as ``(0.8, 0.2, 0.0)``.
    """
    source = Path(annotation_file)
    if not source.is_file():
        raise FileNotFoundError(f"annotation file not found: {source}")

    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3:
        raise ValueError(
            f"ratios must be (train, val, test); got {len(ratios)} value(s)"
        )
    if any(value < 0 for value in ratios):
        raise ValueError(f"ratios must not be negative: {ratios}")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("at least one ratio must be greater than zero")
    ratios = tuple(value / total for value in ratios)

    document = json.loads(source.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if key not in document:
            raise ValueError(f"{source.name} is missing the {key!r} array")

    images: List[dict] = list(document["images"])
    if not images:
        raise ValueError(f"{source.name} contains no images to split")

    by_image: Dict[Any, List[dict]] = {}
    for annotation in document["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    ordered = (
        _stratified_order(images, by_image, seed)
        if stratify_by_class
        else _shuffled(images, seed)
    )
    assignments = _assign(ordered, ratios)

    directory = Path(output_dir) if output_dir else source.parent
    directory.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for name, records in assignments.items():
        if not records:
            continue
        annotations: List[dict] = []
        for record in records:
            annotations.extend(by_image.get(record["id"], []))
        payload = {
            **{
                key: value
                for key, value in document.items()
                if key not in {"images", "annotations"}
            },
            "images": records,
            "annotations": annotations,
        }
        path = directory / f"{prefix}_{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        written[name] = path

    return written


def summarize_split(written: Dict[str, Path]) -> str:
    """A short human-readable report of what a split produced."""
    lines = []
    for name, path in written.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        classes = {a["category_id"] for a in document["annotations"]}
        lines.append(
            f"{name}: {len(document['images'])} images, "
            f"{len(document['annotations'])} annotations, "
            f"{len(classes)} class(es) present -> {path.name}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------- detail


def _shuffled(images: List[dict], seed: int) -> List[dict]:
    ordered = sorted(images, key=lambda record: record["id"])
    random.Random(seed).shuffle(ordered)
    return ordered


def _stratified_order(
    images: List[dict], by_image: Dict[Any, List[dict]], seed: int
) -> List[dict]:
    """Interleave images grouped by their dominant class.

    Detection data is often ordered by class on disk, and a plain shuffle on a
    small dataset can still land every instance of a rare class in one split —
    which shows up later as a per-class AP of -1 rather than as an obvious
    error. Interleaving by dominant class spreads each class across the splits
    roughly in proportion.
    """
    groups: Dict[Any, List[dict]] = {}
    for record in images:
        annotations = by_image.get(record["id"], [])
        if annotations:
            counts: Dict[Any, int] = {}
            for annotation in annotations:
                counts[annotation["category_id"]] = (
                    counts.get(annotation["category_id"], 0) + 1
                )
            dominant = max(sorted(counts), key=lambda key: counts[key])
        else:
            dominant = None  # empty images form their own group
        groups.setdefault(dominant, []).append(record)

    rng = random.Random(seed)
    for records in groups.values():
        records.sort(key=lambda record: record["id"])
        rng.shuffle(records)

    ordered: List[dict] = []
    keys = sorted(groups, key=lambda key: (key is None, key))
    index = 0
    while any(groups[key] for key in keys):
        key = keys[index % len(keys)]
        if groups[key]:
            ordered.append(groups[key].pop())
        index += 1
    return ordered


def _assign(
    ordered: List[dict], ratios: Sequence[float]
) -> Dict[str, List[dict]]:
    """Cut the ordered list into three, guaranteeing non-empty asked-for splits."""
    count = len(ordered)
    sizes = [int(count * ratio) for ratio in ratios]

    # Integer truncation loses up to two images; give them to the largest split.
    remainder = count - sum(sizes)
    for offset in range(remainder):
        sizes[offset % len(sizes)] += 1

    # A requested split that rounded down to zero on a tiny dataset would be
    # silently dropped, so borrow one image from the largest split instead.
    for index, ratio in enumerate(ratios):
        if ratio > 0 and sizes[index] == 0:
            donor = max(range(len(sizes)), key=lambda i: sizes[i])
            if sizes[donor] > 1:
                sizes[donor] -= 1
                sizes[index] += 1

    assignments: Dict[str, List[dict]] = {}
    start = 0
    for name, size in zip(SPLIT_NAMES, sizes):
        assignments[name] = ordered[start : start + size]
        start += size
    return assignments
