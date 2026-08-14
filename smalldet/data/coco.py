"""COCO-format detection dataset.

The annotation file is parsed directly rather than through ``pycocotools``, for
two reasons: it keeps the training path free of a compiled dependency, and it
lets the dataset own the category-id remapping that torchvision's detection
heads require. COCO category ids are arbitrary sparse integers (COCO itself
skips 12, 26, 29, ...); the heads need contiguous indices with 0 reserved for
background. Doing that mapping here, once, means every downstream consumer —
trainer, evaluator, visualizer, Gradio app — agrees on what class 3 means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from ..config import DataConfig, SplitConfig


class CocoDetectionDataset(Dataset):
    """Returns ``(image, target)`` pairs in the torchvision detection contract.

    ``target`` holds ``boxes`` (``tv_tensors.BoundingBoxes``, XYXY), ``labels``
    (contiguous, 1-based), ``image_id``, ``area``, and ``iscrowd``. Wrapping the
    image and boxes as tv_tensors is what allows ``transforms.v2`` to move both
    together; a plain tensor would be resized while its boxes silently were not.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotation_file: str | Path,
        transforms: Optional[Any] = None,
        *,
        keep_empty_images: bool = True,
        min_box_size: float = 1.0,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms
        self.keep_empty_images = keep_empty_images
        self.min_box_size = float(min_box_size)

        if not self.annotation_file.is_file():
            raise FileNotFoundError(f"annotation file not found: {self.annotation_file}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {self.images_dir}")

        with self.annotation_file.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        self._validate_document(document)

        categories = sorted(document.get("categories", []), key=lambda c: c["id"])
        #: raw COCO category id -> contiguous label in [1, num_classes)
        self.category_id_to_label: Dict[int, int] = {
            category["id"]: index for index, category in enumerate(categories, start=1)
        }
        self.label_to_category_id: Dict[int, int] = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }
        names = [str(category.get("name", category["id"])) for category in categories]
        if class_names:
            if len(class_names) != len(names):
                raise ValueError(
                    f"data.class_names has {len(class_names)} entries but "
                    f"{self.annotation_file.name} declares {len(names)} categories"
                )
            names = list(class_names)
        #: Index 0 is background so ``class_names[label]`` is a direct lookup.
        self.class_names: List[str] = ["__background__", *names]

        annotations_by_image: Dict[int, List[dict]] = {}
        for annotation in document.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(
                annotation
            )

        self.images: List[dict] = []
        self._annotations: List[List[dict]] = []
        for record in document["images"]:
            annotations = annotations_by_image.get(record["id"], [])
            kept = [
                annotation
                for annotation in annotations
                if self._is_usable(annotation)
            ]
            if not kept and not self.keep_empty_images:
                continue
            self.images.append(record)
            self._annotations.append(kept)

        self.image_ids: List[int] = [record["id"] for record in self.images]

    # ------------------------------------------------------------- properties

    @property
    def num_classes(self) -> int:
        """Including background, as torchvision's heads count it."""
        return len(self.class_names)

    # ---------------------------------------------------------------- dataset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        record = self.images[index]
        annotations = self._annotations[index]

        path = self.images_dir / record["file_name"]
        with PILImage.open(path) as handle:
            # Detection backbones expect 3 channels; grayscale industrial
            # captures and RGBA PNGs both show up in practice.
            image = tv_tensors.Image(F.pil_to_tensor(handle.convert("RGB")))

        canvas_size = F.get_size(image)  # (H, W)
        boxes = torch.tensor(
            [self._to_xyxy(annotation["bbox"]) for annotation in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)
        labels = torch.tensor(
            [
                self.category_id_to_label[annotation["category_id"]]
                for annotation in annotations
            ],
            dtype=torch.int64,
        )
        iscrowd = torch.tensor(
            [int(annotation.get("iscrowd", 0)) for annotation in annotations],
            dtype=torch.int64,
        )
        # Prefer the annotated area (it is the mask area for segmentation
        # datasets, which is what COCO's small/medium buckets are defined on);
        # fall back to the box area when it is absent.
        area = torch.tensor(
            [
                float(
                    annotation.get(
                        "area", annotation["bbox"][2] * annotation["bbox"][3]
                    )
                )
                for annotation in annotations
            ],
            dtype=torch.float32,
        )

        target: Dict[str, Any] = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=canvas_size
            ),
            "labels": labels,
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    # ----------------------------------------------------------------- detail

    def coco_ground_truth(self) -> Dict[str, Any]:
        """The annotations as the evaluator wants them, with mapped labels.

        Built from the retained records only, so an evaluation never scores
        against images the dataset filtered out.
        """
        annotations: List[dict] = []
        for record, image_annotations in zip(self.images, self._annotations):
            for annotation in image_annotations:
                annotations.append(
                    {
                        "image_id": record["id"],
                        "label": self.category_id_to_label[annotation["category_id"]],
                        "bbox": list(self._to_xyxy(annotation["bbox"])),
                        "area": float(
                            annotation.get(
                                "area", annotation["bbox"][2] * annotation["bbox"][3]
                            )
                        ),
                        "iscrowd": int(annotation.get("iscrowd", 0)),
                    }
                )
        return {
            "image_ids": list(self.image_ids),
            "annotations": annotations,
            "class_names": list(self.class_names),
        }

    def box_areas(self) -> List[float]:
        """Every non-crowd ground-truth area, for area-range calibration."""
        return [
            float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
            for image_annotations in self._annotations
            for annotation in image_annotations
            if not annotation.get("iscrowd", 0)
        ]

    def _is_usable(self, annotation: Dict[str, Any]) -> bool:
        """Drop annotations that would crash or destabilise training.

        Degenerate boxes (zero or negative extent) make the box-regression loss
        produce NaNs, and an unknown category id means the annotation file
        disagrees with its own ``categories`` array.
        """
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) < 4:
            return False
        if annotation.get("category_id") not in self.category_id_to_label:
            return False
        width, height = float(bbox[2]), float(bbox[3])
        return width >= self.min_box_size and height >= self.min_box_size

    @staticmethod
    def _to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        x, y, width, height = (float(value) for value in bbox[:4])
        return x, y, x + width, y + height

    @staticmethod
    def _validate_document(document: Any) -> None:
        if not isinstance(document, dict):
            raise ValueError("COCO annotation file must contain a JSON object")
        for key in ("images", "annotations", "categories"):
            if key not in document:
                raise ValueError(
                    f"COCO annotation file is missing the {key!r} array"
                )
        for record in document["images"]:
            missing = {"id", "file_name"} - set(record)
            if missing:
                raise ValueError(
                    f"image record {record!r} is missing {sorted(missing)}"
                )


def build_dataset(
    data_config: DataConfig,
    split: str,
    transforms: Optional[Any] = None,
) -> CocoDetectionDataset:
    """Construct the dataset for ``split`` ("train", "val", or "test")."""
    split_config: SplitConfig = getattr(data_config, split, None)
    if split_config is None:
        raise ValueError(
            f"unknown split {split!r}; expected one of train, val, test"
        )
    if not split_config.is_configured:
        raise ValueError(
            f"data.{split} needs both `images` and `annotations` set before it "
            "can be loaded"
        )

    root = Path(data_config.root)
    return CocoDetectionDataset(
        images_dir=root / split_config.images,
        annotation_file=root / split_config.annotations,
        transforms=transforms,
        keep_empty_images=data_config.keep_empty_images,
        min_box_size=data_config.min_box_size,
        class_names=data_config.class_names or None,
    )
