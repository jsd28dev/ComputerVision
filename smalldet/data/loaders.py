"""DataLoader construction for detection batches.

Detection breaks the default ``collate_fn`` twice over: images in a batch can
have different spatial sizes, and every image has a different number of
objects, so neither the images nor the targets can be stacked. The fix is the
one torchvision's own reference scripts use — keep the batch as parallel tuples
and let the model's internal transform handle padding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import LoaderConfig

Batch = Tuple[Tuple[Any, ...], Tuple[Dict[str, Any], ...]]


def collate_fn(batch: Sequence[Tuple[Any, Dict[str, Any]]]) -> Batch:
    """Turn a list of ``(image, target)`` into ``(images, targets)`` tuples.

    This is exactly the shape every torchvision detection model's ``forward``
    expects: ``model(images, targets)``.
    """
    return tuple(zip(*batch))  # type: ignore[return-value]


def build_dataloader(
    dataset: Dataset,
    config: LoaderConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a DataLoader from a :class:`LoaderConfig`."""
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 (got {config.batch_size})")

    # persistent_workers is only meaningful with worker processes, and asking
    # for it with num_workers=0 is a hard error in torch rather than a no-op.
    persistent_workers = config.persistent_workers and config.num_workers > 0

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=config.drop_last,
        collate_fn=collate_fn,
        generator=generator,
    )


def move_to_device(
    images: Sequence[Any],
    targets: Sequence[Dict[str, Any]],
    device: torch.device | str,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Move a collated batch to ``device``.

    ``images`` is a tuple, not a tensor, so each element moves individually;
    calling ``.to(device)`` on the batch itself is the classic failure here.
    """
    device = torch.device(device)
    moved_images = [image.to(device) for image in images]
    moved_targets = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    return moved_images, moved_targets


def detach_to_cpu(outputs: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Bring model outputs back to CPU so they can be accumulated safely."""
    return [
        {
            key: value.detach().to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in output.items()
        }
        for output in outputs
    ]
