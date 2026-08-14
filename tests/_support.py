"""Shared test helpers.

The test modules deliberately use plain functions and bare ``assert`` with no
``import pytest`` at module scope. That keeps them runnable two ways: under
pytest as normal, and under ``tests/run_tests.py`` when pytest is not installed.
The only pytest-specific behaviour needed is skipping, which ``skip()`` below
routes to whichever mechanism is available.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The 60-image VOC-in-COCO set that ships with the sibling project. Optional:
#: tests that need it skip when it is absent.
TOY_DATASET = (
    PROJECT_ROOT.parent / "FactoryPartsVision" / "Dataset" / "Toy Dataset"
)


# ------------------------------------------------------------------- skipping


try:  # pragma: no cover - depends on the environment
    import pytest

    _SKIP_EXCEPTION: type = pytest.skip.Exception

    def skip(reason: str) -> None:
        """Skip the current test."""
        pytest.skip(reason)

except ImportError:  # pragma: no cover - depends on the environment

    class Skipped(Exception):
        """Raised in place of pytest.skip when pytest is unavailable."""

    _SKIP_EXCEPTION = Skipped

    def skip(reason: str) -> None:
        raise Skipped(reason)


SKIP_EXCEPTION = _SKIP_EXCEPTION


def has_module(name: str) -> bool:
    """Whether an optional dependency is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require_module(name: str, hint: str = "") -> None:
    if not has_module(name):
        skip(f"{name} is not installed{'; ' + hint if hint else ''}")


# ------------------------------------------------------------------- fixtures


@functools.lru_cache(maxsize=None)
def synthetic_dataset(num_images: int = 6, seed: int = 7) -> Dict[str, Any]:
    """Generate a synthetic dataset once per process and reuse it.

    Cached because generating and writing PNGs is the slowest thing in the
    non-training tests, and every dataset test wants the same fixture.
    """
    from smalldet.data.synthetic import generate_dataset

    root = _scratch() / f"synthetic_{num_images}_{seed}"
    if not (root / "annotations_train.json").is_file():
        generate_dataset(
            root,
            num_images=num_images,
            image_size=(160, 160),
            objects_per_image=(3, 6),
            seed=seed,
        )
    return {
        "root": root,
        "images": root / "images",
        "train": root / "annotations_train.json",
        "val": root / "annotations_val.json",
    }


def _scratch() -> Path:
    """A writable directory for test artefacts, outside the project tree."""
    import tempfile

    path = Path(tempfile.gettempdir()) / "smalldet-tests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scratch_dir(name: str, clean: bool = False) -> Path:
    """A directory for test artefacts.

    Pass ``clean=True`` when a test asserts on which files exist: the scratch
    root survives between runs, so a leftover ``best.pt`` from a previous run
    would otherwise satisfy an assertion that nothing was written.
    """
    import shutil

    path = _scratch() / name
    if clean and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def tiny_config(**overrides: Any) -> Any:
    """A config small and fast enough to run a real model in a test.

    ``weights: null`` matters: pretrained weights are a ~160 MB download, which
    would make the suite slow and network-dependent. Randomly initialised
    weights exercise every code path the same way.
    """
    from smalldet.config import config_from_dict, deep_merge

    data = synthetic_dataset()
    base: Dict[str, Any] = {
        "name": "test",
        "data": {
            "root": str(data["root"]),
            "train": {"images": "images", "annotations": "annotations_train.json"},
            "val": {"images": "images", "annotations": "annotations_val.json"},
            "train_loader": {"batch_size": 2, "shuffle": True, "num_workers": 0},
            "eval_loader": {"batch_size": 1, "shuffle": False, "num_workers": 0},
            "augmentation": {
                "train": [
                    {"name": "random_horizontal_flip", "params": {"p": 0.5}},
                    {"name": "to_dtype", "params": {"dtype": "float32", "scale": True}},
                    {"name": "to_pure_tensor"},
                ],
                "eval": [
                    {"name": "to_dtype", "params": {"dtype": "float32", "scale": True}},
                    {"name": "to_pure_tensor"},
                ],
            },
        },
        "model": {
            "architecture": "fasterrcnn_resnet50_fpn_v2",
            "weights": None,
            "weights_backbone": None,
            "min_size": 160,
            "max_size": 200,
            "anchors": {"enabled": True, "base_sizes": [4, 8, 16, 32, 64]},
            "kwargs": {"box_detections_per_img": 50},
        },
        "finetune": {"strategy": "partial", "trainable_backbone_layers": 1},
        "optimizer": {"lr": 0.001},
        "scheduler": {
            "name": "step",
            "kwargs": {"step_size": 1, "gamma": 0.5},
            "warmup": {"enabled": True, "iters": 2, "epochs": 1},
        },
        "train": {
            "epochs": 1,
            "device": "cpu",
            "output_dir": str(scratch_dir("run")),
            "max_train_batches": 2,
            "max_eval_batches": 2,
            "log_interval": 1,
            "callbacks": [],
            "checkpoint": {"dir": str(scratch_dir("run/checkpoints"))},
        },
        "predict": {"device": "cpu", "postprocess": {"score_threshold": 0.0}},
    }
    return config_from_dict(deep_merge(base, overrides))


def approx(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def assert_close(actual: float, expected: float, tolerance: float = 1e-6, label: str = "") -> None:
    if not approx(actual, expected, tolerance):
        raise AssertionError(
            f"{label or 'value'}: expected {expected}, got {actual} "
            f"(tolerance {tolerance})"
        )


def expect_error(
    func: Callable[[], Any], exception: type = Exception, contains: str = ""
) -> BaseException:
    """Assert that ``func`` raises, and return the exception for inspection."""
    try:
        func()
    except exception as exc:  # noqa: BLE001 - the type is the caller's choice
        if contains and contains.lower() not in str(exc).lower():
            raise AssertionError(
                f"expected the error to mention {contains!r}, got: {exc}"
            ) from exc
        return exc
    raise AssertionError(f"expected {exception.__name__} to be raised, but it was not")
