"""Dependency-free fallback runner for the test suite.

Prefer ``pytest tests/`` when pytest is installed — it gives better failure
output. This runner exists so the suite is still executable in an environment
where pytest cannot be installed, which is exactly the sort of environment a
detection project ends up on.

    python tests/run_tests.py                # everything
    python tests/run_tests.py test_coco_eval # one module
    python tests/run_tests.py -k anchor      # by substring
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR.parent))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from _support import SKIP_EXCEPTION  # noqa: E402


def discover(pattern: str = "") -> List[Tuple[str, str, Callable[[], None]]]:
    """Find ``test_*`` functions in ``test_*.py`` modules."""
    found: List[Tuple[str, str, Callable[[], None]]] = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        module_name = path.stem
        try:
            module = importlib.import_module(module_name)
        except SKIP_EXCEPTION as exc:
            print(f"  skipped module {module_name}: {exc}")
            continue
        except ImportError as exc:
            print(f"  skipped module {module_name}: {exc}")
            continue

        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            if pattern and pattern not in name and pattern not in module_name:
                continue
            found.append((module_name, name, func))
    return found


def run(pattern: str = "", verbose: bool = True) -> int:
    tests = discover(pattern)
    if not tests:
        print(f"no tests matched {pattern!r}")
        return 1

    passed = skipped = 0
    failures: List[Tuple[str, str, str]] = []
    started = time.perf_counter()
    current_module = None

    for module_name, name, func in tests:
        if module_name != current_module:
            current_module = module_name
            print(f"\n{module_name}")
        try:
            func()
        except SKIP_EXCEPTION as exc:
            skipped += 1
            if verbose:
                print(f"  s {name}  ({exc})")
        except Exception:  # noqa: BLE001 - a failing test is any exception
            failures.append((module_name, name, traceback.format_exc()))
            print(f"  F {name}")
        else:
            passed += 1
            if verbose:
                print(f"  . {name}")

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 70}")
    for module_name, name, tb in failures:
        print(f"\nFAILED {module_name}::{name}\n{tb}")
    print(
        f"{passed} passed, {len(failures)} failed, {skipped} skipped "
        f"in {elapsed:.1f}s"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", nargs="?", default="", help="module name filter")
    parser.add_argument("-k", dest="pattern", default="", help="substring filter")
    parser.add_argument("-q", dest="quiet", action="store_true")
    args = parser.parse_args()
    return run(args.pattern or args.module, verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
