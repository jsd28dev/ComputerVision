"""pytest configuration.

The package is used from the project root rather than installed, so the root
goes on ``sys.path`` before any test imports ``smalldet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
