"""Ground-truth box-size statistics, and area-range calibration from them.

COCO's small/medium/large cut-offs (32² and 96² pixels) are calibrated to
COCO's own ~640px images. Ported unchanged onto a 4000px industrial frame they
stop discriminating: every object lands in "small", ``AP_small`` becomes a copy
of ``AP``, and ``AP_medium`` reports the -1 sentinel because its bucket is
empty. Since ``AP_small`` and ``AP_medium`` are the metrics this project is
optimised against, the buckets have to be checked against the actual data.

``eval.auto_area_ranges: true`` replaces them with percentile cuts of this
dataset's own areas, which keeps all three buckets populated by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

_UNBOUNDED_AREA = 1e10


@dataclass(frozen=True)
class AreaStats:
    """Summary of a split's ground-truth object sizes, in pixels²."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    percentiles: Dict[float, float]
    #: Share of objects falling in each configured bucket.
    bucket_fractions: Dict[str, float]

    def describe(self) -> str:
        lines = [
            f"ground-truth objects: {self.count}",
            f"area px^2  min {self.minimum:.0f}  median {self.median:.0f}  "
            f"mean {self.mean:.0f}  max {self.maximum:.0f}",
            "equivalent square side: "
            + "  ".join(
                f"p{percentile:g}={value ** 0.5:.0f}px"
                for percentile, value in sorted(self.percentiles.items())
            ),
        ]
        if self.bucket_fractions:
            lines.append(
                "bucket occupancy: "
                + "  ".join(
                    f"{label}={fraction:.1%}"
                    for label, fraction in self.bucket_fractions.items()
                    if label != "all"
                )
            )
        return "\n".join(lines)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, matching ``numpy.percentile``.

    Implemented here so the data module stays importable without NumPy.
    """
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile must lie in [0, 100] (got {q})")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_areas(
    areas: Sequence[float],
    area_ranges: Dict[str, Sequence[float]] | None = None,
    probes: Sequence[float] = (1.0, 25.0, 50.0, 75.0, 99.0),
) -> AreaStats:
    """Describe a set of ground-truth areas and how they fall into buckets."""
    if not areas:
        raise ValueError(
            "no ground-truth areas to summarise — the split has no annotations"
        )
    values = [float(area) for area in areas]
    fractions: Dict[str, float] = {}
    for label, bounds in (area_ranges or {}).items():
        low, high = float(bounds[0]), float(bounds[1])
        hits = sum(1 for area in values if low <= area < high)
        fractions[label] = hits / len(values)

    return AreaStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=sum(values) / len(values),
        median=percentile(values, 50.0),
        percentiles={q: percentile(values, q) for q in probes},
        bucket_fractions=fractions,
    )


def area_ranges_from_percentiles(
    areas: Sequence[float], percentiles: Sequence[float] = (33.3, 66.6)
) -> Dict[str, List[float]]:
    """Derive small/medium/large buckets that split this dataset evenly.

    Returns the same four labels COCO uses, so every downstream consumer —
    metric names, checkpoint monitor, report headings — keeps working.
    """
    if len(percentiles) != 2:
        raise ValueError(
            "need exactly two percentiles: the small/medium and medium/large cuts"
        )
    low_cut, high_cut = (percentile(areas, q) for q in sorted(percentiles))
    if high_cut <= low_cut:
        # A dataset where a third of the objects share one exact area; nudge the
        # upper cut so the buckets stay non-empty and validation passes.
        high_cut = low_cut + 1.0
    return {
        "all": [0.0, _UNBOUNDED_AREA],
        "small": [0.0, float(low_cut)],
        "medium": [float(low_cut), float(high_cut)],
        "large": [float(high_cut), _UNBOUNDED_AREA],
    }
