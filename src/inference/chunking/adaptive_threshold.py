"""Adaptive silence threshold computation for chunking.

This module provides deterministic quantile-based threshold resolution
for silence-based transcript chunking. It computes inter-segment gaps
and returns a threshold based on the specified quantile.

Algorithm:
    1. Sort segments by start time (stable sort)
    2. Compute gaps: gap_i = segments[i].start - segments[i-1].end
    3. Filter: keep only non-negative gaps (ignore overlaps/jitter)
    4. If usable gaps < min_gaps: return fallback threshold
    5. Compute nearest-rank quantile:
       - k = ceil(q * len(gaps)) - 1
       - Clamp k to [0, len(gaps)-1]
       - Return sorted_gaps[k]
    6. Clamp result to [clamp_min, clamp_max]

Design Decisions:
    - Deterministic: no float interpolation, no numpy
    - Robust: handles unsorted input, negative gaps, edge cases
    - Traceable: returns stats dict for debugging
"""

from __future__ import annotations

import math
from typing import TypedDict

from src.inference.stt.models import Segment


class AdaptiveThresholdStats(TypedDict):
    """Statistics from adaptive threshold computation.

    Attributes:
        mode: Computation mode ('adaptive' or 'adaptive_fallback')
        gap_count_total: Total number of gaps computed
        gap_count_used: Number of non-negative gaps used for quantile
        gap_count_ignored_negative: Number of negative gaps ignored
        quantile: Requested quantile value (0.0-1.0)
        clamp_min: Minimum threshold clamp value
        clamp_max: Maximum threshold clamp value
        min_gaps: Minimum required gaps for adaptive mode
        fallback_threshold: Fallback value used when gaps < min_gaps
    """

    mode: str
    gap_count_total: int
    gap_count_used: int
    gap_count_ignored_negative: int
    quantile: float
    clamp_min: float
    clamp_max: float
    min_gaps: int
    fallback_threshold: float


def compute_adaptive_threshold(
    segments: list[Segment],
    quantile: float = 0.95,
    clamp_min: float = 0.8,
    clamp_max: float = 3.0,
    min_gaps: int = 10,
    fallback_threshold: float = 2.0,
) -> tuple[float, AdaptiveThresholdStats]:
    """Compute adaptive silence threshold from segment gaps.

    Args:
        segments: List of transcription segments (will be sorted by start time)
        quantile: Quantile to compute (0.0-1.0, default: 0.95 for 95th percentile)
        clamp_min: Minimum threshold value (default: 0.8 seconds)
        clamp_max: Maximum threshold value (default: 3.0 seconds)
        min_gaps: Minimum number of gaps required for adaptive mode (default: 10)
        fallback_threshold: Threshold to use when gaps < min_gaps (default: 2.0)

    Returns:
        Tuple of (resolved_threshold, stats_dict)
        - resolved_threshold: Computed threshold in seconds
        - stats_dict: Metadata about computation (mode, gap counts, etc.)

    Raises:
        ValueError: If segments is empty, quantile out of range, or invalid clamps

    Examples:
        >>> segments = [
        ...     Segment('a', 0.0, 1.0),
        ...     Segment('b', 1.5, 2.5),   # gap=0.5
        ...     Segment('c', 3.5, 4.5),   # gap=1.0
        ... ]
        >>> threshold, stats = compute_adaptive_threshold(segments, quantile=0.95)
        >>> threshold  # 95th percentile of [0.5, 1.0]
        1.0
        >>> stats['mode']
        'adaptive_fallback'  # Only 2 gaps, less than min_gaps=10
    """
    # Validate inputs
    if not segments:
        raise ValueError("segments cannot be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0.0, 1.0], got {quantile}")
    if clamp_min < 0.0:
        raise ValueError(f"clamp_min must be >= 0.0, got {clamp_min}")
    if clamp_max < clamp_min:
        raise ValueError(f"clamp_max ({clamp_max}) must be >= clamp_min ({clamp_min})")
    if min_gaps < 1:
        raise ValueError(f"min_gaps must be >= 1, got {min_gaps}")

    # Sort segments by start time (stable sort for determinism)
    sorted_segments = sorted(segments, key=lambda s: s.start)

    # Compute gaps: gap_i = segments[i].start - segments[i-1].end
    gaps: list[float] = []
    for i in range(1, len(sorted_segments)):
        gap = sorted_segments[i].start - sorted_segments[i - 1].end
        gaps.append(gap)

    gap_count_total = len(gaps)

    # Filter: keep only non-negative gaps (ignore overlaps/jitter)
    non_negative_gaps = [g for g in gaps if g >= 0.0]
    gap_count_used = len(non_negative_gaps)
    gap_count_ignored_negative = gap_count_total - gap_count_used

    # Build stats dict (common fields)
    stats: AdaptiveThresholdStats = {
        "mode": "adaptive",
        "gap_count_total": gap_count_total,
        "gap_count_used": gap_count_used,
        "gap_count_ignored_negative": gap_count_ignored_negative,
        "quantile": quantile,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "min_gaps": min_gaps,
        "fallback_threshold": fallback_threshold,
    }

    # Fallback if insufficient gaps
    if gap_count_used < min_gaps:
        stats["mode"] = "adaptive_fallback"
        return (fallback_threshold, stats)

    # Compute nearest-rank quantile
    # k = ceil(q * len(gaps)) - 1, clamped to [0, len-1]
    sorted_gaps = sorted(non_negative_gaps)
    k = math.ceil(quantile * len(sorted_gaps)) - 1
    k = max(0, min(k, len(sorted_gaps) - 1))
    threshold = sorted_gaps[k]

    # Clamp to [clamp_min, clamp_max]
    threshold = max(clamp_min, min(threshold, clamp_max))

    return (threshold, stats)
