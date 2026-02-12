#!/usr/bin/env python3
"""Standalone tests for adaptive silence threshold (no tiktoken dependency)."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.chunking.adaptive_threshold import compute_adaptive_threshold
from inference.stt.models import Segment


def test_adaptive_deterministic():
    """Test adaptive mode with known gap distribution."""
    print("\n" + "="*80)
    print("Test 1: Adaptive Mode - Deterministic Quantile")
    print("="*80)
    
    # Create segments with known gaps: [0.5, 1.0, 1.5, 2.0, 2.5]
    segments = [
        Segment('a', 0.0, 1.0),
        Segment('b', 1.5, 2.5),   # gap=0.5
        Segment('c', 3.5, 4.5),   # gap=1.0
        Segment('d', 6.0, 7.0),   # gap=1.5
        Segment('e', 9.0, 10.0),  # gap=2.0
        Segment('f', 12.5, 13.5), # gap=2.5
    ]
    
    threshold, stats = compute_adaptive_threshold(
        segments=segments,
        quantile=0.95,
        clamp_min=0.8,
        clamp_max=3.0,
        min_gaps=3,
        fallback_threshold=2.0
    )
    
    # Expected: 95th percentile of [0.5, 1.0, 1.5, 2.0, 2.5]
    # k = ceil(0.95 * 5) - 1 = 5 - 1 = 4
    # gaps[4] = 2.5, clamped to [0.8, 3.0] = 2.5
    assert threshold == 2.5, f"Expected 2.5, got {threshold}"
    assert stats['silence_threshold_mode'] == 'adaptive', f"Expected adaptive, got {stats['silence_threshold_mode']}"
    assert stats['gap_count_used'] == 5, f"Expected 5, got {stats['gap_count_used']}"
    assert stats['gap_count_ignored_negative'] == 0
    
    print(f"✓ Threshold: {threshold} (expected 2.5)")
    print(f"✓ Mode: {stats['silence_threshold_mode']}")
    print(f"✓ Gaps used: {stats['gap_count_used']}")
    print("✅ Test passed")


def test_adaptive_fallback():
    """Test adaptive fallback with too few gaps."""
    print("\n" + "="*80)
    print("Test 2: Adaptive Fallback - Insufficient Gaps")
    print("="*80)
    
    # Only 2 segments = 1 gap (< min_gaps=10)
    segments = [
        Segment('a', 0.0, 1.0),
        Segment('b', 2.0, 3.0),  # gap=1.0
    ]
    
    threshold, stats = compute_adaptive_threshold(
        segments=segments,
        quantile=0.95,
        clamp_min=0.8,
        clamp_max=3.0,
        min_gaps=10,
        fallback_threshold=2.0
    )
    
    # Expected: fallback to 2.0 (only 1 gap < min_gaps=10)
    assert threshold == 2.0, f"Expected fallback 2.0, got {threshold}"
    assert stats['silence_threshold_mode'] == 'adaptive_fallback', f"Expected fallback mode, got {stats['silence_threshold_mode']}"
    assert stats['gap_count_used'] == 1
    
    print(f"✓ Threshold: {threshold} (fallback to 2.0)")
    print(f"✓ Mode: {stats['silence_threshold_mode']}")
    print(f"✓ Gaps used: {stats['gap_count_used']} (< min_gaps=10)")
    print("✅ Test passed")


def test_adaptive_negative_gaps():
    """Test adaptive mode with overlapping segments."""
    print("\n" + "="*80)
    print("Test 3: Adaptive Mode - Negative Gap Filtering")
    print("="*80)
    
    # Segments with overlap (negative gap)
    segments = [
        Segment('a', 0.0, 2.0),
        Segment('b', 1.5, 3.0),  # gap=-0.5 (overlap)
        Segment('c', 4.0, 5.0),  # gap=1.0
        Segment('d', 6.5, 7.5),  # gap=1.5
    ]
    
    threshold, stats = compute_adaptive_threshold(
        segments=segments,
        quantile=0.95,
        clamp_min=0.8,
        clamp_max=3.0,
        min_gaps=2,
        fallback_threshold=2.0
    )
    
    # Expected: negative gap ignored, only [1.0, 1.5] used
    assert stats['gap_count_total'] == 3
    assert stats['gap_count_used'] == 2, f"Expected 2 usable gaps, got {stats['gap_count_used']}"
    assert stats['gap_count_ignored_negative'] == 1, f"Expected 1 negative gap, got {stats['gap_count_ignored_negative']}"
    assert stats['silence_threshold_mode'] == 'adaptive'
    
    print(f"✓ Total gaps: {stats['gap_count_total']}")
    print(f"✓ Used gaps: {stats['gap_count_used']}")
    print(f"✓ Ignored negative: {stats['gap_count_ignored_negative']}")
    print(f"✓ Threshold: {threshold}")
    print("✅ Test passed")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Adaptive Silence Threshold Test Suite")
    print("="*80)
    
    try:
        test_adaptive_deterministic()
        test_adaptive_fallback()
        test_adaptive_negative_gaps()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
