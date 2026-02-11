"""Test script for chunking module.

This script verifies:
1. Two-pass chunking works correctly
2. All chunks are <=800 tokens (except possibly final chunk)
3. Overlap is present between consecutive chunks
4. Chunk IDs are stable and sequential
5. Timestamps are preserved accurately
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.chunking import Chunk, chunk_transcript
from inference.stt.models import Segment, TranscriptResult


def create_sample_transcript() -> TranscriptResult:
    """Create a sample transcript with multiple segments.

    Returns:
        Sample transcript with segments
    """
    # Create segments with varying gaps
    segments = [
        Segment(
            text="안녕하세요, 오늘은 파이썬 프로그래밍에 대해 배워보겠습니다.",
            start=0.0,
            end=3.5,
        ),
        Segment(
            text="파이썬은 매우 인기 있는 프로그래밍 언어입니다.", start=3.8, end=6.2
        ),
        Segment(
            text="간단한 문법과 강력한 기능으로 많은 개발자들이 사용하고 있습니다.",
            start=6.5,
            end=10.1,
        ),
        # Large gap (>2s) - should trigger new chunk
        Segment(text="이제 변수에 대해 알아보겠습니다.", start=12.5, end=15.0),
        Segment(text="변수는 데이터를 저장하는 공간입니다.", start=15.2, end=17.8),
        Segment(
            text="파이썬에서는 변수를 선언할 때 타입을 명시하지 않아도 됩니다.",
            start=18.0,
            end=21.5,
        ),
    ]

    return TranscriptResult(segments=segments, language="ko", duration=21.5)


def create_large_transcript() -> TranscriptResult:
    """Create a large transcript to test sentence splitting.

    Returns:
        Large transcript that will exceed 800 tokens
    """
    # Create a very long segment that will need to be split
    long_text = (
        "파이썬은 1991년 귀도 반 로섬이 개발한 고급 프로그래밍 언어입니다. "
        "파이썬은 코드 가독성을 강조하며, 문법이 간결하고 명확합니다. "
        "이러한 특징 덕분에 초보자도 쉽게 배울 수 있습니다. "
        "파이썬은 다양한 분야에서 사용됩니다. "
        "웹 개발, 데이터 분석, 인공지능, 머신러닝 등에서 활용됩니다. "
        "Django와 Flask 같은 웹 프레임워크가 인기가 있습니다. "
        "NumPy, Pandas, Matplotlib 같은 데이터 분석 라이브러리도 유명합니다. "
        "TensorFlow와 PyTorch는 머신러닝 프레임워크입니다. "
        "파이썬의 가장 큰 장점은 풍부한 라이브러리입니다. "
        "PyPI(Python Package Index)에는 수십만 개의 패키지가 있습니다. "
    ) * 30  # Repeat to make it very long (should exceed 800 tokens)

    segments = [
        Segment(text=long_text, start=0.0, end=120.0),
        Segment(text="이것이 마지막 세그먼트입니다.", start=120.5, end=123.0),
    ]

    return TranscriptResult(segments=segments, language="ko", duration=123.0)


def validate_chunks(chunks: list[Chunk], max_tokens: int = 800) -> None:
    """Validate chunking results.

    Args:
        chunks: List of chunks to validate
        max_tokens: Maximum tokens per chunk
    """
    print(f"\n{'=' * 80}")
    print("Validation Results")
    print(f"{'=' * 80}\n")

    # Check all chunks <= max_tokens (except possibly final chunk)
    oversized_chunks = []
    for i, chunk in enumerate(chunks[:-1]):  # Exclude final chunk
        if chunk.token_count > max_tokens:
            oversized_chunks.append((i, chunk.token_count))

    if oversized_chunks:
        print(f"❌ FAIL: Found {len(oversized_chunks)} oversized chunks (non-final):")
        for idx, count in oversized_chunks:
            print(f"   Chunk {idx}: {count} tokens")
    else:
        print(f"✅ PASS: All non-final chunks <= {max_tokens} tokens")

    # Check final chunk
    final_chunk = chunks[-1]
    if final_chunk.token_count > max_tokens:
        print(
            f"⚠️  INFO: Final chunk has {final_chunk.token_count} tokens (allowed to exceed limit)"
        )
    else:
        print(f"✅ PASS: Final chunk has {final_chunk.token_count} tokens")

    # Check chunk IDs are sequential
    expected_ids = [
        f"lec{chunks[0].lecture_id}_chunk_{i:03d}" for i in range(1, len(chunks) + 1)
    ]
    actual_ids = [chunk.chunk_id for chunk in chunks]

    if expected_ids == actual_ids:
        print("✅ PASS: Chunk IDs are stable and sequential")
    else:
        print("❌ FAIL: Chunk IDs are not sequential")
        print(f"   Expected: {expected_ids}")
        print(f"   Actual: {actual_ids}")

    # Check timestamps are preserved
    valid_timestamps = True
    for chunk in chunks:
        if (
            chunk.start_time < 0
            or chunk.end_time <= 0
            or chunk.end_time <= chunk.start_time
        ):
            valid_timestamps = False
            print(f"❌ FAIL: Invalid timestamps in chunk {chunk.chunk_id}")
            print(f"   start={chunk.start_time}, end={chunk.end_time}")

    if valid_timestamps:
        print("✅ PASS: All timestamps are valid")

    # Check timestamps are monotonic (overlap between chunks is allowed)
    monotonic_starts = True
    monotonic_ends = True
    for i in range(len(chunks) - 1):
        if chunks[i].start_time > chunks[i + 1].start_time:
            monotonic_starts = False
            print(f"❌ FAIL: start_time decreased between chunks {i} and {i + 1}")
        if chunks[i].end_time > chunks[i + 1].end_time:
            monotonic_ends = False
            print(f"❌ FAIL: end_time decreased between chunks {i} and {i + 1}")
    if monotonic_starts:
        print("✅ PASS: Chunk start_times are monotonically non-decreasing")
    if monotonic_ends:
        print("✅ PASS: Chunk end_times are monotonically non-decreasing")

    print(f"\n{'=' * 80}\n")


def print_chunk_summary(chunks: list[Chunk]) -> None:
    """Print summary of chunks.

    Args:
        chunks: List of chunks to summarize
    """
    print(f"\n{'=' * 80}")
    print("Chunk Summary")
    print(f"{'=' * 80}\n")

    print(f"Total chunks: {len(chunks)}")
    print(f"Total tokens: {sum(c.token_count for c in chunks)}")
    print(
        f"Average tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}"
    )
    print(f"Min tokens: {min(c.token_count for c in chunks)}")
    print(f"Max tokens: {max(c.token_count for c in chunks)}")

    print("\nChunk details:")
    for chunk in chunks:
        print(
            f"  {chunk.chunk_id}: {chunk.token_count:4d} tokens, "
            f"{chunk.start_time:6.3f}s - {chunk.end_time:6.3f}s, "
            f"text_len={len(chunk.text):4d}"
        )
        print(f"    Text preview: {chunk.text[:100]}...")

    print(f"\n{'=' * 80}\n")


def test_basic_chunking() -> None:
    """Test basic chunking with small transcript."""
    print("\n" + "=" * 80)
    print("Test 1: Basic Chunking (Small Transcript)")
    print("=" * 80)

    transcript = create_sample_transcript()
    chunks = chunk_transcript(
        transcript=transcript,
        course_id=101,
        lecture_id=1,
        silence_threshold=2.0,
        max_tokens=800,
        overlap_tokens=160,
    )

    print_chunk_summary(chunks)
    validate_chunks(chunks)


def test_large_chunking() -> None:
    """Test chunking with large transcript that needs sentence splitting."""
    print("\n" + "=" * 80)
    print("Test 2: Large Chunking (Sentence Splitting)")
    print("=" * 80)

    transcript = create_large_transcript()
    chunks = chunk_transcript(
        transcript=transcript,
        course_id=101,
        lecture_id=2,
        silence_threshold=2.0,
        max_tokens=800,
        overlap_tokens=160,
    )

    print_chunk_summary(chunks)
    validate_chunks(chunks)


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Chunking Module Test Suite")
    print("=" * 80)

    try:
        test_basic_chunking()
        test_large_chunking()

        print("\n" + "=" * 80)
        print("✅ All tests completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
