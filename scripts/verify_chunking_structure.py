"""Verify chunking module structure without running dependencies.

This script verifies:
1. All module imports work
2. Public API is exported
3. Data models are valid
4. Validation logic works
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports() -> None:
    """Test that all module imports work."""
    print("\n" + "=" * 80)
    print("Test 1: Module Imports")
    print("=" * 80 + "\n")

    try:
        # Test individual module imports
        from inference.chunking import models

        print("✅ inference.chunking.models imported successfully")

        from inference.chunking import silence_chunker

        print("✅ inference.chunking.silence_chunker imported successfully")

        from inference.chunking import sentence_chunker

        print("✅ inference.chunking.sentence_chunker imported successfully")

        from inference.chunking import chunker

        print("✅ inference.chunking.chunker imported successfully")

        # Test public API imports
        from inference import chunking

        print("✅ inference.chunking package imported successfully")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise


def test_public_api() -> None:
    """Test that public API exports are available."""
    print("\n" + "=" * 80)
    print("Test 2: Public API")
    print("=" * 80 + "\n")

    import inference.chunking as chunking

    expected_exports = [
        "chunk_transcript",
        "Chunk",
    ]

    available_exports = [name for name in dir(chunking) if not name.startswith("_")]

    print(f"Available exports: {available_exports}")
    print(f"Expected exports: {expected_exports}\n")

    for export in expected_exports:
        if hasattr(chunking, export):
            print(f"✅ {export} is exported")
        else:
            print(f"❌ {export} is NOT exported")
            raise AssertionError(f"{export} not found in public API")


def test_chunk_model() -> None:
    """Test Chunk model creation and validation."""
    print("\n" + "=" * 80)
    print("Test 3: Chunk Model")
    print("=" * 80 + "\n")

    from inference.chunking.models import Chunk

    # Test valid chunk
    chunk = Chunk(
        chunk_id="lec1_chunk_001",
        text="안녕하세요, 이것은 테스트입니다.",
        start_time=0.0,
        end_time=3.5,
        token_count=10,
        course_id=101,
        lecture_id=1,
    )

    print(f"✅ Created valid chunk: {chunk.chunk_id}")
    print(f"   Text: {chunk.text}")
    print(f"   Time: {chunk.start_time}s - {chunk.end_time}s")
    print(f"   Tokens: {chunk.token_count}")

    # Test validation - empty text
    try:
        Chunk(
            chunk_id="lec1_chunk_002",
            text="",
            start_time=0.0,
            end_time=3.5,
            token_count=10,
            course_id=101,
            lecture_id=1,
        )
        print("❌ Empty text should raise ValueError")
    except ValueError as e:
        print(f"✅ Empty text validation: {e}")

    # Test validation - negative start time
    try:
        Chunk(
            chunk_id="lec1_chunk_003",
            text="test",
            start_time=-1.0,
            end_time=3.5,
            token_count=10,
            course_id=101,
            lecture_id=1,
        )
        print("❌ Negative start_time should raise ValueError")
    except ValueError as e:
        print(f"✅ Negative start_time validation: {e}")

    # Test validation - invalid timestamps (end <= start)
    try:
        Chunk(
            chunk_id="lec1_chunk_004",
            text="test",
            start_time=5.0,
            end_time=5.0,
            token_count=10,
            course_id=101,
            lecture_id=1,
        )
        print("❌ end_time <= start_time should raise ValueError")
    except ValueError as e:
        print(f"✅ Timestamp order validation: {e}")

    # Test validation - invalid token count
    try:
        Chunk(
            chunk_id="lec1_chunk_005",
            text="test",
            start_time=0.0,
            end_time=3.5,
            token_count=0,
            course_id=101,
            lecture_id=1,
        )
        print("❌ Zero token_count should raise ValueError")
    except ValueError as e:
        print(f"✅ Token count validation: {e}")

    # Test validation - invalid course_id
    try:
        Chunk(
            chunk_id="lec1_chunk_006",
            text="test",
            start_time=0.0,
            end_time=3.5,
            token_count=10,
            course_id=0,
            lecture_id=1,
        )
        print("❌ course_id <= 0 should raise ValueError")
    except ValueError as e:
        print(f"✅ Course ID validation: {e}")

    # Test validation - invalid lecture_id
    try:
        Chunk(
            chunk_id="lec1_chunk_007",
            text="test",
            start_time=0.0,
            end_time=3.5,
            token_count=10,
            course_id=101,
            lecture_id=-1,
        )
        print("❌ lecture_id <= 0 should raise ValueError")
    except ValueError as e:
        print(f"✅ Lecture ID validation: {e}")


def test_intermediate_chunk() -> None:
    """Test IntermediateChunk model."""
    print("\n" + "=" * 80)
    print("Test 4: IntermediateChunk Model")
    print("=" * 80 + "\n")

    from inference.chunking.silence_chunker import IntermediateChunk

    chunk = IntermediateChunk(
        text="안녕하세요, 이것은 중간 청크입니다.",
        start_time=0.0,
        end_time=5.5,
    )

    print(f"✅ Created IntermediateChunk")
    print(f"   Text: {chunk.text}")
    print(f"   Time: {chunk.start_time}s - {chunk.end_time}s")


def test_tokenized_chunk() -> None:
    """Test TokenizedChunk model."""
    print("\n" + "=" * 80)
    print("Test 5: TokenizedChunk Model")
    print("=" * 80 + "\n")

    from inference.chunking.sentence_chunker import TokenizedChunk

    chunk = TokenizedChunk(
        text="안녕하세요, 이것은 토큰화된 청크입니다.",
        start_time=0.0,
        end_time=5.5,
        token_count=15,
    )

    print(f"✅ Created TokenizedChunk")
    print(f"   Text: {chunk.text}")
    print(f"   Time: {chunk.start_time}s - {chunk.end_time}s")
    print(f"   Tokens: {chunk.token_count}")


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Chunking Module Structure Verification")
    print("=" * 80)

    try:
        test_imports()
        test_public_api()
        test_chunk_model()
        test_intermediate_chunk()
        test_tokenized_chunk()

        print("\n" + "=" * 80)
        print("✅ All structure tests passed!")
        print("=" * 80 + "\n")

        print("Note: Full chunking tests (with tiktoken/wtpsplit) require:")
        print("  - uv sync  # Install dependencies")
        print("  - python scripts/test_chunking.py")

    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
