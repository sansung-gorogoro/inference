#!/usr/bin/env python3
"""Verify STT module structure and imports.

This script validates the module structure without requiring
external dependencies (torch, faster-whisper, openai) to be installed.
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 60)
print("STT Module Structure Verification")
print("=" * 60)

# Test 1: Import models
print("\n[1/5] Importing models...")
try:
    from inference.stt.models import Segment, TranscriptResult

    print("  ✓ Segment imported")
    print("  ✓ TranscriptResult imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import exceptions
print("\n[2/5] Importing exceptions...")
try:
    from inference.stt.exceptions import (
        STTError,
        DecodeError,
        OOMError,
        APIRateLimitError,
        APIAuthError,
        PipelineError,
    )

    print("  ✓ All exception classes imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Import factory
print("\n[3/5] Importing factory...")
try:
    from inference.stt.factory import STTEngine, create_stt_engine

    print("  ✓ STTEngine protocol imported")
    print("  ✓ create_stt_engine factory imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Import public API
print("\n[4/5] Importing public API...")
try:
    import inference.stt

    expected_exports = [
        "create_stt_engine",
        "STTEngine",
        "Segment",
        "TranscriptResult",
        "STTError",
        "DecodeError",
        "OOMError",
        "APIRateLimitError",
        "APIAuthError",
        "PipelineError",
    ]
    for name in expected_exports:
        if not hasattr(inference.stt, name):
            print(f"  ✗ Missing export: {name}")
            sys.exit(1)
    print(f"  ✓ All {len(expected_exports)} exports available")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Test data models
print("\n[5/5] Testing data models...")
try:
    # Create valid segment
    seg = Segment(text="Hello world", start=0.0, end=1.234)
    assert seg.text == "Hello world"
    assert seg.start == 0.0
    assert seg.end == 1.234
    print("  ✓ Segment creation works")

    # Test segment validation
    try:
        Segment(text="", start=0.0, end=1.0)
        print("  ✗ Empty text validation failed")
        sys.exit(1)
    except ValueError:
        print("  ✓ Empty text validation works")

    try:
        Segment(text="test", start=-1.0, end=1.0)
        print("  ✗ Negative start validation failed")
        sys.exit(1)
    except ValueError:
        print("  ✓ Negative start validation works")

    try:
        Segment(text="test", start=2.0, end=1.0)
        print("  ✗ End <= start validation failed")
        sys.exit(1)
    except ValueError:
        print("  ✓ End <= start validation works")

    # Create valid transcript
    result = TranscriptResult(
        segments=[seg],
        language="ko",
        duration=1.234,
    )
    assert len(result.segments) == 1
    assert result.language == "ko"
    assert result.duration == 1.234
    print("  ✓ TranscriptResult creation works")

    # Test transcript validation
    try:
        TranscriptResult(segments=[], language="ko", duration=1.0)
        print("  ✗ Empty segments validation failed")
        sys.exit(1)
    except ValueError:
        print("  ✓ Empty segments validation works")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Test exception hierarchy
print("\n[6/6] Testing exception hierarchy...")
try:
    from inference.jobs.models import ErrorCode

    # Test DecodeError
    err = DecodeError("test error", {"key": "value"})
    assert err.code == ErrorCode.DECODE_ERROR
    assert err.message == "test error"
    assert err.details == {"key": "value"}
    assert isinstance(err, STTError)
    assert isinstance(err, Exception)
    print("  ✓ DecodeError works")

    # Test OOMError
    err = OOMError("out of memory")
    assert err.code == ErrorCode.OOM
    print("  ✓ OOMError works")

    # Test APIRateLimitError
    err = APIRateLimitError("rate limited")
    assert err.code == ErrorCode.API_RATE_LIMIT
    print("  ✓ APIRateLimitError works")

    # Test APIAuthError
    err = APIAuthError("auth failed")
    assert err.code == ErrorCode.API_AUTH_ERROR
    print("  ✓ APIAuthError works")

    # Test PipelineError
    err = PipelineError("pipeline failed")
    assert err.code == ErrorCode.PIPELINE_ERROR
    print("  ✓ PipelineError works")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All structure tests PASSED ✓")
print("=" * 60)
print("\nNote: This validates module structure only.")
print("Full transcription tests require external dependencies:")
print("  - torch (GPU support)")
print("  - faster-whisper (local STT)")
print("  - openai (API STT)")
print("\nThese dependencies are not available on macOS x86_64.")
print("Deploy on Linux x86_64/ARM64 or macOS ARM64 for full functionality.")
