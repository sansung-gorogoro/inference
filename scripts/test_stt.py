#!/usr/bin/env python3
"""Test script for STT module.

Tests both local and API modes with a sample audio file.

Usage:
    # Test with local mode (default)
    python scripts/test_stt.py path/to/audio.mp3

    # Test with API mode
    STT_MODE=api OPENAI_API_KEY=sk-... python scripts/test_stt.py path/to/audio.mp3

    # Test specific model
    WHISPER_MODEL=medium python scripts/test_stt.py path/to/audio.mp3
"""

import logging
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.stt import create_stt_engine, STTError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def test_transcription(audio_path: str) -> None:
    """Test STT transcription with given audio file.

    Args:
        audio_path: Path to audio file
    """
    logger.info("=" * 60)
    logger.info("STT Module Test")
    logger.info("=" * 60)

    # Validate audio file exists
    path = Path(audio_path)
    if not path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        sys.exit(1)

    logger.info(f"Audio file: {path.absolute()}")
    logger.info(f"File size: {path.stat().st_size / 1024:.2f} KB")

    try:
        # Create STT engine (uses env vars for configuration)
        logger.info("\n--- Creating STT Engine ---")
        engine = create_stt_engine()
        logger.info(f"Engine type: {type(engine).__name__}")

        # Transcribe
        logger.info("\n--- Starting Transcription ---")
        result = engine.transcribe(str(path))

        # Display results
        logger.info("\n--- Transcription Results ---")
        logger.info(f"Language: {result.language}")
        logger.info(f"Duration: {result.duration:.3f}s")
        logger.info(f"Segments: {len(result.segments)}")

        # Display first 3 segments
        logger.info("\n--- Sample Segments ---")
        for i, segment in enumerate(result.segments[:3], 1):
            logger.info(
                f"Segment {i}: [{segment.start:.3f}s - {segment.end:.3f}s] "
                f"{segment.text[:100]}{'...' if len(segment.text) > 100 else ''}"
            )

        if len(result.segments) > 3:
            logger.info(f"... ({len(result.segments) - 3} more segments)")

        # Validate output structure
        logger.info("\n--- Validation ---")

        # Check all segments have valid timestamps
        for i, seg in enumerate(result.segments):
            assert seg.start >= 0.0, f"Segment {i}: start < 0"
            assert seg.end > 0.0, f"Segment {i}: end <= 0"
            assert seg.end > seg.start, f"Segment {i}: end <= start"

            # Check timestamp precision (3 decimals)
            start_str = f"{seg.start:.3f}"
            end_str = f"{seg.end:.3f}"
            assert float(start_str) == seg.start, f"Segment {i}: start has > 3 decimals"
            assert float(end_str) == seg.end, f"Segment {i}: end has > 3 decimals"

        logger.info("✓ All segments have valid timestamps")
        logger.info("✓ Timestamp precision is 3 decimals")
        logger.info("✓ Output structure is correct")

        logger.info("\n" + "=" * 60)
        logger.info("Test PASSED")
        logger.info("=" * 60)

    except STTError as e:
        logger.error("\n--- STT Error ---")
        logger.error(f"Error code: {e.code.value}")
        logger.error(f"Message: {e.message}")
        logger.error(f"Details: {e.details}")
        logger.error("\n" + "=" * 60)
        logger.error("Test FAILED")
        logger.error("=" * 60)
        sys.exit(1)

    except Exception as e:
        logger.exception("Unexpected error during transcription")
        logger.error("\n" + "=" * 60)
        logger.error("Test FAILED")
        logger.error("=" * 60)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_stt.py <audio_file>")
        print("\nExample:")
        print("  python scripts/test_stt.py inputs/sample.webm")
        print(
            "  STT_MODE=api OPENAI_API_KEY=sk-... python scripts/test_stt.py inputs/sample.webm"
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    test_transcription(audio_path)


if __name__ == "__main__":
    main()
