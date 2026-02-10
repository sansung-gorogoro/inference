"""OpenAI API STT implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from inference.stt.exceptions import (
    APIAuthError,
    APIRateLimitError,
    DecodeError,
    PipelineError,
)
from inference.stt.models import Segment, TranscriptResult

logger = logging.getLogger(__name__)


def _round_timestamp(value: float) -> float:
    """Round timestamp to 3 decimal places."""
    return round(value, 3)


class OpenAISTT:
    """Speech-to-text using OpenAI Whisper API.

    Requires OPENAI_API_KEY environment variable.
    Uses verbose_json response format to get timestamps.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize OpenAI STT client.

        Args:
            api_key: OpenAI API key (default: read from OPENAI_API_KEY env var)

        Raises:
            APIAuthError: If API key is not provided or found in environment
        """
        self.api_key: str | None = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise APIAuthError(
                message="OPENAI_API_KEY not found in environment or constructor",
                details={},
            )

        logger.info("OpenAISTT initialized")

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """Transcribe audio file using OpenAI API.

        Args:
            audio_path: Path to audio file (supports webm, mp3, wav, m4a, etc.)

        Returns:
            TranscriptResult with segments, language, and duration

        Raises:
            DecodeError: If audio file cannot be read
            APIAuthError: If API authentication fails
            APIRateLimitError: If API rate limit is exceeded
            PipelineError: For other transcription failures
        """
        # Defensive guard: validate file exists
        path = Path(audio_path)
        if not path.exists():
            raise DecodeError(
                message=f"Audio file not found: {audio_path}",
                details={"path": audio_path},
            )

        if not path.is_file():
            raise DecodeError(
                message=f"Path is not a file: {audio_path}",
                details={"path": audio_path},
            )

        logger.info("Starting OpenAI transcription", extra={"audio_path": audio_path})

        try:
            from openai import OpenAI, AuthenticationError, RateLimitError  # type: ignore

            client = OpenAI(api_key=self.api_key)

            # Open audio file
            with open(path, "rb") as audio_file:
                # Call OpenAI API with verbose_json for timestamps
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",  # Korean language hint
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            # Parse response
            segments = []
            for seg in response.segments:
                segments.append(
                    Segment(
                        text=seg.text.strip(),
                        start=_round_timestamp(seg.start),
                        end=_round_timestamp(seg.end),
                    )
                )

            if not segments:
                logger.warning(
                    "No speech detected in audio", extra={"audio_path": audio_path}
                )
                # Return single empty segment to satisfy non-empty constraint
                segments = [Segment(text="", start=0.0, end=0.001)]

            duration = _round_timestamp(response.duration)
            language = response.language or "ko"

            logger.info(
                "OpenAI transcription completed",
                extra={
                    "audio_path": audio_path,
                    "segments": len(segments),
                    "duration": duration,
                    "language": language,
                },
            )

            return TranscriptResult(
                segments=segments,
                language=language,
                duration=duration,
            )

        except ImportError as e:
            raise PipelineError(
                message="openai package not installed",
                details={"error": str(e)},
            ) from e

        except AuthenticationError as e:
            raise APIAuthError(
                message="OpenAI authentication failed - invalid API key",
                details={"error": str(e)},
            ) from e

        except RateLimitError as e:
            raise APIRateLimitError(
                message="OpenAI rate limit exceeded",
                details={"error": str(e)},
            ) from e

        except IOError as e:
            raise DecodeError(
                message=f"Failed to read audio file: {e}",
                details={"audio_path": audio_path, "error": str(e)},
            ) from e

        except Exception as e:
            # Generic pipeline error for unexpected issues
            raise PipelineError(
                message=f"OpenAI transcription failed: {e}",
                details={"audio_path": audio_path, "error": str(e)},
            ) from e
