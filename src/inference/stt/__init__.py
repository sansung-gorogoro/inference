"""Speech-to-text module with local and API modes.

Public API:
    - create_stt_engine: Factory function for creating STT engines
    - Segment: Transcription segment with text and timestamps
    - TranscriptResult: Complete transcription result
    - STTError: Base exception for STT errors
    - DecodeError, OOMError, APIRateLimitError, APIAuthError, PipelineError: Specific error types
    - STTEngine: Protocol for STT engine interface

Example:
    >>> from inference.stt import create_stt_engine
    >>>
    >>> # Create engine (uses STT_MODE env var)
    >>> engine = create_stt_engine()
    >>>
    >>> # Transcribe audio
    >>> result = engine.transcribe("audio.mp3")
    >>> print(result.segments[0].text)
    >>> print(f"Duration: {result.duration}s")
"""

from inference.stt.exceptions import (
    APIAuthError,
    APIRateLimitError,
    DecodeError,
    OOMError,
    PipelineError,
    STTError,
)
from inference.stt.factory import STTEngine, create_stt_engine
from inference.stt.models import Segment, TranscriptResult

__all__ = [
    # Factory
    "create_stt_engine",
    "STTEngine",
    # Models
    "Segment",
    "TranscriptResult",
    # Exceptions
    "STTError",
    "DecodeError",
    "OOMError",
    "APIRateLimitError",
    "APIAuthError",
    "PipelineError",
]
