"""STT-specific exceptions."""

from __future__ import annotations

from inference.jobs.models import ErrorCode


class STTError(Exception):
    """Base exception for STT errors.

    Attributes:
        code: Error code from ErrorCode enum
        message: Human-readable error message
        details: Optional additional error details
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, object] | None = None,
    ):
        super().__init__(message)
        self.code: ErrorCode = code
        self.message: str = message
        self.details: dict[str, object] = details or {}


class DecodeError(STTError):
    """Audio file decoding error."""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(ErrorCode.DECODE_ERROR, message, details)


class OOMError(STTError):
    """Out of memory error."""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(ErrorCode.OOM, message, details)


class APIRateLimitError(STTError):
    """API rate limit exceeded."""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(ErrorCode.API_RATE_LIMIT, message, details)


class APIAuthError(STTError):
    """API authentication error."""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(ErrorCode.API_AUTH_ERROR, message, details)


class PipelineError(STTError):
    """Generic pipeline error."""

    def __init__(self, message: str, details: dict[str, object] | None = None):
        super().__init__(ErrorCode.PIPELINE_ERROR, message, details)
