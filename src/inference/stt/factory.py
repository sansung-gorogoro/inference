"""STT mode selection factory."""

from __future__ import annotations

import logging
import os
from typing import Protocol

from inference.stt.models import TranscriptResult

logger = logging.getLogger(__name__)


class STTEngine(Protocol):
    """Protocol for STT engine interface."""

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """Transcribe audio file to text with timestamps."""
        ...


def create_stt_engine(
    mode: str | None = None,
    **kwargs: object,
) -> STTEngine:
    """Create STT engine based on mode.

    Args:
        mode: STT mode - "local" or "api" (default: read from STT_MODE env var, fallback to "local")
        **kwargs: Additional arguments passed to engine constructor

    Returns:
        STTEngine instance (LocalSTT or OpenAISTT)

    Raises:
        ValueError: If mode is invalid

    Examples:
        >>> # Use environment variable
        >>> engine = create_stt_engine()

        >>> # Explicit local mode
        >>> engine = create_stt_engine(mode="local", model_name="medium")

        >>> # Explicit API mode
        >>> engine = create_stt_engine(mode="api")
    """
    # Resolve mode
    if mode is None:
        mode = os.getenv("STT_MODE", "local")

    mode = mode.lower().strip()

    if mode == "local":
        from inference.stt.local_stt import LocalSTT

        # Extract local-specific config
        model_name = kwargs.get("model_name") or os.getenv("WHISPER_MODEL", "large-v3")
        device = kwargs.get("device")
        compute_type = kwargs.get("compute_type")

        logger.info("Creating local STT engine", extra={"model": model_name})

        return LocalSTT(
            model_name=str(model_name),
            device=str(device) if device else None,
            compute_type=str(compute_type) if compute_type else None,
        )

    elif mode == "api":
        from inference.stt.api_stt import OpenAISTT

        # Extract API-specific config
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        logger.info("Creating OpenAI API STT engine")

        return OpenAISTT(api_key=str(api_key) if api_key else None)

    else:
        raise ValueError(
            f"Invalid STT mode: {mode}. Must be 'local' or 'api'. "
            + "Set STT_MODE environment variable or pass mode explicitly."
        )
