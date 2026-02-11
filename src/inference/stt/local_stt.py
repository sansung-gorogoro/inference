"""Local STT implementation using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from inference.stt.exceptions import (
    DecodeError,
    OOMError,
    PipelineError,
    STTError,
)
from inference.stt.models import Segment, TranscriptResult

logger = logging.getLogger(__name__)


def _round_timestamp(value: float) -> float:
    """Round timestamp to 3 decimal places."""
    return round(value, 3)


class LocalSTT:
    """Local speech-to-text using faster-whisper.

    Uses Korean-optimized Whisper model with CUDA acceleration if available.
    Falls back to CPU with quantization if GPU is unavailable.
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
    ):
        """Initialize local STT engine.

        Args:
            model_name: Whisper model name (default: large-v3)
            device: Device to use ('cuda' or 'cpu', default: auto-detect)
            compute_type: Compute type ('float16', 'int8', default: auto-select)
        """
        self.model_name: str = model_name

        # Auto-detect device if not specified
        if device is None:
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device: str = device

        # Auto-select compute type based on device
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.compute_type: str = compute_type

        self._model: Any = None
        logger.info(
            "LocalSTT initialized",
            extra={
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
            },
        )

    def _ensure_model_loaded(self) -> Any:
        """Lazy load faster-whisper model."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel  # type: ignore

            logger.info(
                "Loading faster-whisper model", extra={"model": self.model_name}
            )
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Model loaded successfully")
            return self._model
        except ImportError as e:
            raise PipelineError(
                message="faster-whisper not installed",
                details={"error": str(e)},
            ) from e
        except Exception as e:
            # Handle OOM or other model loading errors
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                raise OOMError(
                    message="Out of memory loading Whisper model",
                    details={"model": self.model_name, "device": self.device},
                ) from e
            raise PipelineError(
                message=f"Failed to load Whisper model: {e}",
                details={"model": self.model_name},
            ) from e

    def transcribe(self, audio_path: str) -> TranscriptResult:
        """Transcribe audio file using faster-whisper.

        Args:
            audio_path: Path to audio file (supports webm, mp3, wav, m4a, etc.)

        Returns:
            TranscriptResult with segments, language, and duration

        Raises:
            STTError: If transcription fails (decode_error, oom, pipeline_error)
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

        logger.info("Starting transcription", extra={"audio_path": audio_path})

        try:
            model = self._ensure_model_loaded()

            # Transcribe with Korean language hint
            segments_iter, info = model.transcribe(
                str(path),
                language="ko",  # Korean language hint
                beam_size=5,  # Balance speed/accuracy
                vad_filter=True,  # Enable voice activity detection
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Detect pauses
                ),
            )

            # Convert iterator to list and normalize segments
            segments = []
            for seg in segments_iter:
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

            duration = _round_timestamp(info.duration)
            language = info.language or "ko"

            logger.info(
                "Transcription completed",
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

        except STTError:
            # Re-raise STTError without wrapping
            raise

        except IOError as e:
            raise DecodeError(
                message=f"Failed to read audio file: {e}",
                details={"audio_path": audio_path, "error": str(e)},
            ) from e

        except MemoryError as e:
            raise OOMError(
                message="Out of memory during transcription",
                details={"audio_path": audio_path},
            ) from e

        except Exception as e:
            # Check for CUDA OOM
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                raise OOMError(
                    message="GPU out of memory during transcription",
                    details={"audio_path": audio_path},
                ) from e

            # Generic pipeline error
            raise PipelineError(
                message=f"Transcription failed: {e}",
                details={"audio_path": audio_path, "error": str(e)},
            ) from e
