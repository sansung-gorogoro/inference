"""STT output data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Transcription segment with text and timestamps.

    Attributes:
        text: Transcribed text content
        start: Start timestamp in seconds (3 decimal precision)
        end: End timestamp in seconds (3 decimal precision)
    """

    text: str
    start: float
    end: float

    def __post_init__(self) -> None:
        """Validate segment constraints."""
        if not self.text.strip():
            raise ValueError("Segment text cannot be empty")
        if self.start < 0.0:
            raise ValueError(f"Segment start must be >= 0.0, got {self.start}")
        if self.end <= 0.0:
            raise ValueError(f"Segment end must be > 0.0, got {self.end}")
        if self.end <= self.start:
            raise ValueError(
                f"Segment end ({self.end}) must be greater than start ({self.start})"
            )


@dataclass(frozen=True)
class TranscriptResult:
    """Complete transcription result.

    Attributes:
        segments: List of transcribed segments with timestamps
        language: Detected or specified language code (e.g., 'ko', 'en')
        duration: Total audio duration in seconds
    """

    segments: list[Segment]
    language: str
    duration: float

    def __post_init__(self) -> None:
        """Validate transcript constraints."""
        if not self.segments:
            raise ValueError("TranscriptResult must contain at least one segment")
        if self.duration <= 0.0:
            raise ValueError(f"Duration must be > 0.0, got {self.duration}")
        if not self.language.strip():
            raise ValueError("Language code cannot be empty")
