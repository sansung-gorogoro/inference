"""Chunk data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Transcript chunk with metadata.

    Attributes:
        chunk_id: Stable chunk identifier (e.g., "lec1_chunk_001")
        text: Chunk text content
        start_time: Start timestamp in seconds (3 decimal precision)
        end_time: End timestamp in seconds (3 decimal precision)
        token_count: Number of tokens in the chunk
        course_id: Course identifier
        lecture_id: Lecture identifier
    """

    chunk_id: str
    text: str
    start_time: float
    end_time: float
    token_count: int
    course_id: int
    lecture_id: int

    def __post_init__(self) -> None:
        """Validate chunk constraints."""
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty")
        if self.start_time < 0.0:
            raise ValueError(f"Chunk start_time must be >= 0.0, got {self.start_time}")
        if self.end_time <= 0.0:
            raise ValueError(f"Chunk end_time must be > 0.0, got {self.end_time}")
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Chunk end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        if self.token_count <= 0:
            raise ValueError(f"Chunk token_count must be > 0, got {self.token_count}")
        if self.course_id <= 0:
            raise ValueError(f"Course ID must be > 0, got {self.course_id}")
        if self.lecture_id <= 0:
            raise ValueError(f"Lecture ID must be > 0, got {self.lecture_id}")
        if not self.chunk_id.strip():
            raise ValueError("Chunk ID cannot be empty")
