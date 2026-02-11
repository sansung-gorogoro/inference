"""Retrieval data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """Citation for a retrieved chunk.

    Attributes:
        chunk_id: Unique chunk identifier
        text: Chunk text content
        start_time: Start timestamp in seconds (3 decimal precision)
        end_time: End timestamp in seconds (3 decimal precision)
        course_id: Course identifier
        lecture_id: Lecture identifier
        similarity_score: Similarity score (lower is better for distance-based metrics)
    """

    chunk_id: str
    text: str
    start_time: float
    end_time: float
    course_id: int
    lecture_id: int
    similarity_score: float

    def __post_init__(self) -> None:
        """Validate citation constraints."""
        if not self.text.strip():
            raise ValueError("Citation text cannot be empty")
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be >= 0.0, got {self.start_time}")
        if self.end_time <= 0.0:
            raise ValueError(f"end_time must be > 0.0, got {self.end_time}")
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        if self.course_id <= 0:
            raise ValueError(f"course_id must be > 0, got {self.course_id}")
        if self.lecture_id <= 0:
            raise ValueError(f"lecture_id must be > 0, got {self.lecture_id}")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")
        if self.similarity_score < 0.0:
            raise ValueError(
                f"similarity_score must be >= 0.0, got {self.similarity_score}"
            )


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a retrieval query.

    Attributes:
        query: Original query text
        citations: List of retrieved citations (after deduplication)
        total_retrieved: Total number of chunks retrieved before deduplication
        total_deduplicated: Number of chunks after deduplication
    """

    query: str
    citations: list[Citation]
    total_retrieved: int
    total_deduplicated: int

    def __post_init__(self) -> None:
        """Validate retrieval result constraints."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.total_retrieved < 0:
            raise ValueError(
                f"total_retrieved must be >= 0, got {self.total_retrieved}"
            )
        if self.total_deduplicated < 0:
            raise ValueError(
                f"total_deduplicated must be >= 0, got {self.total_deduplicated}"
            )
        if self.total_deduplicated > self.total_retrieved:
            raise ValueError(
                f"total_deduplicated ({self.total_deduplicated}) cannot exceed "
                + f"total_retrieved ({self.total_retrieved})"
            )
        if len(self.citations) != self.total_deduplicated:
            raise ValueError(
                f"citations count ({len(self.citations)}) must match "
                + f"total_deduplicated ({self.total_deduplicated})"
            )
