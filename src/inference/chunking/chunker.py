"""Main chunking orchestrator."""

from __future__ import annotations

import logging

from inference.chunking.models import Chunk
from inference.chunking.sentence_chunker import chunk_by_sentences
from inference.chunking.silence_chunker import chunk_by_silence
from inference.stt.models import TranscriptResult

logger = logging.getLogger(__name__)


def chunk_transcript(
    transcript: TranscriptResult,
    course_id: int,
    lecture_id: int,
    silence_threshold: float = 2.0,
    max_tokens: int = 800,
    overlap_tokens: int = 160,
) -> list[Chunk]:
    """Chunk transcript using two-pass strategy.

    Pass 1: Group segments by silence gaps
    Pass 2: Split oversized chunks at sentence boundaries with overlap

    Args:
        transcript: Transcription result with segments
        course_id: Course identifier (must be > 0)
        lecture_id: Lecture identifier (must be > 0)
        silence_threshold: Maximum gap in seconds to group segments (default: 2.0)
        max_tokens: Maximum tokens per chunk (default: 800)
        overlap_tokens: Number of overlap tokens between chunks (default: 160)

    Returns:
        List of chunks with stable IDs and timestamps

    Raises:
        ValueError: If course_id or lecture_id is invalid, or transcript is empty
    """
    if course_id <= 0:
        raise ValueError(f"Course ID must be > 0, got {course_id}")
    if lecture_id <= 0:
        raise ValueError(f"Lecture ID must be > 0, got {lecture_id}")
    if not transcript.segments:
        raise ValueError("Transcript must contain at least one segment")

    logger.info(
        "Starting transcript chunking",
        extra={
            "course_id": course_id,
            "lecture_id": lecture_id,
            "segments": len(transcript.segments),
            "silence_threshold": silence_threshold,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
    )

    # Pass 1: Silence-gap chunking
    intermediate_chunks = chunk_by_silence(
        segments=transcript.segments,
        silence_threshold=silence_threshold,
    )

    # Pass 2: Sentence-boundary chunking
    tokenized_chunks = chunk_by_sentences(
        chunks=intermediate_chunks,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Generate final chunks with stable IDs
    final_chunks: list[Chunk] = []

    for index, tokenized_chunk in enumerate(tokenized_chunks, start=1):
        chunk_id = f"lec{lecture_id}_chunk_{index:03d}"

        chunk = Chunk(
            chunk_id=chunk_id,
            text=tokenized_chunk.text,
            start_time=tokenized_chunk.start_time,
            end_time=tokenized_chunk.end_time,
            token_count=tokenized_chunk.token_count,
            course_id=course_id,
            lecture_id=lecture_id,
        )

        final_chunks.append(chunk)

    logger.info(
        "Transcript chunking completed",
        extra={
            "course_id": course_id,
            "lecture_id": lecture_id,
            "input_segments": len(transcript.segments),
            "output_chunks": len(final_chunks),
            "avg_tokens_per_chunk": (
                sum(c.token_count for c in final_chunks) / len(final_chunks)
                if final_chunks
                else 0
            ),
        },
    )

    return final_chunks
