"""Main chunking orchestrator."""

from __future__ import annotations

import logging
import os
from typing import cast

from inference.chunking.adaptive_threshold import compute_adaptive_threshold
from inference.chunking.models import Chunk
from inference.chunking.sentence_chunker import chunk_by_sentences
from inference.chunking.silence_chunker import chunk_by_silence
from inference.stt.models import TranscriptResult

logger = logging.getLogger(__name__)


def chunk_transcript(
    transcript: TranscriptResult,
    course_id: int,
    lecture_id: int,
    silence_threshold: float | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> tuple[list[Chunk], dict[str, str | float | int]]:
    """Chunk transcript using two-pass strategy.

    Pass 1: Group segments by silence gaps
    Pass 2: Split oversized chunks at sentence boundaries with overlap

    Env vars (all optional; defaults remain unchanged):
        CHUNK_SILENCE_THRESHOLD_SECONDS: float, default 2.0
        CHUNK_SILENCE_THRESHOLD_MODE: 'fixed' or 'adaptive', default 'fixed'
        CHUNK_SILENCE_ADAPTIVE_QUANTILE: float, default 0.95
        CHUNK_SILENCE_ADAPTIVE_MIN_SECONDS: float, default 0.8
        CHUNK_SILENCE_ADAPTIVE_MAX_SECONDS: float, default 3.0
        CHUNK_SILENCE_ADAPTIVE_MIN_GAPS: int, default 10
        CHUNK_MAX_TOKENS: int, default 800
        CHUNK_OVERLAP_TOKENS: int, default 160

    Args:
        transcript: Transcription result with segments
        course_id: Course identifier (must be > 0)
        lecture_id: Lecture identifier (must be > 0)
        silence_threshold: Maximum gap in seconds to group segments (default: from env or 2.0)
        max_tokens: Maximum tokens per chunk (default: from env or 800)
        overlap_tokens: Number of overlap tokens between chunks (default: from env or 160)

    Returns:
        Tuple of (list of chunks with stable IDs and timestamps, dict of resolved parameters)

    Raises:
        ValueError: If course_id or lecture_id is invalid, or transcript is empty
    """
    if course_id <= 0:
        raise ValueError(f"Course ID must be > 0, got {course_id}")
    if lecture_id <= 0:
        raise ValueError(f"Lecture ID must be > 0, got {lecture_id}")
    if not transcript.segments:
        raise ValueError("Transcript must contain at least one segment")

    # Resolve env defaults: caller-provided explicit values always win
    resolved_silence_threshold = (
        silence_threshold
        if silence_threshold is not None
        else float(os.getenv("CHUNK_SILENCE_THRESHOLD_SECONDS", "2.0"))
    )
    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv("CHUNK_MAX_TOKENS", "800"))
    )
    resolved_overlap_tokens = (
        overlap_tokens
        if overlap_tokens is not None
        else int(os.getenv("CHUNK_OVERLAP_TOKENS", "160"))
    )

    # Resolve adaptive threshold mode and parameters
    silence_threshold_mode = os.getenv("CHUNK_SILENCE_THRESHOLD_MODE", "fixed")
    adaptive_quantile = float(os.getenv("CHUNK_SILENCE_ADAPTIVE_QUANTILE", "0.95"))
    adaptive_min_seconds = float(os.getenv("CHUNK_SILENCE_ADAPTIVE_MIN_SECONDS", "0.8"))
    adaptive_max_seconds = float(os.getenv("CHUNK_SILENCE_ADAPTIVE_MAX_SECONDS", "3.0"))
    adaptive_min_gaps = int(os.getenv("CHUNK_SILENCE_ADAPTIVE_MIN_GAPS", "10"))

    logger.info(
        "Starting transcript chunking",
        extra={
            "course_id": course_id,
            "lecture_id": lecture_id,
            "segments": len(transcript.segments),
            "silence_threshold_mode": silence_threshold_mode,
            "silence_threshold": resolved_silence_threshold,
            "max_tokens": resolved_max_tokens,
            "overlap_tokens": resolved_overlap_tokens,
        },
    )

    # Compute adaptive threshold if mode is 'adaptive'
    adaptive_stats = None
    if silence_threshold_mode == "adaptive":
        resolved_silence_threshold, adaptive_stats = compute_adaptive_threshold(
            segments=transcript.segments,
            quantile=adaptive_quantile,
            clamp_min=adaptive_min_seconds,
            clamp_max=adaptive_max_seconds,
            min_gaps=adaptive_min_gaps,
            fallback_threshold=resolved_silence_threshold,
        )

    # Pass 1: Silence-gap chunking
    intermediate_chunks = chunk_by_silence(
        segments=transcript.segments,
        silence_threshold=resolved_silence_threshold,
    )

    # Pass 2: Sentence-boundary chunking
    tokenized_chunks = chunk_by_sentences(
        chunks=intermediate_chunks,
        max_tokens=resolved_max_tokens,
        overlap_tokens=resolved_overlap_tokens,
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

    resolved_params: dict[str, str | float | int] = {
        "silence_threshold": resolved_silence_threshold,
        "max_tokens": resolved_max_tokens,
        "overlap_tokens": resolved_overlap_tokens,
    }

    if adaptive_stats is not None:
        # Merge adaptive stats (includes resolved silence_threshold_mode)
        resolved_params = cast(
            dict[str, str | float | int],
            {**resolved_params, **adaptive_stats},
        )
    else:
        # Fixed mode: set mode explicitly
        resolved_params["silence_threshold_mode"] = silence_threshold_mode

    return final_chunks, resolved_params
