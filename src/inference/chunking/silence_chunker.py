"""Silence-gap based chunking (first pass)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from inference.stt.models import Segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntermediateChunk:
    """Intermediate chunk from silence-gap chunking.

    Attributes:
        text: Combined text from segments
        start_time: Start timestamp of first segment
        end_time: End timestamp of last segment
    """

    text: str
    start_time: float
    end_time: float


def chunk_by_silence(
    segments: list[Segment],
    silence_threshold: float = 2.0,
) -> list[IntermediateChunk]:
    """Group consecutive segments by silence gaps.

    Args:
        segments: List of transcription segments
        silence_threshold: Maximum gap in seconds to consider segments as continuous

    Returns:
        List of intermediate chunks grouped by silence gaps

    Raises:
        ValueError: If segments list is empty
    """
    if not segments:
        raise ValueError("Segments list cannot be empty")

    logger.info(
        "Starting silence-gap chunking",
        extra={
            "segments": len(segments),
            "silence_threshold": silence_threshold,
        },
    )

    chunks: list[IntermediateChunk] = []
    current_texts: list[str] = []
    current_start: float = segments[0].start
    current_end: float = segments[0].end

    for i, segment in enumerate(segments):
        if i == 0:
            # First segment - initialize
            current_texts.append(segment.text)
            current_start = segment.start
            current_end = segment.end
        else:
            # Calculate gap from previous segment
            gap = segment.start - current_end

            if gap <= silence_threshold:
                # Continue current chunk
                current_texts.append(segment.text)
                current_end = segment.end
            else:
                # Silence gap detected - finalize current chunk
                combined_text = " ".join(current_texts).strip()
                if combined_text:
                    chunks.append(
                        IntermediateChunk(
                            text=combined_text,
                            start_time=current_start,
                            end_time=current_end,
                        )
                    )

                # Start new chunk
                current_texts = [segment.text]
                current_start = segment.start
                current_end = segment.end

    # Add final chunk
    combined_text = " ".join(current_texts).strip()
    if combined_text:
        chunks.append(
            IntermediateChunk(
                text=combined_text,
                start_time=current_start,
                end_time=current_end,
            )
        )

    logger.info(
        "Silence-gap chunking completed",
        extra={
            "input_segments": len(segments),
            "output_chunks": len(chunks),
        },
    )

    return chunks
