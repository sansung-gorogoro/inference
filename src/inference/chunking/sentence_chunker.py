"""Sentence-boundary based chunking (second pass)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    import tiktoken  # type: ignore[import-not-found,import-untyped]
except ImportError:
    tiktoken = None  # type: ignore[assignment]

try:
    from wtpsplit import SaT  # type: ignore[import-not-found,import-untyped]
except ImportError:
    SaT = None  # type: ignore[assignment,misc]

from inference.chunking.silence_chunker import IntermediateChunk

logger = logging.getLogger(__name__)

# WARNING: Changing token limits (CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS) changes chunk boundaries
# and can invalidate existing embeddings/index. For repeatable quiz-only regeneration,
# keep these values stable per dataset unless you intend to re-index.


@dataclass(frozen=True)
class TokenizedChunk:
    """Chunk with token count and adjusted timestamps.

    Attributes:
        text: Chunk text
        start_time: Start timestamp
        end_time: End timestamp
        token_count: Number of tokens
    """

    text: str
    start_time: float
    end_time: float
    token_count: int


class TokenCounter:
    """Token counter using tiktoken."""

    _encoding: Any

    def __init__(self) -> None:
        """Initialize token counter with cl100k_base encoding."""
        if tiktoken is None:
            raise RuntimeError(
                "tiktoken is not installed. Install with: pip install tiktoken"
            )

        self._encoding = tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
        logger.info("Initialized tiktoken with cl100k_base encoding")

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        tokens = self._encoding.encode(text)  # type: ignore[attr-defined]
        return len(tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decode tokens back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self._encoding.decode(tokens)  # type: ignore[attr-defined,no-any-return]

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self._encoding.encode(text)  # type: ignore[attr-defined,no-any-return]


class SentenceSplitter:
    """Sentence splitter using wtpsplit."""

    _model: Any | None

    def __init__(self) -> None:
        """Initialize sentence splitter with wtpsplit."""
        if SaT is None:
            logger.warning("wtpsplit not installed, will use fallback regex splitter")
            self._model = None
        else:
            try:
                self._model = SaT("sat-3l-sm")  # type: ignore[misc]
                logger.info("Initialized wtpsplit with sat-3l-sm model")
            except Exception as e:
                logger.warning(
                    "Failed to load wtpsplit model, will use fallback",
                    extra={"error": str(e)},
                )
                self._model = None

    def split(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if self._model is not None:
            try:
                # wtpsplit returns list of sentences
                sentences = self._model.split(text)  # type: ignore[attr-defined]
                return sentences if isinstance(sentences, list) else [text]
            except Exception as e:
                logger.warning(
                    "wtpsplit failed, falling back to regex",
                    extra={"error": str(e)},
                )

        # Fallback: simple regex-based sentence splitting
        import re

        sentences = re.split(r"[.!?]+\s+", text)
        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(
    chunks: list[IntermediateChunk],
    max_tokens: int = 800,
    overlap_tokens: int = 0,
) -> list[TokenizedChunk]:
    """Split oversized chunks at sentence boundaries with overlap.

    Args:
        chunks: List of intermediate chunks from silence-gap chunking
        max_tokens: Maximum tokens per chunk (default: 800)
        overlap_tokens: Number of overlap tokens between chunks (default: 0)

    Returns:
        List of tokenized chunks with sentence-boundary splits and overlap

    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    logger.info(
        "Starting sentence-boundary chunking",
        extra={
            "input_chunks": len(chunks),
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
    )

    token_counter = TokenCounter()
    sentence_splitter = SentenceSplitter()

    output_chunks: list[TokenizedChunk] = []

    for chunk in chunks:
        token_count = token_counter.count(chunk.text)

        if token_count <= max_tokens:
            # Chunk is within limit - keep as-is
            output_chunks.append(
                TokenizedChunk(
                    text=chunk.text,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    token_count=token_count,
                )
            )
            continue

        # Chunk is too large - split at sentence boundaries
        sentences = sentence_splitter.split(chunk.text)

        # If no sentences detected, treat as single sentence
        if not sentences:
            sentences = [chunk.text]

        # Calculate timestamp per character for proportional splitting
        total_chars = len(chunk.text)
        duration = chunk.end_time - chunk.start_time
        time_per_char = duration / total_chars if total_chars > 0 else 0.0

        # Build sub-chunks with overlap
        current_tokens: list[int] = []
        current_sentences: list[str] = []
        current_char_offset = 0

        for sentence in sentences:
            sentence_tokens = token_counter.encode(sentence)

            # Check if adding this sentence would exceed limit
            if (
                current_tokens
                and len(current_tokens) + len(sentence_tokens) > max_tokens
            ):
                # Finalize current sub-chunk
                sub_text = " ".join(current_sentences).strip()
                sub_start = chunk.start_time + (current_char_offset * time_per_char)
                sub_end = sub_start + (len(sub_text) * time_per_char)

                output_chunks.append(
                    TokenizedChunk(
                        text=sub_text,
                        start_time=round(sub_start, 3),
                        end_time=round(sub_end, 3),
                        token_count=len(current_tokens),
                    )
                )

                # Start new sub-chunk with overlap
                # Keep last N tokens from previous chunk
                if overlap_tokens > 0 and len(current_tokens) > overlap_tokens:
                    overlap_token_ids = current_tokens[-overlap_tokens:]
                    overlap_text = token_counter.decode(overlap_token_ids)

                    # Reset with overlap
                    current_tokens = list(overlap_token_ids) + list(sentence_tokens)
                    current_sentences = [overlap_text, sentence]
                    current_char_offset += len(sub_text) - len(overlap_text)
                else:
                    # No overlap or not enough tokens for overlap: start fresh
                    current_tokens = list(sentence_tokens)
                    current_sentences = [sentence]
                    current_char_offset += len(sub_text)
            else:
                # Add sentence to current sub-chunk
                current_sentences.append(sentence)
                current_tokens.extend(sentence_tokens)

        # Add final sub-chunk if any sentences remain
        if current_sentences:
            sub_text = " ".join(current_sentences).strip()
            sub_start = chunk.start_time + (current_char_offset * time_per_char)
            sub_end = chunk.end_time  # Final chunk ends at original chunk end

            output_chunks.append(
                TokenizedChunk(
                    text=sub_text,
                    start_time=round(sub_start, 3),
                    end_time=round(sub_end, 3),
                    token_count=len(current_tokens),
                )
            )

    logger.info(
        "Sentence-boundary chunking completed",
        extra={
            "input_chunks": len(chunks),
            "output_chunks": len(output_chunks),
        },
    )

    return output_chunks
