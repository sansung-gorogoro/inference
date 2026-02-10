"""Embedding data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of embedding operation.

    Attributes:
        chunk_id: Chunk identifier
        embedding: Dense vector (1024-dim for BGE-M3)
    """

    chunk_id: str
    embedding: list[float]


@dataclass(frozen=True)
class EmbeddingBatch:
    """Batch of embedding results.

    Attributes:
        results: List of embedding results
        model_name: Name of the embedding model used
        dimension: Embedding dimension
    """

    results: list[EmbeddingResult]
    model_name: str
    dimension: int

    def __post_init__(self) -> None:
        """Validate batch constraints."""
        if not self.results:
            raise ValueError("EmbeddingBatch must contain at least one result")
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be > 0, got {self.dimension}")

        # Verify all embeddings have correct dimension
        for result in self.results:
            if len(result.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    + f"got {len(result.embedding)} for chunk {result.chunk_id}"
                )
