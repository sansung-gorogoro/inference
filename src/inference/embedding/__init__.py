"""Embedding module for BGE-M3 with ChromaDB integration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bge_embedder import BGEEmbedder
    from .embedder import Embedder

from .models import EmbeddingBatch, EmbeddingResult


# Lazy imports to avoid requiring torch/numpy at import time
def __getattr__(name: str):
    if name == "BGEEmbedder":
        from .bge_embedder import BGEEmbedder

        return BGEEmbedder
    elif name == "Embedder":
        from .embedder import Embedder

        return Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BGEEmbedder", "Embedder", "EmbeddingBatch", "EmbeddingResult"]
