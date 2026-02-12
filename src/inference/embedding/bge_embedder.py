"""BGE-M3 embedding implementation."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer  # type: ignore

from ..chunking.models import Chunk
from .models import EmbeddingBatch, EmbeddingResult

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
MAX_BATCH_SIZE = 64  # For 8GB VRAM constraint
EMBEDDING_DIMENSION = 1024  # BGE-M3 output dimension


class BGEEmbedder:
    """BGE-M3 embedder with batch processing and device selection.

    Automatically selects CUDA if available, otherwise CPU.
    Processes chunks in batches of <=64 for 8GB VRAM constraint.
    """

    _model: SentenceTransformer
    _device: str

    def __init__(self) -> None:
        """Initialize BGE-M3 model.

        Raises:
            RuntimeError: If model loading fails
        """
        # Parse EMBEDDING_DEVICE env var (auto|cpu|cuda, case-insensitive)
        device_env = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
        if device_env == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_env == "cpu":
            self._device = "cpu"
        elif device_env == "cuda":
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                logger.warning(
                    "EMBEDDING_DEVICE=cuda requested but CUDA unavailable, falling back to CPU"
                )
                self._device = "cpu"
        else:
            logger.warning(
                f"Invalid EMBEDDING_DEVICE='{device_env}', must be 'auto|cpu|cuda', using 'auto'"
            )
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Parse EMBEDDING_BATCH_SIZE env var (int, clamped to [1, MAX_BATCH_SIZE])
        batch_size_env = os.getenv("EMBEDDING_BATCH_SIZE", "").strip()
        if batch_size_env:
            try:
                batch_size_int = int(batch_size_env)
                if batch_size_int < 1:
                    logger.warning(
                        f"EMBEDDING_BATCH_SIZE={batch_size_int} is invalid (must be >=1), using default 64"
                    )
                    self._batch_size = MAX_BATCH_SIZE
                elif batch_size_int > MAX_BATCH_SIZE:
                    logger.warning(
                        f"EMBEDDING_BATCH_SIZE={batch_size_int} exceeds max {MAX_BATCH_SIZE}, clamping to {MAX_BATCH_SIZE}"
                    )
                    self._batch_size = MAX_BATCH_SIZE
                else:
                    self._batch_size = batch_size_int
            except ValueError:
                logger.warning(
                    f"EMBEDDING_BATCH_SIZE='{batch_size_env}' is not a valid integer, using default 64"
                )
                self._batch_size = MAX_BATCH_SIZE
        else:
            self._batch_size = MAX_BATCH_SIZE

        try:
            logger.info(
                f"Loading BGE-M3 model on device: {self._device}, batch_size: {self._batch_size}"
            )
            self._model = SentenceTransformer(MODEL_NAME, device=self._device)
            logger.info(
                f"BGE-M3 model loaded successfully on {self._device} with batch_size={self._batch_size}"
            )
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise RuntimeError(f"BGE-M3 model initialization failed: {e}") from e

    def embed_chunks(self, chunks: list[Chunk]) -> EmbeddingBatch:
        """Embed chunks with batch processing.

        Args:
            chunks: List of chunks to embed

        Returns:
            EmbeddingBatch with results

        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If embedding fails
        """
        if not chunks:
            raise ValueError("Cannot embed empty chunks list")

        logger.info(
            f"Embedding {len(chunks)} chunks with batch_size={self._batch_size}, device={self._device}"
        )

        texts = [chunk.text for chunk in chunks]

        try:
            # Encode with batch processing
            embeddings_array: Any = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert numpy array to list of lists
            if isinstance(embeddings_array, np.ndarray):
                embeddings_list: list[list[float]] = embeddings_array.tolist()
            else:
                raise RuntimeError(
                    f"Unexpected embedding type: {type(embeddings_array)}"
                )

            # Create results
            results = [
                EmbeddingResult(chunk_id=chunk.chunk_id, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings_list, strict=True)
            ]

            logger.info(f"Successfully embedded {len(results)} chunks")

            return EmbeddingBatch(
                results=results, model_name=MODEL_NAME, dimension=EMBEDDING_DIMENSION
            )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Failed to embed chunks: {e}") from e

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If query is empty
            RuntimeError: If embedding fails
        """
        if not query.strip():
            raise ValueError("Cannot embed empty query")

        logger.debug(f"Embedding query: '{query[:50]}...'")

        try:
            # Encode single query (use effective batch size, but min 1)
            effective_batch_size = max(1, self._batch_size)
            embedding_array: Any = self._model.encode(
                [query],
                batch_size=effective_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert numpy array to list
            if isinstance(embedding_array, np.ndarray):
                embedding_list: list[float] = embedding_array[0].tolist()
            else:
                raise RuntimeError(
                    f"Unexpected embedding type: {type(embedding_array)}"
                )

            logger.debug(f"Successfully embedded query (dim={len(embedding_list)})")

            return embedding_list

        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RuntimeError(f"Failed to embed query: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats

        Raises:
            ValueError: If texts list is empty or contains empty strings
            RuntimeError: If embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty texts list")

        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed texts with empty strings")

        logger.info(
            f"Embedding {len(texts)} texts with batch_size={self._batch_size}, device={self._device}"
        )

        try:
            # Encode with batch processing
            embeddings_array: Any = self._model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert numpy array to list of lists
            if isinstance(embeddings_array, np.ndarray):
                embeddings_list: list[list[float]] = embeddings_array.tolist()
            else:
                raise RuntimeError(
                    f"Unexpected embedding type: {type(embeddings_array)}"
                )

            logger.debug(f"Successfully embedded {len(embeddings_list)} texts")

            return embeddings_list

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise RuntimeError(f"Failed to embed texts: {e}") from e

    @property
    def model_name(self) -> str:
        """Get model name."""
        return MODEL_NAME

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return EMBEDDING_DIMENSION
