"""Main embedder interface with ChromaDB integration."""

from __future__ import annotations

import logging

from ..chunking.models import Chunk
from ..vectorstore.chroma_client import ChromaVectorStore
from ..vectorstore.schemas import LectureChunk
from .bge_embedder import BGEEmbedder

logger = logging.getLogger(__name__)


class Embedder:
    """Main embedder with ChromaDB integration.

    Provides idempotent embedding and indexing operations.
    For same (course_id, lecture_id), replaces existing vectors.
    """

    _bge: BGEEmbedder
    _vectorstore: ChromaVectorStore

    def __init__(
        self,
        vectorstore: ChromaVectorStore | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        """Initialize embedder with vectorstore.

        Args:
            vectorstore: ChromaVectorStore instance (creates new if None)
            embedder: BGEEmbedder instance (creates new if None)
        """
        self._bge = embedder or BGEEmbedder()
        self._vectorstore = vectorstore or ChromaVectorStore()

        logger.info(
            f"Initialized Embedder with model={self._bge.model_name}, device={self._bge.device}"
        )

    def embed_and_index(
        self, course_id: int, lecture_id: int, chunks: list[Chunk]
    ) -> None:
        """Embed chunks and index into ChromaDB (idempotent).

        For same (course_id, lecture_id), deletes existing vectors before upserting.

        Args:
            course_id: Course ID
            lecture_id: Lecture ID
            chunks: List of chunks to embed and index

        Raises:
            ValueError: If chunks list is empty
            RuntimeError: If embedding or indexing fails
        """
        if not chunks:
            raise ValueError("Cannot embed and index empty chunks list")

        logger.info(
            f"Starting embed_and_index for course_id={course_id}, lecture_id={lecture_id}, chunks={len(chunks)}"
        )

        try:
            # Step 1: Delete existing vectors for idempotency
            logger.info(
                f"Deleting existing vectors for course_id={course_id}, lecture_id={lecture_id}"
            )
            self._vectorstore.delete_lecture_chunks(course_id, lecture_id)

            # Step 2: Embed chunks
            logger.info(f"Embedding {len(chunks)} chunks")
            embedding_batch = self._bge.embed_chunks(chunks)

            # Step 3: Convert to LectureChunk format for vectorstore
            lecture_chunks = [
                LectureChunk(
                    text=chunk.text,
                    course_id=chunk.course_id,
                    lecture_id=chunk.lecture_id,
                    chunk_id=chunk.chunk_id,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                )
                for chunk in chunks
            ]

            # Step 4: Extract embeddings
            embeddings = [result.embedding for result in embedding_batch.results]

            # Step 5: Upsert into ChromaDB
            logger.info(f"Upserting {len(lecture_chunks)} chunks into ChromaDB")
            self._vectorstore.upsert_chunks(
                course_id=course_id,
                lecture_id=lecture_id,
                chunks=lecture_chunks,
                embeddings=embeddings,
            )

            logger.info(
                f"Successfully embedded and indexed {len(chunks)} chunks for "
                + f"course_id={course_id}, lecture_id={lecture_id}"
            )

        except ValueError as e:
            logger.error(f"Validation error during embed_and_index: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error during embed_and_index: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embed_and_index: {e}")
            raise RuntimeError(f"Embed and index operation failed: {e}") from e

    @property
    def model_name(self) -> str:
        """Get embedding model name."""
        return self._bge.model_name

    @property
    def device(self) -> str:
        """Get device (cuda/cpu)."""
        return self._bge.device

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._bge.dimension
