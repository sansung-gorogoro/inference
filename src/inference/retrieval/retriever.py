"""Main retrieval module with lecture-scoped querying and deduplication."""

from __future__ import annotations

import logging

from ..embedding.bge_embedder import BGEEmbedder
from ..vectorstore.chroma_client import ChromaVectorStore
from ..vectorstore.schemas import QueryFilter
from .models import Citation, RetrievalResult

logger = logging.getLogger(__name__)


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    # Tokenize by whitespace and convert to sets
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    if not tokens1 and not tokens2:
        return 1.0  # Both empty

    if not tokens1 or not tokens2:
        return 0.0  # One empty

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


def deduplicate_chunks(
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, str | int | float]],
    distances: list[float],
    similarity_threshold: float = 0.8,
) -> tuple[list[str], list[str], list[dict[str, str | int | float]], list[float]]:
    """Deduplicate overlapping chunks using Jaccard similarity.

    Compares consecutive chunks and removes if similarity > threshold.
    Assumes chunks are already sorted by relevance (distance).

    Args:
        chunk_ids: List of chunk IDs
        texts: List of chunk texts
        metadatas: List of metadata dicts
        distances: List of similarity distances
        similarity_threshold: Jaccard similarity threshold (default: 0.8)

    Returns:
        Tuple of deduplicated (chunk_ids, texts, metadatas, distances)
    """
    if not texts:
        return [], [], [], []

    if len(texts) == 1:
        return chunk_ids, texts, metadatas, distances

    # Track indices to keep
    keep_indices = [0]  # Always keep first result

    for i in range(1, len(texts)):
        # Compare with previous kept chunk
        prev_idx = keep_indices[-1]
        current_text = texts[i]
        previous_text = texts[prev_idx]

        similarity = calculate_jaccard_similarity(current_text, previous_text)

        if similarity < similarity_threshold:
            keep_indices.append(i)
        else:
            logger.debug(
                f"Deduplicating chunk {chunk_ids[i]} (similarity={similarity:.3f} with {chunk_ids[prev_idx]})"
            )

    # Filter to keep only non-duplicate indices
    dedup_chunk_ids = [chunk_ids[i] for i in keep_indices]
    dedup_texts = [texts[i] for i in keep_indices]
    dedup_metadatas = [metadatas[i] for i in keep_indices]
    dedup_distances = [distances[i] for i in keep_indices]

    logger.info(
        f"Deduplication: {len(texts)} -> {len(dedup_texts)} chunks (removed {len(texts) - len(dedup_texts)})"
    )

    return dedup_chunk_ids, dedup_texts, dedup_metadatas, dedup_distances


class Retriever:
    """Lecture-scoped retrieval with deduplication.

    Queries ChromaDB with strict lecture-level filtering and
    deduplicates overlapping chunks from 20% overlap in chunking.
    """

    _vectorstore: ChromaVectorStore
    _embedder: BGEEmbedder

    def __init__(
        self,
        vectorstore: ChromaVectorStore | None = None,
        embedder: BGEEmbedder | None = None,
    ) -> None:
        """Initialize retriever.

        Args:
            vectorstore: ChromaVectorStore instance (creates new if None)
            embedder: BGEEmbedder instance (creates new if None)
        """
        self._vectorstore = vectorstore or ChromaVectorStore()
        self._embedder = embedder or BGEEmbedder()

        logger.info(
            f"Initialized Retriever with model={self._embedder.model_name}, device={self._embedder.device}"
        )

    def retrieve(
        self,
        query: str,
        course_id: int,
        lecture_id: int,
        top_k: int = 10,
        similarity_threshold: float | None = None,
        dedup_threshold: float = 0.8,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query within a specific lecture.

        Args:
            query: Natural language query
            course_id: Course ID to filter by
            lecture_id: Lecture ID to filter by
            top_k: Maximum number of results to retrieve (default: 10)
            similarity_threshold: Optional distance threshold for filtering results
            dedup_threshold: Jaccard similarity threshold for deduplication (default: 0.8)

        Returns:
            RetrievalResult with deduplicated citations

        Raises:
            ValueError: If query is empty or IDs are invalid
            RuntimeError: If embedding or retrieval fails
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if course_id <= 0:
            raise ValueError(f"course_id must be > 0, got {course_id}")

        if lecture_id <= 0:
            raise ValueError(f"lecture_id must be > 0, got {lecture_id}")

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        logger.info(
            f"Retrieving for query='{query[:50]}...' course_id={course_id}, lecture_id={lecture_id}, top_k={top_k}"
        )

        try:
            # Step 1: Embed query
            logger.debug(f"Embedding query: '{query[:50]}...'")
            query_embedding = self._embedder.embed_query(query)

            # Step 2: Query ChromaDB with lecture filter
            filter = QueryFilter(course_id=course_id, lecture_id=lecture_id)
            logger.debug(f"Querying ChromaDB with filter={filter.to_where_clause()}")

            results = self._vectorstore.query_by_lecture(
                query_embedding=query_embedding,
                filter=filter,
                top_k=top_k,
            )

            # Step 3: Extract results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not documents:
                logger.warning(
                    f"No results found for course_id={course_id}, lecture_id={lecture_id}"
                )
                return RetrievalResult(
                    query=query,
                    citations=[],
                    total_retrieved=0,
                    total_deduplicated=0,
                )

            total_retrieved = len(documents)
            logger.info(f"Retrieved {total_retrieved} chunks from ChromaDB")

            # Step 4: Filter by similarity threshold if provided
            if similarity_threshold is not None:
                filtered_indices = [
                    i
                    for i, dist in enumerate(distances)
                    if dist <= similarity_threshold
                ]

                documents = [documents[i] for i in filtered_indices]
                metadatas = [metadatas[i] for i in filtered_indices]
                distances = [distances[i] for i in filtered_indices]

                logger.info(
                    f"Filtered by similarity_threshold={similarity_threshold}: {total_retrieved} -> {len(documents)} chunks"
                )

            # Step 5: Extract chunk IDs for deduplication
            chunk_ids = [meta["chunk_id"] for meta in metadatas]

            # Step 6: Deduplicate overlapping chunks
            logger.debug(
                f"Deduplicating with threshold={dedup_threshold}, chunks={len(documents)}"
            )
            dedup_ids, dedup_texts, dedup_metadatas, dedup_distances = (
                deduplicate_chunks(
                    chunk_ids=chunk_ids,
                    texts=documents,
                    metadatas=metadatas,
                    distances=distances,
                    similarity_threshold=dedup_threshold,
                )
            )

            # Step 7: Create citations with type validation
            citations: list[Citation] = []
            for i in range(len(dedup_ids)):
                meta = dedup_metadatas[i]

                start_time_val = meta["start_time"]
                end_time_val = meta["end_time"]
                course_id_val = meta["course_id"]
                lecture_id_val = meta["lecture_id"]

                if not isinstance(start_time_val, (int, float)):
                    raise RuntimeError(
                        f"Invalid start_time type: {type(start_time_val)}"
                    )
                if not isinstance(end_time_val, (int, float)):
                    raise RuntimeError(f"Invalid end_time type: {type(end_time_val)}")
                if not isinstance(course_id_val, int):
                    raise RuntimeError(f"Invalid course_id type: {type(course_id_val)}")
                if not isinstance(lecture_id_val, int):
                    raise RuntimeError(
                        f"Invalid lecture_id type: {type(lecture_id_val)}"
                    )

                citations.append(
                    Citation(
                        chunk_id=dedup_ids[i],
                        text=dedup_texts[i],
                        start_time=float(start_time_val),
                        end_time=float(end_time_val),
                        course_id=course_id_val,
                        lecture_id=lecture_id_val,
                        similarity_score=dedup_distances[i],
                    )
                )

            logger.info(
                f"Retrieval complete: {total_retrieved} retrieved, {len(citations)} after deduplication"
            )

            return RetrievalResult(
                query=query,
                citations=citations,
                total_retrieved=total_retrieved,
                total_deduplicated=len(citations),
            )

        except ValueError as e:
            logger.error(f"Validation error during retrieval: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {e}")
            raise RuntimeError(f"Retrieval operation failed: {e}") from e

    @property
    def model_name(self) -> str:
        """Get embedding model name."""
        return self._embedder.model_name

    @property
    def device(self) -> str:
        """Get device (cuda/cpu)."""
        return self._embedder.device

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedder.dimension
