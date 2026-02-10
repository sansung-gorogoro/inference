"""Retrieval module for lecture-scoped semantic search.

Provides lecture-scoped retrieval with automatic deduplication of overlapping chunks.

Example:
    >>> from inference.retrieval import Retriever, RetrievalResult, Citation
    >>>
    >>> retriever = Retriever()
    >>>
    >>> result = retriever.retrieve(
    ...     query="What is machine learning?",
    ...     course_id=101,
    ...     lecture_id=1,
    ...     top_k=10
    ... )
    >>>
    >>> print(f"Found {result.total_deduplicated} chunks")
    >>> for citation in result.citations:
    ...     print(f"{citation.chunk_id}: {citation.text[:50]}...")
"""

from .models import Citation, RetrievalResult
from .retriever import Retriever

__all__ = ["Retriever", "RetrievalResult", "Citation"]
