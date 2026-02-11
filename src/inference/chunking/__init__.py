"""Chunking module for lecture transcript processing.

This module provides two-pass chunking:
1. Silence-gap chunking: Group segments by silence gaps
2. Sentence-boundary chunking: Split oversized chunks at sentence boundaries

Public API:
    - chunk_transcript: Main function to chunk a transcript
    - Chunk: Chunk data model
"""

from __future__ import annotations

from inference.chunking.chunker import chunk_transcript
from inference.chunking.models import Chunk

__all__ = [
    "chunk_transcript",
    "Chunk",
]
