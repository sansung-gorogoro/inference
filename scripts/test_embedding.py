#!/usr/bin/env python3
"""Test script for embedding module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.chunking.models import Chunk
from inference.embedding import Embedder
from inference.vectorstore.chroma_client import ChromaVectorStore
from inference.vectorstore.schemas import QueryFilter


def test_embedding_and_indexing() -> None:
    """Test embedding and ChromaDB indexing."""
    print("=== Testing Embedding and Indexing ===\n")

    # Create test chunks
    chunks = [
        Chunk(
            chunk_id="test_chunk_001",
            text="머신러닝은 인공지능의 한 분야입니다. Machine learning is a field of AI.",
            start_time=0.0,
            end_time=5.5,
            token_count=15,
            course_id=1,
            lecture_id=1,
        ),
        Chunk(
            chunk_id="test_chunk_002",
            text="딥러닝은 신경망을 사용하는 머신러닝 기법입니다. Deep learning uses neural networks.",
            start_time=5.5,
            end_time=10.0,
            token_count=18,
            course_id=1,
            lecture_id=1,
        ),
        Chunk(
            chunk_id="test_chunk_003",
            text="자연어 처리는 텍스트를 이해하는 기술입니다. NLP understands text.",
            start_time=10.0,
            end_time=14.2,
            token_count=16,
            course_id=1,
            lecture_id=1,
        ),
    ]

    print(f"Created {len(chunks)} test chunks\n")

    # Initialize embedder
    print("Initializing Embedder...")
    embedder = Embedder()
    print(f"  Model: {embedder.model_name}")
    print(f"  Device: {embedder.device}")
    print(f"  Dimension: {embedder.dimension}\n")

    # Test 1: Embed and index (first time)
    print("Test 1: Embedding and indexing chunks (first time)...")
    embedder.embed_and_index(course_id=1, lecture_id=1, chunks=chunks)
    print("  ✓ Successfully embedded and indexed\n")

    # Verify in ChromaDB
    print("Verifying in ChromaDB...")
    vectorstore = ChromaVectorStore()
    stats = vectorstore.get_collection_stats()
    print(f"  Total vectors in collection: {stats['total_vectors']}")
    assert stats["total_vectors"] == 3, (
        f"Expected 3 vectors, got {stats['total_vectors']}"
    )
    print("  ✓ Vector count correct\n")

    # Test 2: Query by lecture
    print("Test 2: Querying vectors with filter...")
    from inference.embedding.bge_embedder import BGEEmbedder

    bge = BGEEmbedder()
    test_texts = ["머신러닝에 대해 설명해주세요"]
    query_embedding = (
        bge.embed_chunks(
            [
                Chunk(
                    chunk_id="query",
                    text=test_texts[0],
                    start_time=0.0,
                    end_time=1.0,
                    token_count=10,
                    course_id=1,
                    lecture_id=1,
                )
            ]
        )
        .results[0]
        .embedding
    )

    results = vectorstore.query_by_lecture(
        query_embedding=query_embedding,
        filter=QueryFilter(course_id=1, lecture_id=1),
        top_k=2,
    )

    print(f"  Retrieved {len(results['documents'][0])} results")
    for i, (doc, metadata) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], strict=True)
    ):
        print(f"  [{i + 1}] chunk_id={metadata['chunk_id']}, text={doc[:50]}...")

    assert len(results["documents"][0]) == 2, "Expected 2 results"
    print("  ✓ Query successful\n")

    # Test 3: Idempotency - re-index same lecture
    print("Test 3: Testing idempotency (re-indexing same lecture)...")
    new_chunks = [
        Chunk(
            chunk_id="test_chunk_004",
            text="새로운 강의 내용입니다. This is new lecture content.",
            start_time=0.0,
            end_time=3.0,
            token_count=12,
            course_id=1,
            lecture_id=1,
        ),
    ]

    embedder.embed_and_index(course_id=1, lecture_id=1, chunks=new_chunks)

    stats = vectorstore.get_collection_stats()
    print(f"  Total vectors after re-index: {stats['total_vectors']}")
    assert stats["total_vectors"] == 1, (
        f"Expected 1 vector, got {stats['total_vectors']}"
    )
    print("  ✓ Old vectors deleted, new vectors added\n")

    # Test 4: Multiple lectures
    print("Test 4: Indexing multiple lectures...")
    embedder.embed_and_index(course_id=1, lecture_id=2, chunks=chunks[:2])

    stats = vectorstore.get_collection_stats()
    print(f"  Total vectors: {stats['total_vectors']}")
    assert stats["total_vectors"] == 3, (
        f"Expected 3 vectors, got {stats['total_vectors']}"
    )
    print("  ✓ Multiple lectures indexed\n")

    # Verify filtering
    print("  Verifying lecture_id filtering...")
    results_lec1 = vectorstore.query_by_lecture(
        query_embedding=query_embedding,
        filter=QueryFilter(course_id=1, lecture_id=1),
        top_k=10,
    )
    results_lec2 = vectorstore.query_by_lecture(
        query_embedding=query_embedding,
        filter=QueryFilter(course_id=1, lecture_id=2),
        top_k=10,
    )

    print(f"    Lecture 1 results: {len(results_lec1['documents'][0])}")
    print(f"    Lecture 2 results: {len(results_lec2['documents'][0])}")

    assert len(results_lec1["documents"][0]) == 1, "Expected 1 result for lecture 1"
    assert len(results_lec2["documents"][0]) == 2, "Expected 2 results for lecture 2"
    print("  ✓ Filtering works correctly\n")

    print("=== All Tests Passed! ===")


if __name__ == "__main__":
    test_embedding_and_indexing()
