import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.vectorstore import ChromaVectorStore, LectureChunk, QueryFilter


def main() -> None:
    print("Initializing ChromaDB vector store...")
    store = ChromaVectorStore(persist_path=".test_chroma_data")

    print(f"Creating collection: {store.create_or_get_collection().name}")

    test_chunks = [
        LectureChunk(
            text="Machine learning is a subset of artificial intelligence.",
            course_id=101,
            lecture_id=1,
            chunk_id="lec1_chunk_001",
            start_time=0.0,
            end_time=5.2,
        ),
        LectureChunk(
            text="Deep learning uses neural networks with multiple layers.",
            course_id=101,
            lecture_id=1,
            chunk_id="lec1_chunk_002",
            start_time=5.2,
            end_time=10.5,
        ),
        LectureChunk(
            text="Supervised learning requires labeled training data.",
            course_id=101,
            lecture_id=2,
            chunk_id="lec2_chunk_001",
            start_time=0.0,
            end_time=4.8,
        ),
    ]

    test_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
    ]

    print("\nUpserting chunks...")
    store.upsert_chunks(
        course_id=101,
        lecture_id=1,
        chunks=test_chunks[:2],
        embeddings=test_embeddings[:2],
    )

    store.upsert_chunks(
        course_id=101,
        lecture_id=2,
        chunks=test_chunks[2:],
        embeddings=test_embeddings[2:],
    )

    print("\nQuerying with filter (course_id=101, lecture_id=1)...")
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]

    results = store.query_by_lecture(
        query_embedding=query_embedding,
        filter=QueryFilter(course_id=101, lecture_id=1),
        top_k=2,
    )

    print(f"Found {len(results['documents'][0])} results:")
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  - {metadata['chunk_id']}: {doc[:50]}...")

    print("\nCollection stats:")
    print(store.get_collection_stats())

    print("\nDeleting lecture 1 chunks...")
    store.delete_lecture_chunks(course_id=101, lecture_id=1)

    print("\nFinal collection stats:")
    print(store.get_collection_stats())

    print("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
