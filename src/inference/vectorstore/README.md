# Vector Store Module

ChromaDB-based vector storage for lecture transcript chunks with metadata filtering.

## Quick Start

```python
from inference.vectorstore import ChromaVectorStore, LectureChunk, QueryFilter

store = ChromaVectorStore()

chunks = [
    LectureChunk(
        text="Introduction to machine learning concepts.",
        course_id=101,
        lecture_id=1,
        chunk_id="lec1_chunk_001",
        start_time=0.0,
        end_time=5.2,
    )
]

embeddings = [[0.1, 0.2, 0.3, ...]]

store.upsert_chunks(course_id=101, lecture_id=1, chunks=chunks, embeddings=embeddings)

results = store.query_by_lecture(
    query_embedding=[0.15, 0.25, 0.35, ...],
    filter=QueryFilter(course_id=101, lecture_id=1),
    top_k=5,
)
```

## Configuration

Set environment variable `CHROMA_PERSIST_PATH` to customize storage location (default: `.chroma_data/`).

## API Reference

### `ChromaVectorStore`

**Methods:**
- `create_or_get_collection()` - Get collection (idempotent)
- `upsert_chunks(course_id, lecture_id, chunks, embeddings)` - Add/update vectors
- `query_by_lecture(query_embedding, filter, top_k)` - Filtered search
- `delete_lecture_chunks(course_id, lecture_id)` - Remove lecture vectors
- `get_collection_stats()` - Collection metadata

### `LectureChunk`

Pydantic model for chunk validation:
- `text: str` - Chunk content
- `course_id: int` - Course ID (> 0)
- `lecture_id: int` - Lecture ID (> 0)
- `chunk_id: str` - Unique chunk identifier
- `start_time: float` - Start timestamp (>= 0)
- `end_time: float` - End timestamp (> 0)

### `QueryFilter`

Filter builder for retrieval:
- `course_id: int | None` - Filter by course
- `lecture_id: int | None` - Filter by lecture

## Deployment Notes

ChromaDB requires Linux x86_64 or ARM64 for full compatibility. Development on macOS x86_64 may encounter dependency issues with `onnxruntime`.

## Collection Schema

**Collection:** `lecture_chunks`

**Metadata:**
```python
{
    "course_id": int,
    "lecture_id": int,
    "chunk_id": str,
    "start_time": float,
    "end_time": float
}
```

**Document:** Full chunk text (searchable)

## Filter Examples

```python
QueryFilter(course_id=101)
QueryFilter(course_id=101, lecture_id=1)
QueryFilter(lecture_id=1)
```
