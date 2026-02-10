# Retrieval Module

Lecture-scoped semantic retrieval with automatic deduplication for quiz generation.

## Overview

- **Scope**: Strict lecture-level filtering (no cross-lecture retrieval)
- **Model**: BGE-M3 (via embedding module)
- **Backend**: ChromaDB vector store
- **Deduplication**: Jaccard similarity to remove overlapping chunks (from 20% overlap in chunking)
- **Citations**: Full provenance with chunk_id + timestamps

## Quick Start

```python
from inference.retrieval import Retriever, RetrievalResult, Citation

retriever = Retriever()

result = retriever.retrieve(
    query="What is machine learning?",
    course_id=101,
    lecture_id=1,
    top_k=10
)

print(f"Found {result.total_deduplicated} chunks (removed {result.total_retrieved - result.total_deduplicated} duplicates)")

for citation in result.citations:
    print(f"{citation.chunk_id}: {citation.text[:50]}...")
    print(f"  Time: {citation.start_time:.1f}s - {citation.end_time:.1f}s")
    print(f"  Similarity: {citation.similarity_score:.4f}")
```

## API Reference

### `Retriever`

Main retrieval interface with lecture-scoped querying and deduplication.

**Constructor:**
```python
Retriever(
    vectorstore: ChromaVectorStore | None = None,
    embedder: BGEEmbedder | None = None
)
```

**Methods:**

#### `retrieve()`
```python
def retrieve(
    query: str,
    course_id: int,
    lecture_id: int,
    top_k: int = 10,
    similarity_threshold: float | None = None,
    dedup_threshold: float = 0.8
) -> RetrievalResult
```

Retrieve relevant chunks for a query within a specific lecture.

**Parameters:**
- `query` (str): Natural language query
- `course_id` (int): Course ID to filter by (must be > 0)
- `lecture_id` (int): Lecture ID to filter by (must be > 0)
- `top_k` (int): Maximum number of results to retrieve (default: 10)
- `similarity_threshold` (float | None): Optional distance threshold for filtering results
- `dedup_threshold` (float): Jaccard similarity threshold for deduplication (default: 0.8)

**Returns:**
- `RetrievalResult` with deduplicated citations

**Raises:**
- `ValueError`: If query is empty or IDs are invalid
- `RuntimeError`: If embedding or retrieval fails

**Properties:**
- `model_name: str` - Embedding model name
- `device: str` - Device (cuda/cpu)
- `dimension: int` - Embedding dimension

### `RetrievalResult`

Result of a retrieval query.

**Attributes:**
- `query: str` - Original query text
- `citations: list[Citation]` - List of retrieved citations (after deduplication)
- `total_retrieved: int` - Total chunks retrieved before deduplication
- `total_deduplicated: int` - Number of chunks after deduplication

### `Citation`

Citation for a retrieved chunk with full provenance.

**Attributes:**
- `chunk_id: str` - Unique chunk identifier
- `text: str` - Chunk text content
- `start_time: float` - Start timestamp in seconds
- `end_time: float` - End timestamp in seconds
- `course_id: int` - Course identifier
- `lecture_id: int` - Lecture identifier
- `similarity_score: float` - Similarity score (lower is better for distance-based metrics)

## Architecture

```
Retriever
├── BGEEmbedder (query embedding)
│   └── BAAI/bge-m3 model
└── ChromaVectorStore (vector search)
    └── ChromaDB collection
```

**Flow:**
1. Embed query using BGE-M3
2. Query ChromaDB with lecture filter
3. Filter by similarity threshold (optional)
4. Deduplicate overlapping chunks (Jaccard > 0.8)
5. Return citations with provenance

## Deduplication Strategy

Chunks are created with 20% overlap to preserve context at boundaries. Retrieval deduplicates these overlaps:

**Algorithm:**
1. Receive chunks sorted by similarity (best first)
2. Keep first chunk
3. For each subsequent chunk:
   - Calculate Jaccard similarity with previous kept chunk
   - If similarity < threshold (0.8): keep chunk
   - If similarity >= threshold: discard (duplicate)

**Jaccard Similarity:**
```
J(A, B) = |A ∩ B| / |A ∪ B|
```
Where A and B are sets of words in the chunk texts.

**Example:**
```
Retrieved chunks:
1. "Machine learning is a subset of AI. It focuses on..."  (score: 0.12)
2. "It focuses on training algorithms to learn from..."    (score: 0.15)  [85% overlap - removed]
3. "Deep learning is a specialized form of machine..."     (score: 0.18)

Deduplicated: [1, 3]
```

## Lecture Isolation

**Strict filtering** ensures no cross-lecture contamination:
- Query filter: `course_id == X AND lecture_id == Y`
- ChromaDB enforces filter at query time
- Zero results for non-existent lectures

**Example:**
```python
result = retriever.retrieve(
    query="What is ML?",
    course_id=1,
    lecture_id=99  # Non-existent
)

assert result.total_retrieved == 0
assert len(result.citations) == 0
```

## Advanced Usage

### Similarity Threshold Filtering

Filter results by distance threshold (lower = more similar):

```python
result = retriever.retrieve(
    query="machine learning",
    course_id=1,
    lecture_id=1,
    top_k=20,
    similarity_threshold=0.5  # Keep only scores <= 0.5
)
```

### Strict Deduplication

Use lower threshold for stricter deduplication:

```python
result = retriever.retrieve(
    query="machine learning",
    course_id=1,
    lecture_id=1,
    top_k=10,
    dedup_threshold=0.5  # Remove more duplicates
)
```

### Citation Provenance

Citations include full metadata for quiz generation:

```python
for citation in result.citations:
    # Generate quiz question
    question = generate_question(citation.text)
    
    # Store provenance
    provenance = {
        "chunk_id": citation.chunk_id,
        "course_id": citation.course_id,
        "lecture_id": citation.lecture_id,
        "timestamp": f"{citation.start_time:.1f}s - {citation.end_time:.1f}s",
        "similarity": citation.similarity_score
    }
```

## Performance

- **Query Time**: ~50-100ms (depends on collection size)
- **Deduplication**: O(n) where n = retrieved chunks
- **Memory**: Minimal (embeddings already in ChromaDB)

## Error Handling

```python
try:
    result = retriever.retrieve(query="", course_id=1, lecture_id=1)
except ValueError as e:
    print(f"Validation error: {e}")

try:
    result = retriever.retrieve(query="test", course_id=1, lecture_id=1)
except RuntimeError as e:
    print(f"Retrieval failed: {e}")
```

## Testing

See `scripts/test_retrieval.py` for comprehensive tests:
- Lecture-scoped retrieval
- Deduplication verification
- Citation format validation
- Lecture isolation enforcement
- Similarity threshold filtering
- Error handling

Run with:
```bash
.venv/bin/python3 scripts/test_retrieval.py
```

## Deployment Notes

### Linux + CUDA (Production)
Full functionality with GPU acceleration.

### macOS x86_64 (Development)
Platform compatibility issues with onnxruntime. Use ARM64 Mac or Linux for local testing.

### Model Loading
BGE-M3 model auto-loads from HuggingFace cache (~2GB). First run downloads model.

## Dependencies

- `sentence-transformers>=2.0.0` - Embedding model (via embedding module)
- `chromadb>=0.4.0` - Vector storage (via vectorstore module)
- `numpy` - Array operations (via sentence-transformers)
- `torch>=2.0.0` - PyTorch backend (via sentence-transformers)

## Next Steps (Phase 2)

Future enhancements:
- [ ] Query expansion for better recall
- [ ] Re-ranking with cross-encoder
- [ ] Hybrid search (vector + keyword)
- [ ] Caching for repeated queries
- [ ] Batch retrieval API

## References

- BGE-M3 paper: https://arxiv.org/abs/2402.03216
- ChromaDB docs: https://docs.trychroma.com/
- Jaccard similarity: https://en.wikipedia.org/wiki/Jaccard_index
