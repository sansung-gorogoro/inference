# Embedding Module

BGE-M3 embedding with ChromaDB integration for lecture transcript chunks.

## Overview

- **Model**: BAAI/bge-m3 (multilingual, 1024 dimensions)
- **Backend**: sentence-transformers + PyTorch
- **Batch Size**: ≤64 (for 8GB VRAM constraint)
- **Device**: Auto-selects CUDA if available, otherwise CPU
- **Idempotency**: Delete + upsert for same (course_id, lecture_id)

## Quick Start

```python
from inference.chunking.models import Chunk
from inference.embedding import Embedder

# Initialize embedder
embedder = Embedder()
print(f"Model: {embedder.model_name}, Device: {embedder.device}")

# Create chunks
chunks = [
    Chunk(
        chunk_id="lec1_chunk_001",
        text="머신러닝은 인공지능의 한 분야입니다.",
        start_time=0.0,
        end_time=5.5,
        token_count=15,
        course_id=1,
        lecture_id=1,
    ),
]

# Embed and index (idempotent)
embedder.embed_and_index(course_id=1, lecture_id=1, chunks=chunks)
```

## API Reference

### `Embedder`

Main interface for embedding and indexing operations.

**Methods:**
- `embed_and_index(course_id, lecture_id, chunks)` - Embed chunks and index into ChromaDB
  - Idempotent: deletes existing vectors for same lecture before upserting
  - Raises `ValueError` for empty chunks list
  - Raises `RuntimeError` for embedding or indexing failures

**Properties:**
- `model_name: str` - Embedding model name ("BAAI/bge-m3")
- `device: str` - Device ("cuda" or "cpu")
- `dimension: int` - Embedding dimension (1024)

### `BGEEmbedder`

Low-level BGE-M3 embedding implementation.

**Methods:**
- `embed_chunks(chunks)` - Embed chunks with batch processing
  - Returns `EmbeddingBatch` with results
  - Batch size: 64 (for 8GB VRAM)

### Models

**`EmbeddingResult`**
- `chunk_id: str` - Chunk identifier
- `embedding: list[float]` - Dense vector (1024-dim)

**`EmbeddingBatch`**
- `results: list[EmbeddingResult]` - Embedding results
- `model_name: str` - Model name
- `dimension: int` - Embedding dimension

## Architecture

```
Embedder (high-level orchestrator)
├── BGEEmbedder (sentence-transformers)
│   └── BAAI/bge-m3 model
└── ChromaVectorStore (persistence)
    └── ChromaDB collection
```

## Idempotency

Re-running `embed_and_index()` for the same `(course_id, lecture_id)` will:
1. Delete existing vectors for that lecture
2. Embed new chunks
3. Upsert new vectors

This ensures no duplicate vectors and allows re-processing lectures.

## Performance

- **Batch Size**: 64 chunks per batch (8GB VRAM constraint)
- **Throughput**: ~100-200 chunks/sec on GPU
- **Memory**: ~2GB VRAM for model + batch processing

## Error Handling

All errors are logged with structured messages. Common errors:

- `ValueError`: Empty chunks list or validation failure
- `RuntimeError`: Model loading, embedding, or indexing failure

## Deployment Notes

### Linux + CUDA (Production)
```bash
# Install with GPU support
uv sync

# Model auto-downloads to ~/.cache/huggingface/
# First run will download ~2GB model
```

### macOS x86_64 (Development)
PyTorch 2.10+ dropped x86_64 macOS support. Use ARM64 Mac or Linux for local testing.

### Model Storage

Model files are cached in `~/.cache/huggingface/hub/models--BAAI--bge-m3/`.
First run downloads ~2GB. Subsequent runs load from cache.

## Testing

See `scripts/test_embedding.py` for comprehensive tests:
- Embedding and indexing
- Query and filtering
- Idempotency verification
- Multiple lecture handling

Run with:
```bash
.venv/bin/python3 scripts/test_embedding.py
```

## Dependencies

- `sentence-transformers>=2.0.0` - Embedding model framework
- `torch>=2.0.0` - PyTorch backend (CUDA support)
- `numpy` - Array operations (via sentence-transformers)
- `chromadb>=0.4.0` - Vector storage (via vectorstore module)
