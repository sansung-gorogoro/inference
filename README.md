# LXP5 Inference Server

FastAPI-based inference server for lecture processing and quiz generation.

---

## PoC Usage

### Starting the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (default: http://localhost:8000)
uvicorn src.inference.main:app --reload
```

### API Endpoints

#### 1. Full Pipeline (STT → Indexing → Quiz)

Processes audio, creates embeddings, and generates a quiz.

```bash
curl -X POST http://localhost:8000/api/v1/pipelines/full \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": 1,
    "lecture_id": 101,
    "audio_path": "inputs/sample.webm",
    "config": {
      "num_questions": 10,
      "retrieval_top_k": 10
    }
  }'
```

**Response** (HTTP 202):
```json
{
  "job_id": "job_abc123def456",
  "status": "pending"
}
```

**Output Files** (created in `outputs/{job_id}/`):
- `stage-01-transcript.json` - Speech-to-text result
- `stage-02-chunks.json` - Chunked transcript with parameters
- `stage-03-index.json` - Embedding/indexing statistics
- `stage-04-retrieval.json` - Retrieved citations
- `stage-05-quiz.json` - Generated quiz

---

#### 2. Quiz-Only Pipeline (Regenerate Quiz from Existing Index)

Generates a new quiz from an already-indexed lecture without reprocessing audio.

```bash
curl -X POST "http://localhost:8000/api/v1/pipelines/quiz?course_id=1&lecture_id=101"
```

**Response** (HTTP 202):
```json
{
  "job_id": "job_def456ghi789",
  "status": "pending"
}
```

**Output Files** (created in `outputs/{job_id}/`):
- `stage-04-retrieval.json` - Retrieved citations
- `stage-05-quiz.json` - Generated quiz

**Error Response** (HTTP 400 - if index doesn't exist):
```json
{
  "error": {
    "code": "INDEX_NOT_READY",
    "message": "No index found for lecture_id=101. Run full pipeline first."
  }
}
```

---

#### 3. Poll Job Status

Check the status of a submitted job.

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}
```

**Response** (completed job):
```json
{
  "job_id": "job_abc123def456",
  "status": "completed",
  "result": {
    "quiz": {
      "course_id": 1,
      "lecture_id": 101,
      "questions": [
        {
          "question": "What is...",
          "options": ["A", "B", "C", "D"],
          "correct_index": 2,
          "explanation": "..."
        }
      ]
    },
    "delivered": false,
    "delivery_http_status": null,
    "delivery_error": null
  },
  "error": null,
  "created_at": "2026-02-12T10:30:00Z",
  "updated_at": "2026-02-12T10:32:15Z"
}
```

**Status Values**: `pending`, `running`, `completed`, `failed`

---

### Output File Locations

All stage outputs are written to:
```
outputs/{job_id}/stage-{XX}-{name}.json
```

**Example**:
```
outputs/job_abc123def456/
├── stage-01-transcript.json
├── stage-02-chunks.json
├── stage-03-index.json
├── stage-04-retrieval.json
└── stage-05-quiz.json
```

Files are prefixed with stage numbers for natural sort order.

---

### Cross-Platform Compatibility

- Output paths use `pathlib.Path` for Windows/Linux/macOS compatibility
- No hardcoded POSIX-only path separators
- Tested on Windows AMD64, macOS, and Linux

---

### Environment Variables (Optional Tuning)

Chunking parameters can be tuned via environment variables:

```bash
# Silence threshold for chunk boundaries (default: 2.0 seconds)
export CHUNK_SILENCE_THRESHOLD_SECONDS=1.5

# Token limits for semantic chunking (default: 800 max, 160 overlap)
# WARNING: Changing these invalidates existing embeddings/index
export CHUNK_MAX_TOKENS=1000
export CHUNK_OVERLAP_TOKENS=200
```

**Note**: Changing token limits requires re-running the full pipeline to rebuild the index.

---

## Development

### Project Structure

```
src/inference/
├── api/          # FastAPI routes and schemas
├── chunking/     # Transcript chunking logic
├── embedding/    # Embedding and indexing
├── jobs/         # Async job management
├── pipeline/     # Pipeline orchestration
├── quiz/         # Quiz generation
├── retrieval/    # RAG retrieval
├── stt/          # Speech-to-text
└── vectorstore/  # ChromaDB integration
```

### Running Tests

```bash
# Run existing test scripts
python scripts/e2e_test.py
```

---

## License

[Add license information here]
