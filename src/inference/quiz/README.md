# Quiz Generation Module

OpenAI-based multiple choice question generation with schema validation and retry logic.

## Overview

- **Model**: OpenAI Chat Completions API (configurable via `QUIZ_MODEL` env var)
- **Default Model**: `gpt-4o-mini`
- **Output Format**: Structured JSON with validation
- **Retry Logic**: Max 2 retries for malformed responses
- **Question Format**: 4 options, 1 correct answer, optional explanation

## Quick Start

```python
from inference.quiz import QuizGenerator
from inference.retrieval import Retriever

# Initialize generator
generator = QuizGenerator()
print(f"Using model: {generator.model}")

# Get context from retrieval
retriever = Retriever()
retrieval_result = retriever.retrieve(
    query="Summarize the lecture",
    course_id=1,
    lecture_id=1,
    top_k=10
)

# Generate quiz
quiz = generator.generate(
    retrieval_result=retrieval_result,
    course_id=1,
    lecture_id=1,
    num_questions=10
)

print(f"Generated {len(quiz.questions)} questions")

for i, q in enumerate(quiz.questions, 1):
    print(f"\nQuestion {i}: {q.question}")
    for j, opt in enumerate(q.options):
        marker = "✓" if j == q.correct_index else " "
        print(f"  [{marker}] {opt.text}")
    if q.explanation:
        print(f"  Explanation: {q.explanation}")
```

## API Reference

### `QuizGenerator`

Main quiz generation interface using OpenAI API.

**Constructor:**
```python
QuizGenerator(
    api_key: str | None = None,
    model: str | None = None,
    max_retries: int = 2
)
```

**Parameters:**
- `api_key` (str | None): OpenAI API key (default: read from `OPENAI_API_KEY` env var)
- `model` (str | None): Model name (default: read from `QUIZ_MODEL` env var, fallback to `gpt-4o-mini`)
- `max_retries` (int): Maximum retries for malformed JSON (default: 2)

**Methods:**

#### `generate()`
```python
def generate(
    retrieval_result: RetrievalResult,
    course_id: int,
    lecture_id: int,
    num_questions: int = 10
) -> Quiz
```

Generate quiz from retrieved context.

**Parameters:**
- `retrieval_result` (RetrievalResult): Retrieved context with citations
- `course_id` (int): Course identifier
- `lecture_id` (int): Lecture identifier
- `num_questions` (int): Number of questions to generate (default: 10)

**Returns:**
- `Quiz` with validated questions

**Raises:**
- `ValueError`: If num_questions is invalid or retrieval_result has no citations
- `RuntimeError`: If quiz generation fails after max retries
- `AuthenticationError`: If OpenAI API key is invalid
- `RateLimitError`: If OpenAI API rate limit is exceeded

**Properties:**
- `model: str` - Get model name

### Models

#### `MCQOption`

Multiple choice option.

**Attributes:**
- `text: str` - Option text content

#### `Question`

Multiple choice question with 4 options.

**Attributes:**
- `question: str` - Question text
- `options: list[MCQOption]` - List of 4 MCQ options
- `correct_index: int` - Index of correct option (0-3)
- `explanation: str | None` - Optional explanation for the correct answer

**Validation:**
- Question text cannot be empty
- Must have exactly 4 options
- `correct_index` must be 0-3
- All options must be non-empty

#### `Quiz`

Quiz with multiple choice questions.

**Attributes:**
- `questions: list[Question]` - List of MCQ questions
- `course_id: int` - Course identifier
- `lecture_id: int` - Lecture identifier

**Validation:**
- Must contain at least one question
- `course_id` must be > 0
- `lecture_id` must be > 0

## Prompt Engineering

### System Prompt

Defines the quiz generator's role and requirements:
- Expert quiz generator for educational content
- MCQs test understanding, not memorization
- Plausible and non-trivial options
- Optional explanations for correct answers

### User Prompt

Includes:
- Retrieved context with chunk IDs and timestamps
- Number of questions requested
- Reference to output format in system prompt

### Example Output Format

```json
{
  "questions": [
    {
      "question": "What is machine learning?",
      "options": [
        "A subset of AI",
        "A programming language",
        "A database system",
        "An operating system"
      ],
      "correct_index": 0,
      "explanation": "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data."
    }
  ]
}
```

## Retry Logic

Handles malformed responses from OpenAI API:

```python
for attempt in range(max_retries + 1):
    try:
        response = client.chat.completions.create(...)
        quiz_dict = json.loads(response.content)
        quiz = _parse_quiz(quiz_dict)
        return quiz
    except (json.JSONDecodeError, ValueError) as e:
        if attempt >= max_retries:
            raise RuntimeError(f"Failed after {max_retries} retries")
```

**Retry Triggers:**
- `json.JSONDecodeError`: Malformed JSON response
- `ValueError`: Invalid quiz structure (wrong number of options, invalid index, etc.)
- `KeyError`: Missing required fields

**Non-Retriable Errors:**
- `AuthenticationError`: Invalid API key (fails immediately)
- `RateLimitError`: Rate limit exceeded (fails immediately)

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required)
- `QUIZ_MODEL`: Model name (default: `gpt-4o-mini`)

### Supported Models

Any OpenAI Chat Completions model:
- `gpt-4o-mini` (default, cost-effective)
- `gpt-4o` (more capable, higher cost)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Error Handling

```python
try:
    quiz = generator.generate(retrieval_result, course_id=1, lecture_id=1)
except ValueError as e:
    print(f"Validation error: {e}")
except AuthenticationError as e:
    print(f"OpenAI authentication failed: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except RuntimeError as e:
    print(f"Quiz generation failed: {e}")
```

## Schema Validation

Validation occurs at two levels:

1. **JSON Parsing**: Ensures valid JSON structure
2. **Model Validation**: Ensures compliance with Question/Quiz constraints

**Common Validation Errors:**
- "Missing 'questions' key in quiz response"
- "Question must have exactly 4 options, got X"
- "correct_index must be 0-3, got X"
- "Option X cannot be empty"

## Context Building

Quiz generator builds context from retrieval results:

```python
def _build_context(retrieval_result: RetrievalResult) -> str:
    context_parts = []
    for i, citation in enumerate(retrieval_result.citations, 1):
        context_parts.append(
            f"[Chunk {i}] ({citation.chunk_id}, {citation.start_time:.1f}s-{citation.end_time:.1f}s):\n{citation.text}"
        )
    return "\n\n".join(context_parts)
```

**Format:**
```
[Chunk 1] (lec1_chunk_001, 0.0s-15.5s):
Machine learning is a subset of artificial intelligence...

[Chunk 2] (lec1_chunk_005, 45.2s-58.7s):
Deep learning uses neural networks with multiple layers...
```

## Performance

- **Generation Time**: ~2-5 seconds for 10 questions (depends on model and context size)
- **Cost**: ~$0.001-0.01 per quiz (depends on model and context size)
- **Token Usage**: ~1000-3000 tokens per quiz (500-1500 input, 500-1500 output)

## Testing

See `scripts/test_pipeline_integration.py` for basic tests:
- Quiz model validation
- QuizGenerator instantiation
- Error handling

**Production Testing:**
Run on Linux with proper OpenAI API key:
```bash
export OPENAI_API_KEY="your-key"
export QUIZ_MODEL="gpt-4o-mini"
.venv/bin/python3 scripts/test_pipeline_integration.py
```

## Dependencies

- `openai>=1.0.0` - OpenAI API client
- `inference.retrieval` - Retrieval module for context

## Phase 1 Limitations

- No Bloom taxonomy classification
- No multi-stage validation pipeline
- No streaming generation
- No difficulty levels
- Fixed temperature (0.7)

## Future Enhancements (Phase 2+)

- [ ] Bloom taxonomy-based question classification
- [ ] Multi-stage validation pipeline for quiz quality
- [ ] Streaming generation for faster UX
- [ ] Question difficulty levels
- [ ] Multi-language support
- [ ] Adaptive question count based on content length
- [ ] Citation references in questions/explanations

## References

- OpenAI Chat Completions API: https://platform.openai.com/docs/guides/chat
- JSON mode: https://platform.openai.com/docs/guides/structured-outputs
