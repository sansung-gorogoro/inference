"""Pipeline orchestrator for end-to-end lecture processing."""

from __future__ import annotations

import logging

from openai import AuthenticationError, RateLimitError

from ..chunking.chunker import chunk_transcript
from ..embedding.embedder import Embedder
from ..quiz.generator import QuizGenerator
from ..quiz.models import Quiz
from ..retrieval.retriever import Retriever
from ..stt.exceptions import (
    APIAuthError,
    APIRateLimitError,
    DecodeError,
    OOMError,
)
from ..stt.factory import create_stt_engine

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates full lecture processing pipeline.

    Pipeline flow:
    1. STT: audio → transcript
    2. Chunking: transcript → chunks
    3. Embedding: chunks → embeddings
    4. Indexing: embeddings → ChromaDB
    5. Retrieval: query → context
    6. Quiz: context → MCQ questions
    """

    _embedder: Embedder
    _retriever: Retriever
    _quiz_generator: QuizGenerator

    def __init__(
        self,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
        quiz_generator: QuizGenerator | None = None,
    ) -> None:
        """Initialize pipeline orchestrator.

        Args:
            embedder: Embedder instance (creates new if None)
            retriever: Retriever instance (creates new if None)
            quiz_generator: QuizGenerator instance (creates new if None)
        """
        self._embedder = embedder or Embedder()
        self._retriever = retriever or Retriever()
        self._quiz_generator = quiz_generator or QuizGenerator()

        logger.info(
            f"Initialized PipelineOrchestrator with embedder={self._embedder.model_name}, "
            + f"retriever={self._retriever.model_name}, quiz_model={self._quiz_generator.model}"
        )

    def process_lecture(
        self,
        audio_path: str,
        course_id: int,
        lecture_id: int,
        num_questions: int = 10,
        retrieval_top_k: int = 10,
        stt_mode: str | None = None,
    ) -> Quiz:
        """Process lecture audio to generate quiz.

        Args:
            audio_path: Path to audio file
            course_id: Course identifier
            lecture_id: Lecture identifier
            num_questions: Number of quiz questions to generate (default: 10)
            retrieval_top_k: Number of chunks to retrieve for context (default: 10)
            stt_mode: STT mode override ("local" or "api", default: from env)

        Returns:
            Generated Quiz object

        Raises:
            DecodeError: If audio file cannot be decoded
            OOMError: If STT runs out of memory
            APIAuthError: If STT API authentication fails
            APIRateLimitError: If STT API rate limit is exceeded
            AuthenticationError: If OpenAI quiz API authentication fails
            RateLimitError: If OpenAI quiz API rate limit is exceeded
            RuntimeError: For other pipeline errors
        """
        logger.info(
            f"Starting pipeline for course_id={course_id}, lecture_id={lecture_id}, "
            + f"audio_path={audio_path}, num_questions={num_questions}"
        )

        # Stage 1: Speech-to-Text
        logger.info("[Stage 1/6] Starting STT transcription")

        try:
            stt_engine = create_stt_engine(mode=stt_mode)
            transcript = stt_engine.transcribe(audio_path)

            logger.info(
                f"[Stage 1/6] STT completed: {len(transcript.segments)} segments, "
                + f"duration={transcript.duration:.1f}s, language={transcript.language}"
            )

        except DecodeError:
            logger.error("[Stage 1/6] STT failed: audio decode error")
            raise

        except OOMError:
            logger.error("[Stage 1/6] STT failed: out of memory")
            raise

        except APIAuthError:
            logger.error("[Stage 1/6] STT failed: authentication error")
            raise

        except APIRateLimitError:
            logger.error("[Stage 1/6] STT failed: rate limit exceeded")
            raise

        except Exception as e:
            logger.error(f"[Stage 1/6] STT failed: {e}", exc_info=True)
            raise RuntimeError(f"STT stage failed: {e}") from e

        # Stage 2: Chunking
        logger.info("[Stage 2/6] Starting chunking")

        try:
            chunks = chunk_transcript(
                transcript=transcript,
                course_id=course_id,
                lecture_id=lecture_id,
            )

            logger.info(
                f"[Stage 2/6] Chunking completed: {len(chunks)} chunks, "
                + f"avg_tokens={sum(c.token_count for c in chunks) / len(chunks):.1f}"
            )

        except Exception as e:
            logger.error(f"[Stage 2/6] Chunking failed: {e}", exc_info=True)
            raise RuntimeError(f"Chunking stage failed: {e}") from e

        # Stage 3: Embedding
        logger.info("[Stage 3/6] Starting embedding")

        try:
            self._embedder.embed_and_index(
                course_id=course_id,
                lecture_id=lecture_id,
                chunks=chunks,
            )

            logger.info(
                f"[Stage 3/6] Embedding completed: {len(chunks)} chunks indexed"
            )

        except Exception as e:
            logger.error(f"[Stage 3/6] Embedding failed: {e}", exc_info=True)
            raise RuntimeError(f"Embedding stage failed: {e}") from e

        # Stage 4: Indexing (already done in embed_and_index)
        logger.info("[Stage 4/6] Indexing completed (idempotent upsert)")

        # Stage 5: Retrieval
        logger.info("[Stage 5/6] Starting retrieval")

        try:
            # Use generic query for full lecture context
            retrieval_result = self._retriever.retrieve(
                query="Summarize the key concepts and topics covered in this lecture",
                course_id=course_id,
                lecture_id=lecture_id,
                top_k=retrieval_top_k,
            )

            logger.info(
                f"[Stage 5/6] Retrieval completed: {retrieval_result.total_retrieved} retrieved, "
                + f"{retrieval_result.total_deduplicated} after deduplication"
            )

        except Exception as e:
            logger.error(f"[Stage 5/6] Retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"Retrieval stage failed: {e}") from e

        # Stage 6: Quiz Generation
        logger.info("[Stage 6/6] Starting quiz generation")

        try:
            quiz = self._quiz_generator.generate(
                retrieval_result=retrieval_result,
                course_id=course_id,
                lecture_id=lecture_id,
                num_questions=num_questions,
            )

            logger.info(
                f"[Stage 6/6] Quiz generation completed: {len(quiz.questions)} questions"
            )

        except AuthenticationError:
            logger.error("[Stage 6/6] Quiz generation failed: authentication error")
            raise

        except RateLimitError:
            logger.error("[Stage 6/6] Quiz generation failed: rate limit exceeded")
            raise

        except Exception as e:
            logger.error(f"[Stage 6/6] Quiz generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Quiz generation stage failed: {e}") from e

        logger.info(
            f"Pipeline completed successfully for course_id={course_id}, lecture_id={lecture_id}"
        )

        return quiz
