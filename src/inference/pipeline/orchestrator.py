"""Pipeline orchestrator for end-to-end lecture processing."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import cast

from openai import AuthenticationError, RateLimitError

from ..chunking.chunker import chunk_transcript
from ..embedding.embedder import Embedder
from ..jobs.models import Job, JobStatus, JobType
from ..outputs.writer import write_stage_output
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
    - Segment A: STT → chunking → embed/index
    - Segment B: retrieval → quiz generation

    Stage outputs written at 5 boundaries:
    - stage-01-transcript: After STT
    - stage-02-chunks: After chunking
    - stage-03-index: After embedding/indexing
    - stage-04-retrieval: After retrieval
    - stage-05-quiz: After quiz generation
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

    def _run_segment_a(self, job: Job) -> None:
        """Run Segment A: STT → chunking → embed/index.

        Writes stage outputs:
        - stage-01-transcript: Transcript with segments
        - stage-02-chunks: Chunks with resolved parameters
        - stage-03-index: Indexing statistics

        Args:
            job: Job containing audio_path, course_id, lecture_id, config

        Raises:
            DecodeError: If audio file cannot be decoded
            OOMError: If STT runs out of memory
            APIAuthError: If STT API authentication fails
            APIRateLimitError: If STT API rate limit is exceeded
            RuntimeError: For other stage errors
        """
        # Validate audio_path is provided for Segment A
        if not job.audio_path:
            raise ValueError("audio_path is required for Segment A (STT)")

        # Type narrowing for basedpyright
        assert job.audio_path is not None

        # Extract config (with type narrowing)
        stt_mode_raw = job.config.get("stt_mode")
        stt_mode: str | None = str(stt_mode_raw) if stt_mode_raw is not None else None

        # Stage 1: Speech-to-Text
        logger.info("[Stage 1/6] Starting STT transcription")

        try:
            stt_engine = create_stt_engine(mode=stt_mode)
            transcript = stt_engine.transcribe(job.audio_path)

            logger.info(
                f"[Stage 1/6] STT completed: {len(transcript.segments)} segments, "
                + f"duration={transcript.duration:.1f}s, language={transcript.language}"
            )

            # Write stage-01 output
            full_text = " ".join(seg.text for seg in transcript.segments)
            write_stage_output(
                job_id=job.job_id,
                stage="stage-01-transcript",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                data={
                    "text": full_text,
                    "segments": [asdict(seg) for seg in transcript.segments],
                    "duration": transcript.duration,
                    "language": transcript.language,
                },
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
            chunks, chunking_params = chunk_transcript(
                transcript=transcript,
                course_id=job.course_id,
                lecture_id=job.lecture_id,
            )

            logger.info(
                f"[Stage 2/6] Chunking completed: {len(chunks)} chunks, "
                + f"avg_tokens={sum(c.token_count for c in chunks) / len(chunks):.1f}"
            )

            # Write stage-02 output
            write_stage_output(
                job_id=job.job_id,
                stage="stage-02-chunks",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                data={
                    "chunks": [asdict(chunk) for chunk in chunks],
                    "chunking_params": chunking_params,
                },
            )

        except Exception as e:
            logger.error(f"[Stage 2/6] Chunking failed: {e}", exc_info=True)
            raise RuntimeError(f"Chunking stage failed: {e}") from e

        # Stage 3: Embedding & Indexing
        logger.info("[Stage 3/6] Starting embedding and indexing")

        try:
            self._embedder.embed_and_index(
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                chunks=chunks,
            )

            logger.info(
                f"[Stage 3/6] Embedding completed: {len(chunks)} chunks indexed"
            )

            # Write stage-03 output
            chroma_persist_path = self._embedder._vectorstore._persist_path
            write_stage_output(
                job_id=job.job_id,
                stage="stage-03-index",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                data={
                    "vectors_indexed": len(chunks),
                    "persist_path": str(chroma_persist_path),
                    "lecture_id": job.lecture_id,
                },
            )

        except Exception as e:
            logger.error(f"[Stage 3/6] Embedding failed: {e}", exc_info=True)
            raise RuntimeError(f"Embedding stage failed: {e}") from e

        logger.info("[Stage 4/6] Indexing completed (idempotent upsert)")

    def _run_segment_b(self, job: Job) -> Quiz:
        """Run Segment B: retrieval → quiz generation.

        Writes stage outputs:
        - stage-04-retrieval: Retrieved citations
        - stage-05-quiz: Generated quiz

        Args:
            job: Job containing course_id, lecture_id, config (num_questions, retrieval_top_k)

        Returns:
            Generated Quiz object

        Raises:
            AuthenticationError: If OpenAI quiz API authentication fails
            RateLimitError: If OpenAI quiz API rate limit is exceeded
            RuntimeError: For other stage errors
        """
        # Extract config (with type narrowing and defaults)
        num_questions_raw = job.config.get("num_questions", 10)
        num_questions = (
            int(cast(int, num_questions_raw)) if num_questions_raw is not None else 10
        )

        retrieval_top_k_raw = job.config.get("retrieval_top_k", 10)
        retrieval_top_k = (
            int(cast(int, retrieval_top_k_raw))
            if retrieval_top_k_raw is not None
            else 10
        )

        # Stage 5: Retrieval
        logger.info("[Stage 5/6] Starting retrieval")

        try:
            # Use generic query for full lecture context
            retrieval_result = self._retriever.retrieve(
                query="Summarize the key concepts and topics covered in this lecture",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                top_k=retrieval_top_k,
            )

            logger.info(
                f"[Stage 5/6] Retrieval completed: {retrieval_result.total_retrieved} retrieved, "
                + f"{retrieval_result.total_deduplicated} after deduplication"
            )

            # Write stage-04 output
            write_stage_output(
                job_id=job.job_id,
                stage="stage-04-retrieval",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                data={
                    "citations": [
                        asdict(citation) for citation in retrieval_result.citations
                    ]
                },
            )

        except Exception as e:
            logger.error(f"[Stage 5/6] Retrieval failed: {e}", exc_info=True)
            raise RuntimeError(f"Retrieval stage failed: {e}") from e

        # Stage 6: Quiz Generation
        logger.info("[Stage 6/6] Starting quiz generation")

        try:
            quiz = self._quiz_generator.generate(
                retrieval_result=retrieval_result,
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                num_questions=num_questions,
            )

            logger.info(
                f"[Stage 6/6] Quiz generation completed: {len(quiz.questions)} questions"
            )

            # Write stage-05 output
            write_stage_output(
                job_id=job.job_id,
                stage="stage-05-quiz",
                course_id=job.course_id,
                lecture_id=job.lecture_id,
                data={"quiz": asdict(quiz)},
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

        return quiz

    def run_pipeline(self, job: Job) -> Quiz:
        """Run full pipeline: Segment A → Segment B.

        Entry point for Job-based pipeline execution.

        Args:
            job: Job with all required fields (job_id, audio_path, course_id, lecture_id, config)

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
            f"Starting pipeline for job_id={job.job_id}, course_id={job.course_id}, "
            + f"lecture_id={job.lecture_id}, audio_path={job.audio_path}"
        )

        # Run Segment A: STT → chunking → embed/index
        self._run_segment_a(job)

        # Run Segment B: retrieval → quiz
        quiz = self._run_segment_b(job)

        logger.info(
            f"Pipeline completed successfully for job_id={job.job_id}, "
            + f"course_id={job.course_id}, lecture_id={job.lecture_id}"
        )

        return quiz

    def process_lecture(
        self,
        audio_path: str,
        course_id: int,
        lecture_id: int,
        num_questions: int = 10,
        retrieval_top_k: int = 10,
        stt_mode: str | None = None,
        job_id: str | None = None,
    ) -> Quiz:
        """Process lecture audio to generate quiz (backward compatible wrapper).

        This method creates a Job internally and delegates to run_pipeline().
        For new code, prefer using run_pipeline() directly with a Job object.

        Args:
            audio_path: Path to audio file
            course_id: Course identifier
            lecture_id: Lecture identifier
            num_questions: Number of quiz questions to generate (default: 10)
            retrieval_top_k: Number of chunks to retrieve for context (default: 10)
            stt_mode: STT mode override ("local" or "api", default: from env)
            job_id: Optional job ID (auto-generated if None)

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
        import uuid

        # Generate job_id if not provided
        if job_id is None:
            job_id = f"job_{uuid.uuid4().hex[:12]}"

        # Create Job object
        job = Job(
            job_id=job_id,
            type=JobType.PROCESS_LECTURE,
            course_id=course_id,
            lecture_id=lecture_id,
            audio_path=audio_path,
            callback_url=None,
            config={
                "num_questions": num_questions,
                "retrieval_top_k": retrieval_top_k,
                "stt_mode": stt_mode,
            },
            status=JobStatus.PENDING,
            stage=None,
            result=None,
            error=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Delegate to run_pipeline
        return self.run_pipeline(job)
