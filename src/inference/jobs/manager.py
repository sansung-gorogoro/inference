from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from openai import AuthenticationError, RateLimitError

from ..pipeline.orchestrator import PipelineOrchestrator
from ..stt.exceptions import (
    APIAuthError,
    APIRateLimitError,
    DecodeError,
    OOMError,
)
from .models import ErrorCode, Job, JobError, JobResult, JobStatus, JobType

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, max_gpu_workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_gpu_workers)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._pipeline: PipelineOrchestrator = PipelineOrchestrator()

    async def submit_job(
        self,
        type: JobType,
        course_id: int,
        lecture_id: int,
        audio_path: str,
        callback_url: str | None = None,
        config: dict[str, object] | None = None,
    ) -> str:
        job_id = str(uuid4())

        if type == JobType.PROCESS_LECTURE and not callback_url:
            raise ValueError("callback_url is required for process_lecture job type")

        job = Job(
            job_id=job_id,
            type=type,
            course_id=course_id,
            lecture_id=lecture_id,
            audio_path=audio_path,
            callback_url=callback_url,
            config=config or {},
            status=JobStatus.PENDING,
            stage=None,
            result=None,
            error=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        async with self._lock:
            self._jobs[job_id] = job

        logger.info(
            "Job submitted",
            extra={
                "job_id": job_id,
                "type": type.value,
                "course_id": course_id,
                "lecture_id": lecture_id,
            },
        )

        _ = asyncio.create_task(self._process_job(job_id))

        return job_id

    async def get_job(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def _process_job(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if not job:
            logger.error("Job not found", extra={"job_id": job_id})
            return

        async with self._gpu_semaphore:
            try:
                logger.info(
                    "Job processing started",
                    extra={
                        "job_id": job_id,
                        "type": job.type.value,
                    },
                )

                job.transition_to(JobStatus.PROCESSING, stage="started")

                result = await self._execute_pipeline(job)

                job.mark_completed(result)

                logger.info(
                    "Job completed successfully",
                    extra={
                        "job_id": job_id,
                        "delivered": result.delivered,
                    },
                )

            except (DecodeError, OOMError, APIAuthError, APIRateLimitError) as e:
                # STT-specific errors
                error = JobError(
                    code=e.code,
                    message=e.message,
                    details=e.details,
                )
                job.mark_failed(error)

                logger.error(
                    "Job failed (STT error)",
                    extra={
                        "job_id": job_id,
                        "error_code": error.code.value,
                        "error_message": error.message,
                    },
                    exc_info=True,
                )

            except AuthenticationError as e:
                # OpenAI quiz API authentication error
                error = JobError(
                    code=ErrorCode.API_AUTH_ERROR,
                    message=str(e),
                    details={"service": "openai_quiz"},
                )
                job.mark_failed(error)

                logger.error(
                    "Job failed (OpenAI authentication error)",
                    extra={
                        "job_id": job_id,
                        "error_code": error.code.value,
                        "error_message": error.message,
                    },
                    exc_info=True,
                )

            except RateLimitError as e:
                # OpenAI quiz API rate limit error
                error = JobError(
                    code=ErrorCode.API_RATE_LIMIT,
                    message=str(e),
                    details={"service": "openai_quiz"},
                )
                job.mark_failed(error)

                logger.error(
                    "Job failed (OpenAI rate limit)",
                    extra={
                        "job_id": job_id,
                        "error_code": error.code.value,
                        "error_message": error.message,
                    },
                    exc_info=True,
                )

            except Exception as e:
                # Generic pipeline error
                error = JobError(
                    code=ErrorCode.PIPELINE_ERROR,
                    message=str(e),
                    details={"exception_type": type(e).__name__},
                )
                job.mark_failed(error)

                logger.error(
                    "Job failed (pipeline error)",
                    extra={
                        "job_id": job_id,
                        "error_code": error.code.value,
                        "error_message": error.message,
                    },
                    exc_info=True,
                )

    async def _execute_pipeline(self, job: Job) -> JobResult:
        if job.type == JobType.PROCESS_LECTURE:
            # Execute pipeline in asyncio loop
            loop = asyncio.get_event_loop()

            # Extract config parameters with type safety
            num_questions_val = job.config.get("num_questions", 10)
            num_questions = (
                int(num_questions_val)
                if isinstance(num_questions_val, (int, str))
                else 10
            )

            retrieval_top_k_val = job.config.get("retrieval_top_k", 10)
            retrieval_top_k = (
                int(retrieval_top_k_val)
                if isinstance(retrieval_top_k_val, (int, str))
                else 10
            )

            stt_mode_val = job.config.get("stt_mode")
            stt_mode = str(stt_mode_val) if stt_mode_val is not None else None

            # Run synchronous pipeline in executor
            quiz = await loop.run_in_executor(
                None,
                self._pipeline.process_lecture,
                job.audio_path,
                job.course_id,
                job.lecture_id,
                num_questions,
                retrieval_top_k,
                stt_mode,
            )

            # Convert quiz to dict
            quiz_dict: dict[str, object] = {
                "course_id": quiz.course_id,
                "lecture_id": quiz.lecture_id,
                "questions": [
                    {
                        "question": q.question,
                        "options": [opt.text for opt in q.options],
                        "correct_index": q.correct_index,
                        "explanation": q.explanation,
                    }
                    for q in quiz.questions
                ],
            }

            # Create result
            result = JobResult(
                quiz=quiz_dict,
                delivered=False,
                delivery_http_status=None,
                delivery_error=None,
            )

            # Deliver callback if URL provided
            if job.callback_url:
                result = await self._deliver_callback(job, result)

            return result

        elif job.type in {JobType.TRANSCRIBE, JobType.GENERATE_QUIZ}:
            return JobResult()

        else:
            raise ValueError(f"Unsupported job type: {job.type.value}")

    async def _deliver_callback(self, job: Job, result: JobResult) -> JobResult:
        if not job.callback_url:
            return result

        payload = {
            "job_id": job.job_id,
            "type": job.type.value,
            "course_id": job.course_id,
            "lecture_id": job.lecture_id,
            "status": "completed",
            "result": {
                "transcript": result.transcript,
                "chunks": result.chunks,
                "index": result.index,
                "quiz": result.quiz,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(job.callback_url, json=payload)

                logger.info(
                    "Callback delivered",
                    extra={
                        "job_id": job.job_id,
                        "callback_url": job.callback_url,
                        "status_code": response.status_code,
                    },
                )

                return JobResult(
                    transcript=result.transcript,
                    chunks=result.chunks,
                    index=result.index,
                    quiz=result.quiz,
                    delivered=True,
                    delivery_http_status=response.status_code,
                    delivery_error=None,
                )

        except Exception as e:
            logger.error(
                "Callback delivery failed",
                extra={
                    "job_id": job.job_id,
                    "callback_url": job.callback_url,
                    "error": str(e),
                },
                exc_info=True,
            )

            return JobResult(
                transcript=result.transcript,
                chunks=result.chunks,
                index=result.index,
                quiz=result.quiz,
                delivered=False,
                delivery_http_status=None,
                delivery_error=str(e),
            )
