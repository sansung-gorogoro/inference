from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from .models import ErrorCode, Job, JobError, JobResult, JobStatus, JobType

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, max_gpu_workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._gpu_semaphore: asyncio.Semaphore = asyncio.Semaphore(max_gpu_workers)
        self._lock: asyncio.Lock = asyncio.Lock()

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

            except Exception as e:
                error = JobError(
                    code=ErrorCode.PIPELINE_ERROR,
                    message=str(e),
                    details={"exception_type": type(e).__name__},
                )
                job.mark_failed(error)

                logger.error(
                    "Job failed",
                    extra={
                        "job_id": job_id,
                        "error_code": error.code.value,
                        "error_message": error.message,
                    },
                    exc_info=True,
                )

    async def _execute_pipeline(self, job: Job) -> JobResult:
        if job.type == JobType.PROCESS_LECTURE:
            result = JobResult()

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
