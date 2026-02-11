from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from ..jobs.manager import JobManager
from ..jobs.models import JobType
from .schemas import (
    HealthResponse,
    JobErrorData,
    JobResultData,
    JobStatusResponse,
    JobSubmissionRequest,
    JobSubmissionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")
job_manager = JobManager(max_gpu_workers=1)


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post(
    "/jobs",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Jobs"],
)
async def submit_job(request: JobSubmissionRequest) -> JobSubmissionResponse:
    try:
        job_type = JobType(request.type)

        job_id = await job_manager.submit_job(
            type=job_type,
            course_id=request.course_id,
            lecture_id=request.lecture_id,
            audio_path=request.audio_path,
            callback_url=request.callback_url,
            config=request.config,
        )

        logger.info(
            "Job submission accepted",
            extra={
                "job_id": job_id,
                "type": request.type,
            },
        )

        return JobSubmissionResponse(job_id=job_id, status="pending")

    except ValueError as e:
        logger.error(
            "Job submission validation failed",
            extra={"error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "Job submission failed",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = await job_manager.get_job(job_id)

    if not job:
        logger.warning("Job not found", extra={"job_id": job_id})
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    result_data = None
    if job.result:
        result_data = JobResultData(
            transcript=job.result.transcript,
            chunks=job.result.chunks,
            index=job.result.index,
            quiz=job.result.quiz,
            delivered=job.result.delivered,
            delivery_http_status=job.result.delivery_http_status,
            delivery_error=job.result.delivery_error,
        )

    error_data = None
    if job.error:
        error_data = JobErrorData(
            code=job.error.code.value,
            message=job.error.message,
            details=job.error.details,
        )

    return JobStatusResponse(
        job_id=job.job_id,
        type=job.type.value,
        course_id=job.course_id,
        lecture_id=job.lecture_id,
        status=job.status.value,
        stage=job.stage,
        result=result_data,
        error=error_data,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )
