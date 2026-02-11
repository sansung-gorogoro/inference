from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobSubmissionRequest(BaseModel):
    type: Literal["process_lecture", "transcribe", "generate_quiz"]
    course_id: int = Field(..., gt=0)
    lecture_id: int = Field(..., gt=0)
    audio_path: str = Field(..., min_length=1)
    callback_url: str | None = None
    config: dict[str, object] | None = None


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str


class JobResultData(BaseModel):
    transcript: dict[str, object] | None = None
    chunks: list[dict[str, object]] | None = None
    index: dict[str, object] | None = None
    quiz: dict[str, object] | None = None
    delivered: bool = False
    delivery_http_status: int | None = None
    delivery_error: str | None = None


class JobErrorData(BaseModel):
    code: str
    message: str
    details: dict[str, object] | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    type: str
    course_id: int
    lecture_id: int
    status: str
    stage: str | None
    result: JobResultData | None
    error: JobErrorData | None
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
