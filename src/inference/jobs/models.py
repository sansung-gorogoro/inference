from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    PROCESS_LECTURE = "process_lecture"
    TRANSCRIBE = "transcribe"
    GENERATE_QUIZ = "generate_quiz"


class ErrorCode(str, Enum):
    DECODE_ERROR = "decode_error"
    OOM = "oom"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTH_ERROR = "api_auth_error"
    PIPELINE_ERROR = "pipeline_error"
    CALLBACK_DELIVERY_FAILED = "callback_delivery_failed"
    INVALID_TRANSITION = "invalid_transition"


@dataclass(frozen=True)
class JobResult:
    transcript: dict[str, object] | None = None
    chunks: list[dict[str, object]] | None = None
    index: dict[str, object] | None = None
    quiz: dict[str, object] | None = None
    delivered: bool = False
    delivery_http_status: int | None = None
    delivery_error: str | None = None


@dataclass(frozen=True)
class JobError:
    code: ErrorCode
    message: str
    details: dict[str, object] | None = None


@dataclass
class Job:
    job_id: str
    type: JobType
    course_id: int
    lecture_id: int
    audio_path: str
    callback_url: str | None
    config: dict[str, object]
    status: JobStatus
    stage: str | None
    result: JobResult | None
    error: JobError | None
    created_at: datetime
    updated_at: datetime

    _VALID_TRANSITIONS: ClassVar[dict[JobStatus, set[JobStatus]]] = {
        JobStatus.PENDING: {JobStatus.PROCESSING, JobStatus.FAILED},
        JobStatus.PROCESSING: {JobStatus.COMPLETED, JobStatus.FAILED},
        JobStatus.COMPLETED: set(),
        JobStatus.FAILED: set(),
    }

    def transition_to(self, new_status: JobStatus, stage: str | None = None) -> None:
        if new_status not in self._VALID_TRANSITIONS[self.status]:
            valid_transitions = ", ".join(
                s.value for s in self._VALID_TRANSITIONS[self.status]
            )
            raise ValueError(
                f"Invalid state transition: {self.status.value} -> {new_status.value}. "
                + f"Valid transitions from {self.status.value}: {valid_transitions}"
            )

        self.status = new_status
        self.stage = stage
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, result: JobResult) -> None:
        self.transition_to(JobStatus.COMPLETED)
        self.result = result

    def mark_failed(self, error: JobError) -> None:
        self.transition_to(JobStatus.FAILED)
        self.error = error

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "type": self.type.value,
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "status": self.status.value,
            "stage": self.stage,
            "result": self._serialize_result(self.result) if self.result else None,
            "error": self._serialize_error(self.error) if self.error else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @staticmethod
    def _serialize_result(result: JobResult) -> dict[str, object]:
        return {
            "transcript": result.transcript,
            "chunks": result.chunks,
            "index": result.index,
            "quiz": result.quiz,
            "delivered": result.delivered,
            "delivery_http_status": result.delivery_http_status,
            "delivery_error": result.delivery_error,
        }

    @staticmethod
    def _serialize_error(error: JobError) -> dict[str, object]:
        return {
            "code": error.code.value,
            "message": error.message,
            "details": error.details,
        }
