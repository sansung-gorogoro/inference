from typing import TypedDict

from pydantic import BaseModel, Field


class ChunkMetadata(TypedDict):
    course_id: int
    lecture_id: int
    chunk_id: str
    start_time: float
    end_time: float


class LectureChunk(BaseModel):
    text: str = Field(..., min_length=1)
    course_id: int = Field(..., gt=0)
    lecture_id: int = Field(..., gt=0)
    chunk_id: str = Field(..., min_length=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)

    def to_metadata(self) -> ChunkMetadata:
        return {
            "course_id": self.course_id,
            "lecture_id": self.lecture_id,
            "chunk_id": self.chunk_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class QueryFilter(BaseModel):
    course_id: int | None = None
    lecture_id: int | None = None

    def to_where_clause(self) -> dict[str, object] | None:
        if self.lecture_id is not None:
            return {"lecture_id": {"$eq": self.lecture_id}}

        if self.course_id is not None:
            return {"course_id": {"$eq": self.course_id}}

        return None
