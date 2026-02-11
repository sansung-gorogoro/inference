"""Quiz generation data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MCQOption:
    """Multiple choice option.

    Attributes:
        text: Option text content
    """

    text: str

    def __post_init__(self) -> None:
        """Validate option constraints."""
        if not self.text.strip():
            raise ValueError("Option text cannot be empty")


@dataclass(frozen=True)
class Question:
    """Multiple choice question with 4 options.

    Attributes:
        question: Question text
        options: List of 4 MCQ options
        correct_index: Index of correct option (0-3)
        explanation: Optional explanation for the correct answer
    """

    question: str
    options: list[MCQOption]
    correct_index: int
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Validate question constraints."""
        if not self.question.strip():
            raise ValueError("Question text cannot be empty")

        if len(self.options) != 4:
            raise ValueError(
                f"Question must have exactly 4 options, got {len(self.options)}"
            )

        if self.correct_index < 0 or self.correct_index > 3:
            raise ValueError(f"correct_index must be 0-3, got {self.correct_index}")

        # Validate all options are non-empty
        for i, option in enumerate(self.options):
            if not option.text.strip():
                raise ValueError(f"Option {i} cannot be empty")


@dataclass(frozen=True)
class Quiz:
    """Quiz with multiple choice questions.

    Attributes:
        questions: List of MCQ questions (default: 10)
        course_id: Course identifier
        lecture_id: Lecture identifier
    """

    questions: list[Question]
    course_id: int
    lecture_id: int

    def __post_init__(self) -> None:
        """Validate quiz constraints."""
        if not self.questions:
            raise ValueError("Quiz must contain at least one question")

        if self.course_id <= 0:
            raise ValueError(f"course_id must be > 0, got {self.course_id}")

        if self.lecture_id <= 0:
            raise ValueError(f"lecture_id must be > 0, got {self.lecture_id}")
