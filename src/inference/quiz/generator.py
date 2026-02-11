"""Quiz generation using OpenAI API."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AuthenticationError, OpenAI, RateLimitError

from ..retrieval.models import RetrievalResult
from .models import MCQOption, Question, Quiz

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Quiz generator using OpenAI Chat Completions API.

    Uses retrieved context to generate multiple choice questions with
    schema validation and retry logic for malformed responses.
    """

    _client: OpenAI
    _model: str
    _max_retries: int

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize quiz generator.

        Args:
            api_key: OpenAI API key (default: read from OPENAI_API_KEY env var)
            model: Model name (default: read from QUIZ_MODEL env var, fallback to gpt-4o-mini)
            max_retries: Maximum retries for malformed JSON (default: 2)

        Raises:
            ValueError: If API key is not provided
        """
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY environment variable"
            )

        self._client = OpenAI(api_key=resolved_api_key)
        self._model = model or os.getenv("QUIZ_MODEL", "gpt-4o-mini")
        self._max_retries = max_retries

        logger.info(
            f"Initialized QuizGenerator with model={self._model}, max_retries={self._max_retries}"
        )

    def generate(
        self,
        retrieval_result: RetrievalResult,
        course_id: int,
        lecture_id: int,
        num_questions: int = 10,
    ) -> Quiz:
        """Generate quiz from retrieved context.

        Args:
            retrieval_result: Retrieved context with citations
            course_id: Course identifier
            lecture_id: Lecture identifier
            num_questions: Number of questions to generate (default: 10)

        Returns:
            Quiz with generated questions

        Raises:
            ValueError: If num_questions is invalid or retrieval_result has no citations
            RuntimeError: If quiz generation fails after max retries
            AuthenticationError: If OpenAI API key is invalid
            RateLimitError: If OpenAI API rate limit is exceeded
        """
        if num_questions <= 0:
            raise ValueError(f"num_questions must be > 0, got {num_questions}")

        if not retrieval_result.citations:
            raise ValueError(
                "Cannot generate quiz from empty retrieval result. Need at least 1 citation."
            )

        logger.info(
            f"Generating quiz for course_id={course_id}, lecture_id={lecture_id}, "
            + f"num_questions={num_questions}, citations={len(retrieval_result.citations)}"
        )

        # Build context from citations
        context = self._build_context(retrieval_result)

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, num_questions)

        # Generate quiz with retries
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(
                    f"Quiz generation attempt {attempt + 1}/{self._max_retries + 1}"
                )

                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )

                content = response.choices[0].message.content

                if not content:
                    raise RuntimeError("OpenAI API returned empty response")

                # Parse and validate JSON
                quiz_dict = json.loads(content)
                quiz = self._parse_quiz(quiz_dict, course_id, lecture_id)

                logger.info(
                    f"Successfully generated quiz with {len(quiz.questions)} questions"
                )

                return quiz

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Malformed JSON response (attempt {attempt + 1}): {e}",
                    extra={"attempt": attempt + 1, "max_retries": self._max_retries},
                )

                if attempt >= self._max_retries:
                    raise RuntimeError(
                        f"Quiz generation failed after {self._max_retries} retries: malformed JSON"
                    ) from e

            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Invalid quiz schema (attempt {attempt + 1}): {e}",
                    extra={"attempt": attempt + 1, "max_retries": self._max_retries},
                )

                if attempt >= self._max_retries:
                    raise RuntimeError(
                        f"Quiz generation failed after {self._max_retries} retries: invalid schema"
                    ) from e

            except (AuthenticationError, RateLimitError):
                # Don't retry on auth/rate limit errors
                raise

            except Exception as e:
                logger.error(f"Unexpected error during quiz generation: {e}")
                raise RuntimeError(f"Quiz generation failed: {e}") from e

        # Should never reach here
        raise RuntimeError("Quiz generation failed unexpectedly")

    def _build_context(self, retrieval_result: RetrievalResult) -> str:
        """Build context string from retrieval result.

        Args:
            retrieval_result: Retrieved context with citations

        Returns:
            Formatted context string with chunk IDs and timestamps
        """
        context_parts: list[str] = []

        for i, citation in enumerate(retrieval_result.citations, start=1):
            context_parts.append(
                f"[Chunk {i}] ({citation.chunk_id}, {citation.start_time:.1f}s-{citation.end_time:.1f}s):\n{citation.text}"
            )

        return "\n\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt for quiz generation.

        Returns:
            System prompt string
        """
        return """당신은 교육 콘텐츠 전문 퀴즈 출제자입니다.

주어진 강의 내용을 바탕으로 4지선다형 객관식 문제(MCQ)를 생성하는 것이 당신의 역할입니다.

출제 기준:
- 각 문제는 반드시 4개의 보기를 가져야 합니다
- 정답은 반드시 1개여야 합니다
- 단순 암기가 아닌 내용 이해를 평가하는 문제를 출제하세요
- 보기는 그럴듯하고 변별력이 있어야 합니다
- "해당 없음", "위의 모두" 같은 보기는 사용하지 마세요
- 각 문제에 정답에 대한 해설을 작성하세요

출력 형식 (JSON):
{
  "questions": [
    {
      "question": "머신러닝의 정의로 가장 적절한 것은?",
      "options": ["인공지능의 한 분야로 데이터로부터 학습하는 알고리즘을 다룬다", "웹 개발에 사용되는 프로그래밍 언어이다", "대규모 데이터를 저장하는 데이터베이스 시스템이다", "컴퓨터 하드웨어를 관리하는 운영체제이다"],
      "correct_index": 0,
      "explanation": "머신러닝은 명시적으로 프로그래밍하지 않아도 데이터로부터 패턴을 학습하여 예측이나 결정을 수행하는 인공지능의 한 분야입니다."
    }
  ]
}

유의사항:
- correct_index는 0부터 시작합니다 (0, 1, 2, 3 중 하나)
- explanation은 반드시 작성해 주세요
- 강의 내용의 다양한 측면을 고르게 다루는 문제를 출제하세요"""

    def _build_user_prompt(self, context: str, num_questions: int) -> str:
        """Build user prompt with context and requirements.

        Args:
            context: Formatted context from retrieval
            num_questions: Number of questions to generate

        Returns:
            User prompt string
        """
        return f"""다음 강의 내용을 바탕으로 객관식 문제 {num_questions}개를 출제해 주세요.

강의 내용:
{context}

위 내용을 바탕으로 정확히 {num_questions}개의 문제를 시스템 프롬프트에 지정된 JSON 형식으로 생성하세요."""

    def _parse_quiz(
        self, quiz_dict: dict[str, Any], course_id: int, lecture_id: int
    ) -> Quiz:
        """Parse quiz dictionary into Quiz model.

        Args:
            quiz_dict: Quiz dictionary from OpenAI API
            course_id: Course identifier
            lecture_id: Lecture identifier

        Returns:
            Validated Quiz object

        Raises:
            ValueError: If quiz structure is invalid
            KeyError: If required keys are missing
        """
        if "questions" not in quiz_dict:
            raise KeyError("Missing 'questions' key in quiz response")

        questions_data = quiz_dict["questions"]

        if not isinstance(questions_data, list):
            raise ValueError("'questions' must be a list")

        questions: list[Question] = []

        for i, q_dict in enumerate(questions_data):
            try:
                question_text = q_dict["question"]
                options_data = q_dict["options"]
                correct_index = q_dict["correct_index"]
                explanation = q_dict.get("explanation")

                if not isinstance(options_data, list):
                    raise ValueError(f"Question {i}: options must be a list")

                if len(options_data) != 4:
                    raise ValueError(
                        f"Question {i}: expected 4 options, got {len(options_data)}"
                    )

                options = [MCQOption(text=str(opt)) for opt in options_data]

                question = Question(
                    question=str(question_text),
                    options=options,
                    correct_index=int(correct_index),
                    explanation=str(explanation) if explanation else None,
                )

                questions.append(question)

            except KeyError as e:
                raise KeyError(f"Question {i}: missing required key {e}") from e

            except ValueError as e:
                raise ValueError(f"Question {i}: {e}") from e

        return Quiz(questions=questions, course_id=course_id, lecture_id=lecture_id)

    @property
    def model(self) -> str:
        """Get model name."""
        return self._model
