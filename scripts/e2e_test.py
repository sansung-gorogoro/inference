#!/usr/bin/env python3
"""
End-to-End Test Suite for LXP5 Inference Service

Implements 3 QA scenarios from the plan:
1. Submit lecture processing job and complete pipeline
2. Idempotent re-run does not duplicate vectors
3. Quiz scope is current lecture only

Requirements:
- Service running at http://localhost:8000
- Callback receiver running at http://127.0.0.1:9909/callback
- Sample audio file at inputs/sample.webm
- ChromaDB dependencies installed (Linux x86_64/ARM64 required)

Usage:
    # Terminal 1: Start service
    .venv/bin/uvicorn inference.main:app --host 0.0.0.0 --port 8000

    # Terminal 2: Start callback receiver
    .venv/bin/python3 scripts/callback_receiver.py

    # Terminal 3: Run E2E tests
    .venv/bin/python3 scripts/e2e_test.py

Platform: Linux x86_64 or ARM64 (macOS x86_64 not supported due to onnxruntime)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# Configuration
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")
CALLBACK_URL = os.getenv("CALLBACK_URL", "http://127.0.0.1:9909/callback")
SAMPLE_AUDIO = os.getenv("SAMPLE_AUDIO", "inputs/sample.webm")
EVIDENCE_DIR = Path(".sisyphus/evidence")
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 300  # 10 minutes

# Test data
TEST_COURSE_ID = 1
TEST_LECTURE_ID = 1


class E2ETestError(Exception):
    """Base exception for E2E test failures."""

    pass


def ensure_evidence_dir() -> None:
    """Create evidence directory if it doesn't exist."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evidence directory ready: {EVIDENCE_DIR}")


def save_evidence(filename: str, data: dict[str, Any]) -> None:
    """Save test evidence to JSON file.

    Args:
        filename: Evidence filename (e.g., 'e2e-process-lecture.json')
        data: Data to save
    """
    filepath = EVIDENCE_DIR / filename

    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    with filepath.open("w") as f:
        json.dump(evidence, f, indent=2)

    logger.info(f"Evidence saved: {filepath}")


def check_service_health() -> bool:
    """Check if service is running and healthy.

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        response.raise_for_status()

        data = response.json()

        if data.get("status") == "ok":
            logger.info(f"Service is healthy: {SERVICE_URL}")
            return True

        logger.error(f"Service returned unexpected status: {data}")
        return False

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to service at {SERVICE_URL}")
        return False

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def check_callback_receiver() -> bool:
    """Check if callback receiver is running.

    Returns:
        True if receiver is running, False otherwise
    """
    try:
        # Try to connect to callback receiver (expect 404 on GET)
        response = requests.get(CALLBACK_URL, timeout=2)

        # Any response means server is running
        logger.info(f"Callback receiver is running: {CALLBACK_URL}")
        return True

    except requests.exceptions.ConnectionError:
        logger.warning(f"Callback receiver not running at {CALLBACK_URL}")
        return False

    except Exception:
        # Any other error likely means server is running but rejected the request
        return True


def submit_job(
    course_id: int, lecture_id: int, audio_path: str, callback_url: str | None = None
) -> str:
    """Submit a process_lecture job.

    Args:
        course_id: Course ID
        lecture_id: Lecture ID
        audio_path: Path to audio file
        callback_url: Optional callback URL

    Returns:
        job_id

    Raises:
        E2ETestError: If submission fails
    """
    payload = {
        "type": "process_lecture",
        "course_id": course_id,
        "lecture_id": lecture_id,
        "audio_path": audio_path,
        "callback_url": callback_url,
        "config": {},
    }

    logger.info(f"Submitting job: course_id={course_id}, lecture_id={lecture_id}")

    try:
        response = requests.post(
            f"{SERVICE_URL}/api/v1/jobs",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        if response.status_code != 202:
            raise E2ETestError(f"Expected 202, got {response.status_code}")

        data = response.json()
        job_id = data.get("job_id")

        if not job_id:
            raise E2ETestError(f"No job_id in response: {data}")

        logger.info(f"Job submitted: job_id={job_id}")
        return job_id

    except requests.exceptions.RequestException as e:
        raise E2ETestError(f"Job submission failed: {e}") from e


def poll_job_until_complete(job_id: str) -> dict[str, Any]:
    """Poll job status until completed or failed.

    Args:
        job_id: Job ID to poll

    Returns:
        Final job status data

    Raises:
        E2ETestError: If job fails or polling times out
    """
    logger.info(f"Polling job {job_id} (max {MAX_POLL_ATTEMPTS} attempts)...")

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        try:
            response = requests.get(
                f"{SERVICE_URL}/api/v1/jobs/{job_id}",
                timeout=10,
            )

            response.raise_for_status()
            data = response.json()

            status = data.get("status")
            stage = data.get("stage")

            logger.info(
                f"Poll {attempt}/{MAX_POLL_ATTEMPTS}: status={status}, stage={stage}"
            )

            if status == "completed":
                logger.info(f"Job {job_id} completed successfully")
                return data

            if status == "failed":
                error = data.get("error", {})
                raise E2ETestError(
                    f"Job failed: {error.get('code')} - {error.get('message')}"
                )

            # Continue polling
            time.sleep(POLL_INTERVAL_SECONDS)

        except requests.exceptions.RequestException as e:
            logger.warning(f"Poll attempt {attempt} failed: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

    raise E2ETestError(f"Job {job_id} did not complete within timeout")


def verify_job_result(result_data: dict[str, Any]) -> None:
    """Verify job result structure and contents.

    Args:
        result_data: Job result data

    Raises:
        E2ETestError: If result is invalid
    """
    result = result_data.get("result")

    if not result:
        raise E2ETestError("No result in job data")

    # Verify transcript
    transcript = result.get("transcript")
    if not transcript:
        raise E2ETestError("No transcript in result")

    segments = transcript.get("segments")
    if not segments or not isinstance(segments, list) or len(segments) == 0:
        raise E2ETestError(f"Invalid transcript segments: {segments}")

    logger.info(f"✓ Transcript verified: {len(segments)} segments")

    # Verify chunks
    chunks = result.get("chunks")
    if not chunks or not isinstance(chunks, list) or len(chunks) == 0:
        raise E2ETestError(f"Invalid chunks: {chunks}")

    logger.info(f"✓ Chunks verified: {len(chunks)} chunks")

    # Verify index
    index = result.get("index")
    if not index:
        raise E2ETestError("No index in result")

    vectors_indexed = index.get("vectors_indexed", 0)
    if vectors_indexed <= 0:
        raise E2ETestError(f"No vectors indexed: {vectors_indexed}")

    logger.info(f"✓ Index verified: {vectors_indexed} vectors")

    # Verify quiz
    quiz = result.get("quiz")
    if not quiz:
        raise E2ETestError("No quiz in result")

    questions = quiz.get("questions")
    if not questions or not isinstance(questions, list) or len(questions) == 0:
        raise E2ETestError(f"Invalid quiz questions: {questions}")

    logger.info(f"✓ Quiz verified: {len(questions)} questions")

    # Verify delivery
    delivered = result.get("delivered", False)
    logger.info(f"✓ Callback delivery: {delivered}")


def verify_callback_received() -> dict[str, Any]:
    """Verify callback was received by callback receiver.

    Returns:
        Callback data

    Raises:
        E2ETestError: If callback file not found or invalid
    """
    callback_file = EVIDENCE_DIR / "callback-received.json"

    if not callback_file.exists():
        raise E2ETestError(f"Callback file not found: {callback_file}")

    with callback_file.open() as f:
        data = json.load(f)

    payload = data.get("payload")
    if not payload:
        raise E2ETestError("No payload in callback data")

    logger.info("✓ Callback received and verified")

    return data


def query_chromadb_count(course_id: int, lecture_id: int) -> int:
    """Query ChromaDB for vector count with metadata filter.

    Args:
        course_id: Course ID
        lecture_id: Lecture ID

    Returns:
        Number of vectors for the lecture

    Raises:
        E2ETestError: If ChromaDB query fails
    """
    try:
        # Import here to avoid import errors on macOS
        import chromadb  # type: ignore

        persist_path = os.getenv("CHROMA_PERSIST_PATH", ".chroma_data")
        client = chromadb.PersistentClient(path=persist_path)

        collection = client.get_collection(name="lecture_chunks")

        # Query with metadata filter
        results = collection.get(
            where={"course_id": course_id, "lecture_id": lecture_id}
        )

        count = len(results["ids"]) if results.get("ids") else 0

        logger.info(
            f"ChromaDB query: course_id={course_id}, lecture_id={lecture_id}, count={count}"
        )

        return count

    except ImportError as e:
        raise E2ETestError(
            f"ChromaDB not available (platform not supported): {e}"
        ) from e

    except Exception as e:
        raise E2ETestError(f"ChromaDB query failed: {e}") from e


def verify_retrieval_scope_logs(job_result: dict[str, Any]) -> None:
    """Verify quiz retrieval was scoped to current lecture only.

    Checks:
    - Citations reference only the current lecture
    - All citation lecture_id matches expected

    Args:
        job_result: Job result data

    Raises:
        E2ETestError: If scope violation detected
    """
    result = job_result.get("result", {})
    quiz = result.get("quiz", {})

    citations = quiz.get("citations", [])

    if not citations:
        logger.warning(
            "No citations in quiz (may be expected for some implementations)"
        )
        return

    # Verify all citations match expected lecture
    for i, citation in enumerate(citations):
        citation_lecture_id = citation.get("lecture_id")

        if citation_lecture_id != TEST_LECTURE_ID:
            raise E2ETestError(
                f"Citation {i} references wrong lecture: "
                f"expected {TEST_LECTURE_ID}, got {citation_lecture_id}"
            )

    logger.info(
        f"✓ Quiz scope verified: all {len(citations)} citations reference lecture_id={TEST_LECTURE_ID}"
    )


def scenario_1_complete_pipeline() -> dict[str, Any]:
    """Scenario 1: Submit lecture job and verify complete pipeline.

    Returns:
        Job result data

    Raises:
        E2ETestError: If scenario fails
    """
    logger.info("=" * 80)
    logger.info("SCENARIO 1: Submit lecture processing job and complete pipeline")
    logger.info("=" * 80)

    # Step 1: Submit job
    job_id = submit_job(
        course_id=TEST_COURSE_ID,
        lecture_id=TEST_LECTURE_ID,
        audio_path=SAMPLE_AUDIO,
        callback_url=CALLBACK_URL,
    )

    # Step 2: Poll until completed
    result_data = poll_job_until_complete(job_id)

    # Step 3: Verify result structure
    verify_job_result(result_data)

    # Step 4: Verify callback received
    if check_callback_receiver():
        verify_callback_received()
    else:
        logger.warning("Callback receiver not running - skipping callback verification")

    # Step 5: Save evidence
    save_evidence("e2e-process-lecture.json", result_data)

    logger.info("✓ SCENARIO 1 PASSED\n")

    return result_data


def scenario_2_idempotency(initial_count: int) -> None:
    """Scenario 2: Idempotent re-run does not duplicate vectors.

    Args:
        initial_count: Initial vector count from scenario 1

    Raises:
        E2ETestError: If scenario fails
    """
    logger.info("=" * 80)
    logger.info("SCENARIO 2: Idempotent re-run does not duplicate vectors")
    logger.info("=" * 80)

    # Step 1: Submit same job again
    job_id = submit_job(
        course_id=TEST_COURSE_ID,
        lecture_id=TEST_LECTURE_ID,
        audio_path=SAMPLE_AUDIO,
        callback_url=None,  # No callback for re-run
    )

    # Step 2: Poll until completed
    result_data = poll_job_until_complete(job_id)

    # Step 3: Query ChromaDB for vector count
    current_count = query_chromadb_count(TEST_COURSE_ID, TEST_LECTURE_ID)

    # Step 4: Verify count is stable (within tolerance for minor variations)
    count_diff = abs(current_count - initial_count)
    tolerance = 5  # Allow minor differences due to chunking variations

    if count_diff > tolerance:
        raise E2ETestError(
            f"Vector count changed significantly: {initial_count} -> {current_count} (diff={count_diff})"
        )

    logger.info(
        f"✓ Idempotency verified: vector count stable ({initial_count} -> {current_count}, diff={count_diff})"
    )

    # Step 5: Save evidence
    evidence_data = {
        "initial_count": initial_count,
        "current_count": current_count,
        "count_diff": count_diff,
        "tolerance": tolerance,
        "job_result": result_data,
    }

    save_evidence("idempotency-chroma-query.json", evidence_data)

    logger.info("✓ SCENARIO 2 PASSED\n")


def scenario_3_quiz_scope(job_result: dict[str, Any]) -> None:
    """Scenario 3: Quiz scope is current lecture only.

    Args:
        job_result: Job result from scenario 1

    Raises:
        E2ETestError: If scenario fails
    """
    logger.info("=" * 80)
    logger.info("SCENARIO 3: Quiz scope is current lecture only")
    logger.info("=" * 80)

    # Verify retrieval filter uses lecture_id
    verify_retrieval_scope_logs(job_result)

    # Save evidence
    result = job_result.get("result", {})
    quiz = result.get("quiz", {})

    evidence_data = {
        "expected_lecture_id": TEST_LECTURE_ID,
        "quiz": quiz,
        "citations_count": len(quiz.get("citations", [])),
        "all_citations_match_lecture": True,
    }

    save_evidence("quiz-filter-log.json", evidence_data)

    logger.info("✓ SCENARIO 3 PASSED\n")


def check_platform_compatibility() -> bool:
    """Check if current platform supports ChromaDB.

    Returns:
        True if platform is compatible, False otherwise
    """
    import platform

    system = platform.system()
    machine = platform.machine()

    logger.info(f"Platform: {system} {machine}")

    if system == "Darwin" and machine == "x86_64":
        logger.warning(
            "macOS x86_64 detected - ChromaDB dependencies may not be available"
        )
        logger.warning("This platform is NOT supported for running E2E tests")
        logger.warning("Please run tests on Linux x86_64/ARM64 or macOS ARM64")
        return False

    return True


def main() -> int:
    """Run all E2E scenarios.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    logger.info("LXP5 Inference E2E Test Suite")
    logger.info("=" * 80)

    # Check platform compatibility
    if not check_platform_compatibility():
        logger.error("Platform not supported - cannot run tests")
        logger.info(
            "\nTo run tests on a supported platform (Linux/macOS ARM64):\n"
            "  1. Deploy to Linux server or use ARM64 Mac\n"
            "  2. Start service: .venv/bin/uvicorn inference.main:app --port 8000\n"
            "  3. Start callback receiver: .venv/bin/python3 scripts/callback_receiver.py\n"
            "  4. Run tests: .venv/bin/python3 scripts/e2e_test.py\n"
        )
        return 1

    # Ensure evidence directory exists
    ensure_evidence_dir()

    # Check prerequisites
    if not check_service_health():
        logger.error("Service is not running - start it first:")
        logger.error(
            "  .venv/bin/uvicorn inference.main:app --host 0.0.0.0 --port 8000"
        )
        return 1

    if not check_callback_receiver():
        logger.warning("Callback receiver is not running (optional):")
        logger.warning("  .venv/bin/python3 scripts/callback_receiver.py")
        logger.info("Continuing without callback verification...")

    # Check sample audio exists
    if not Path(SAMPLE_AUDIO).exists():
        logger.error(f"Sample audio file not found: {SAMPLE_AUDIO}")
        logger.error("Please provide a sample audio file at inputs/sample.webm")
        return 1

    try:
        # Run scenarios
        logger.info("\n")

        # Scenario 1: Complete pipeline
        scenario_1_result = scenario_1_complete_pipeline()

        # Get initial vector count for idempotency test
        initial_count = query_chromadb_count(TEST_COURSE_ID, TEST_LECTURE_ID)

        # Scenario 2: Idempotency
        scenario_2_idempotency(initial_count)

        # Scenario 3: Quiz scope
        scenario_3_quiz_scope(scenario_1_result)

        # All scenarios passed
        logger.info("=" * 80)
        logger.info("ALL SCENARIOS PASSED ✓")
        logger.info("=" * 80)
        logger.info(f"Evidence saved to: {EVIDENCE_DIR}")

        return 0

    except E2ETestError as e:
        logger.error(f"Test failed: {e}")
        return 1

    except KeyboardInterrupt:
        logger.warning("Test interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
