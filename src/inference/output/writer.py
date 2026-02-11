"""Disk output writer for pipeline stage results.

Writes stage output as atomic JSON files under output/{job_id}/
"""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUTS_BASE_DIR = Path("outputs")


def write_stage_output(
    job_id: str,
    stage: str,
    course_id: int,
    lecture_id: int,
    data: dict[str, Any],
) -> Path:
    """Write stage output to disk atomically.

    Creates JSON file at output/{job_id}/{stage}.json with metadata envelope.

    Args:
        job_id: Job identifier
        stage: Stage name (e.g., "stage-01-transcript")
        course_id: Course ID
        lecture_id: Lecture ID
        data: Stage output data

    Returns:
        Path to written file

    Example:
        >>> from inference.output.writer import write_stage_output
        >>> path = write_stage_output(
        ...     job_id="job_abc123",
        ...     stage="stage-01-transcript",
        ...     course_id=1,
        ...     lecture_id=101,
        ...     data={"text": "...", "segments": [...]}
        ... )
        >>> print(path)
        output/job_abc123/stage-01-transcript.json
    """
    # Create job output directory
    job_dir = OUTPUTS_BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Build envelope
    envelope = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "job_id": job_id,
        "course_id": course_id,
        "lecture_id": lecture_id,
        "data": data,
    }

    # Atomic write: temp file then rename
    final_path = job_dir / f"{stage}.json"

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=job_dir,
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        json.dump(envelope, tmp_file, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp_file.name)

    # Atomic rename
    _ = tmp_path.rename(final_path)

    logger.info(
        f"Wrote stage output: {final_path} (course_id={course_id}, lecture_id={lecture_id})"
    )

    return final_path
