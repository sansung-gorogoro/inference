#!/usr/bin/env python3
"""
Verify environment loading behavior.

Checks:
  1. load_env() runs without errors
  2. .env.sample is never loaded (forbidden)
  3. Resolved non-secret values are printed for inspection
  4. Exits non-zero if .env.sample would be selected

Usage:
    # Default (loads .env if present):
    .venv/bin/python3 scripts/verify_env_loading.py

    # With E2E profile:
    INFERENCE_ENV_PROFILE=e2e .venv/bin/python3 scripts/verify_env_loading.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on the path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from inference.env import load_env  # noqa: E402


def main() -> int:
    """Run env loading verification and print results."""

    # Guard: reject if someone tries to use .env.sample as a profile
    env_file = os.getenv("INFERENCE_ENV_FILE", "")
    if env_file and Path(env_file).name == ".env.sample":
        print("ERROR: .env.sample must never be loaded (forbidden by allowlist)")
        return 1

    # Load env
    load_env()

    # Non-secret values to inspect
    check_vars = [
        "STT_MODE",
        "WHISPER_MODEL",
        "QUIZ_MODEL",
        "CHROMA_PERSIST_PATH",
        "SERVICE_URL",
        "CALLBACK_URL",
        "SAMPLE_AUDIO",
    ]

    print("=== Environment Loading Verification ===")
    print(f"INFERENCE_ENV_FILE   = {os.getenv('INFERENCE_ENV_FILE', '(not set)')}")
    print(f"INFERENCE_ENV_PROFILE = {os.getenv('INFERENCE_ENV_PROFILE', '(not set)')}")
    print()

    for var in check_vars:
        val = os.getenv(var)
        print(f"  {var} = {val if val is not None else '(not set)'}")

    # Check that OPENAI_API_KEY is present (don't print value)
    api_key = os.getenv("OPENAI_API_KEY")
    print(
        f"  OPENAI_API_KEY = {'(set, {len(api_key)} chars)' if api_key else '(not set)'}"
    )

    print()
    print("OK: env loading completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
