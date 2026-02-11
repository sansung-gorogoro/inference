"""
Safe environment variable loader with allowlist strategy.

This module provides a controlled way to load environment variables from dotenv files,
ensuring that only explicitly allowed files are loaded and OS environment variables
always take precedence.

Allowlist:
  - .env (local development overrides)
  - .env.e2e (E2E test defaults)
  - Explicit file via INFERENCE_ENV_FILE

Forbidden:
  - .env.sample (template only; never auto-loaded)

Precedence (highest to lowest):
  1. OS environment (CI/containers)
  2. Explicit file (INFERENCE_ENV_FILE)
  3. Profile file (INFERENCE_ENV_PROFILE -> .env.e2e)
  4. .env (fill missing keys only)
  5. Code defaults
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env() -> None:
    """
    Load environment variables from allowlisted dotenv files.

    Selection logic (in order):
      1. If INFERENCE_ENV_FILE is set, load that explicit file
      2. If INFERENCE_ENV_PROFILE is set (e.g., 'e2e'), load .env.{profile}
      3. Load .env if it exists (only fills missing keys due to override=False)

    All loads use override=False to ensure OS environment variables always win.

    Raises:
        FileNotFoundError: If INFERENCE_ENV_FILE is set but file does not exist.

    Side Effects:
        - Logs INFO level message for each file loaded (file name only)
        - Modifies os.environ with loaded variables (respecting override=False)
    """
    # Compute repo root: src/inference/env.py -> src/inference -> src -> repo root
    repo_root = Path(__file__).resolve().parents[2]

    files_loaded = []

    # 1. Check for explicit file override
    explicit_file = os.getenv("INFERENCE_ENV_FILE")
    if explicit_file:
        explicit_path = Path(explicit_file)
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"INFERENCE_ENV_FILE points to non-existent file: {explicit_file}"
            )
        load_dotenv(explicit_path, override=False)
        files_loaded.append(explicit_path.name)
        logger.info(f"Loaded environment from: {explicit_path.name}")
        return  # Explicit file is the only source when set

    # 2. Check for profile-based file
    profile = os.getenv("INFERENCE_ENV_PROFILE")
    if profile:
        profile_file = repo_root / f".env.{profile}"
        if profile_file.exists():
            load_dotenv(profile_file, override=False)
            files_loaded.append(profile_file.name)
            logger.info(f"Loaded environment from: {profile_file.name}")

    # 3. Load .env as fallback (only fills missing keys)
    default_env = repo_root / ".env"
    if default_env.exists():
        load_dotenv(default_env, override=False)
        files_loaded.append(default_env.name)
        logger.info(f"Loaded environment from: {default_env.name}")

    # Log summary if nothing was loaded
    if not files_loaded:
        logger.info("No environment files loaded (none found in allowlist)")
