"""Environment variables for Starlings configuration."""

from __future__ import annotations

import os


def _parse_bool_env(value: str) -> bool:
    """Parse boolean-like environment variable."""
    return value.lower() in ("1", "true", "on", "yes")


DEBUG_ENABLED: bool = _parse_bool_env(os.getenv("STARLINGS_DEBUG", ""))
"""Enable debug output and detailed logging for troubleshooting."""
