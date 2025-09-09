"""Environment variables for Starlings configuration."""

from __future__ import annotations

import os
from enum import Enum


class SafetyLevel(str, Enum):
    """Memory safety levels for resource-intensive operations."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    UNSAFE = "unsafe"


def _parse_bool_env(value: str) -> bool:
    """Parse boolean-like environment variable."""
    return value.lower() in ("1", "true", "on", "yes")


SAFETY_LEVEL: SafetyLevel = SafetyLevel(
    os.getenv("STARLINGS_SAFETY_LEVEL", SafetyLevel.CONSERVATIVE)
)
"""Current safety level for memory usage in resource-intensive operations.

Controls how much system memory Starlings is allowed to use:
- conservative: Max 50% RAM usage (safest)
- balanced: Max 70% RAM usage  
- performance: Max 85% RAM usage
- unsafe: Max 95% RAM usage (requires STARLINGS_UNSAFE=1)
"""

DEBUG_ENABLED: bool = _parse_bool_env(os.getenv("STARLINGS_DEBUG", ""))
"""Enable debug output and detailed logging for troubleshooting."""

UNSAFE_ENABLED: bool = _parse_bool_env(os.getenv("STARLINGS_UNSAFE", ""))
"""Enable unsafe operations that may risk system stability.

Required to use SafetyLevel.UNSAFE. Only set if you understand the risks
and have sufficient system resources.
"""
