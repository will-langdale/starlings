"""Debug utilities for performance instrumentation."""

from __future__ import annotations

import logging
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def get_memory_mb() -> float:
    """Get current process memory usage in MB using psutil."""
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


class DebugTimer:
    """Context manager for timing and memory tracking."""

    def __init__(self, phase_name: str, debug_enabled: bool = False):
        """Initialise the debug timer."""
        self.phase_name = phase_name
        self.debug_enabled = debug_enabled
        self.start_time = 0.0
        self.start_memory = 0.0

    def __enter__(self) -> DebugTimer:
        """Start timing and memory measurement."""
        if self.debug_enabled:
            self.start_time = time.perf_counter()
            self.start_memory = get_memory_mb()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: ARG002
        """End timing and print debug information."""
        if self.debug_enabled:
            end_time = time.perf_counter()
            end_memory = get_memory_mb()
            duration = end_time - self.start_time
            memory_delta = end_memory - self.start_memory

            logger.debug(
                "%s: %.3fs, %+.1fMB",
                self.phase_name,
                duration,
                memory_delta,
            )
