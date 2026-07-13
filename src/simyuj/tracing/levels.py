"""Logging levels used by the tracing package.

``LogLevel`` is the small public enum shared by timeline, component, logger,
and sink code. Higher numeric values include more verbose records; ``OFF``
disables logging through :class:`simyuj.tracing.logger.SimulationLogger`.
"""

from __future__ import annotations

from enum import IntEnum


class LogLevel(IntEnum):
    """
    Logging verbosity levels for simulation tracing.

    The enum is ordered from least to most verbose. A logger configured at a
    given level emits records at that level and at all lower numeric levels.
    """

    OFF = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5
