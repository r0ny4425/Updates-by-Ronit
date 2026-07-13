"""Public tracing API for structured simulation logging.

The package exports the log level enum, immutable record type, logger classes,
and built-in sinks used by timelines, components, runtime code, and tests.
Tracing is an observational side channel and does not participate in event
ordering.
"""

from .levels import LogLevel
from .logger import NullLogger, SimulationLogger
from .records import SimulationLogRecord
from .sinks import JsonlSink, LogSink, MemorySink, NullSink, TextSink

__all__ = [
    "JsonlSink",
    "LogLevel",
    "LogSink",
    "MemorySink",
    "NullLogger",
    "NullSink",
    "SimulationLogRecord",
    "SimulationLogger",
    "TextSink",
]
