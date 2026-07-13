"""Public import surface for the deterministic event engine.

The engine namespace exposes the generic simulation primitives used to build
timeline-driven components: events, targets, queues, execution summaries,
statistics, and deterministic RNG streams. Protocol and device behavior lives
outside this package.
"""

from __future__ import annotations

from .component import Component
from .event import Event
from .event_ordering import event_ordering_key
from .event_queue import EventQueue
from .execution_summary import ExecutionSummary
from .rng_manager import DeterministicRNG, RNGManager
from .timeline import Timeline
from .timeline_statistics import TimelineStatistics

__all__ = [
    "Component",
    "DeterministicRNG",
    "Event",
    "EventQueue",
    "ExecutionSummary",
    "RNGManager",
    "Timeline",
    "TimelineStatistics",
    "event_ordering_key",
]
