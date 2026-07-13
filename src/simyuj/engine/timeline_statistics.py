"""Snapshot records for timeline execution counters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineStatistics:
    """Immutable snapshot returned by ``Timeline.stats``.

    The record captures the timeline counters at the moment the property is
    read. Reading the snapshot does not schedule, execute, cancel, or reorder
    events.

    Attributes
    ----------
    total_events_scheduled : int
        Number of events successfully scheduled on the timeline. Cancelled
        events remain included in this count.
    total_events_executed : int
        Number of non-cancelled events that have been dispatched to targets.
        Batched events each contribute one count.
    max_queue_size : int
        Largest raw queue size observed after scheduling. Cancelled events may
        contribute until the queue lazily discards them.
    current_time : int
        Current timeline time in simulation ticks.

    Raises
    ------
    ValueError
        If any counter or ``current_time`` is negative.

    Notes
    -----
    The constructor checks non-negative values but does not perform explicit
    type validation.
    """

    total_events_scheduled: int
    """Total number of events successfully scheduled."""
    total_events_executed: int
    """Total number of events that have executed."""
    max_queue_size: int
    """Maximum size the event queue has reached."""
    current_time: int
    """Current logical time (in ticks)."""

    def __post_init__(self):
        """Validate statistics invariants."""
        if self.total_events_scheduled < 0:
            raise ValueError("events_scheduled must be non-negative")
        if self.total_events_executed < 0:
            raise ValueError("events_executed must be non-negative")
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be non-negative")
        if self.current_time < 0:
            raise ValueError("current_time must be non-negative")
