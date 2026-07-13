"""Heap-backed event queue with deterministic ordering and lazy cancellation."""

from __future__ import annotations

import heapq
from typing import List, Tuple

from .event import Event
from .event_ordering import event_ordering_key


class EventQueue:
    """Priority queue for scheduled ``Event`` objects.

    The queue stores heap entries keyed by ``(time, priority, event_id)``.
    Cancelled events are removed lazily when they reach the head of the heap,
    so queue length can include events that will never execute.

    Notes
    -----
    ``EventQueue`` does not assign event identifiers, advance simulation time,
    execute events, or mutate event records. Those responsibilities belong to
    ``Timeline``.
    """

    def __init__(self) -> None:
        # Internal heap entries:
        # (time, priority, event_id, Event)
        self._heap: List[Tuple[int, int, int, Event]] = []

    def _discard_cancelled(self) -> None:
        """Remove cancelled events from the heap head.

        Notes
        -----
        Cancellation is lazy: cancelled events away from the head remain stored
        until earlier events have been popped or discarded.
        """
        while self._heap and self._heap[0][3].cancelled:
            heapq.heappop(self._heap)

    def push(self, event: Event) -> None:
        """Add an event with an assigned id to the queue.

        Parameters
        ----------
        event : Event
            Event whose ``event_id`` has already been assigned by ``Timeline``.

        Raises
        ------
        ValueError
            If ``event.event_id`` is ``None``.
        """
        if event.event_id is None:
            raise ValueError(
                "Event must have event_id assigned before pushing to EventQueue"
            )

        key = event_ordering_key(event)
        heapq.heappush(self._heap, (*key, event))

    def peek(self) -> Event:
        """Return the next non-cancelled event without removing it.

        Returns
        -------
        Event
            Earliest executable event according to ``event_ordering_key``.

        Raises
        ------
        IndexError
            If the queue is empty or contains only cancelled events.
        """
        self._discard_cancelled()

        if not self._heap:
            raise IndexError("peek from empty EventQueue")

        return self._heap[0][3]

    def pop(self) -> Event:
        """Remove and return the next non-cancelled event.

        Returns
        -------
        Event
            Earliest executable event according to ``event_ordering_key``.

        Raises
        ------
        IndexError
            If the queue is empty or contains only cancelled events.
        """
        self._discard_cancelled()

        if not self._heap:
            raise IndexError("pop from empty EventQueue")

        return heapq.heappop(self._heap)[3]

    def __len__(self) -> int:
        """Return the number of heap entries currently stored.

        Notes
        -----
        The count includes cancelled events that have not yet been discarded
        lazily, so it is not necessarily the number of executable events.
        """
        return len(self._heap)
