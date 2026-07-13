"""Ordering keys shared by the timeline and event queue."""

from typing import Tuple

from .event import Event


def event_ordering_key(event: Event) -> Tuple[int, int, int]:
    """Return the deterministic ordering key for an event.

    Parameters
    ----------
    event : Event
        Event with an assigned ``event_id``.

    Returns
    -------
    Tuple[int, int, int]
        ``(time, priority, event_id)``. Lower tuple values execute earlier.

    Raises
    ------
    ValueError
        If ``event.event_id`` is ``None``.

    Notes
    -----
    ``event_id`` is the final tie-breaker, so equal-time, equal-priority events
    execute in timeline scheduling order.
    """
    if event.event_id is None:
        raise ValueError("Cannot order Event before event_id is assigned by Timeline")

    return (event.time, event.priority, event.event_id)
