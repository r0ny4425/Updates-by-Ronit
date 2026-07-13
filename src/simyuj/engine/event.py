"""Immutable event records for the deterministic simulation engine.

``Event`` carries the causal fields that the timeline orders and dispatches.
Events do not execute themselves and do not contain protocol-specific logic;
the target object interprets ``action`` and ``payload_ref`` when the timeline
delivers the event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Scheduled causal record delivered by ``Timeline``.

    An event states that, at integer simulation time ``time``, ``target_ref``
    should handle ``action`` with ``payload_ref``. The record carries only
    scheduling, identity, tracing, and payload-reference data; execution logic
    belongs to the target and ordering belongs to the timeline and event queue.

    Attributes
    ----------
    time : int
        Simulation timestamp, in engine ticks, at which the event is eligible
        to execute. The constructor checks the type but does not reject negative
        values; ``Timeline.schedule`` rejects events scheduled in the past.
    target_ref : Any
        Object that will receive the event. The timeline dispatch path expects
        the target to provide ``handle_event(event, timeline)``.
    action : str
        Symbolic event label interpreted by the target. The constructor checks
        only that the label is a string.
    payload_ref : Any
        Payload or payload reference delivered with the event. The engine does
        not inspect or copy this value.
    priority : int
        Tie-breaker for events with the same ``time``. Lower values execute
        first.
    event_id : Optional[int]
        Identifier assigned by ``Timeline.schedule`` for normal execution.
        Events passed directly to ``EventQueue`` must already have an id;
        events passed to ``Timeline.schedule`` must have ``event_id=None``.
    source : Any
        Optional object or identifier for the event's source. The engine uses
        this for tracing context, not ordering.
    subsystem_id : str
        Logical subsystem tag used as context for tracing or analysis.
    meta : dict[str, Any]
        Metadata for tracing, debugging, or analysis. The dictionary is not
        copied or frozen by ``Event``.
    cancelled : bool
        Lazy-cancellation flag. It is changed by timeline-owned internal
        mutation and skipped by the queue on access.

    Raises
    ------
    TypeError
        If ``time``, ``priority``, ``event_id``, ``action``, ``subsystem_id``,
        ``meta``, or ``cancelled`` has the wrong type.

    Notes
    -----
    The dataclass is frozen and slot-backed, but the engine intentionally uses
    internal mutation to assign ``event_id`` once and to mark cancellation. The
    referenced ``target_ref``, ``payload_ref``, ``source``, and ``meta`` objects
    remain owned by their callers.

    Equality follows the dataclass comparison fields: ``time``, ``priority``,
    and ``event_id``. Target, action, payload, context, metadata, and
    cancellation state are excluded from equality.
    """

    # Required causal fields.
    time: int
    target_ref: Any = field(compare=False)
    action: str = field(compare=False)
    payload_ref: Any = field(compare=False)

    # Engine-controlled ordering and identity.
    priority: int = 0

    # Assigned by Timeline; None until scheduled.
    event_id: Optional[int] = None

    # Context and tracing fields. They do not affect ordering or equality.
    source: Any = field(compare=False, default=None)
    subsystem_id: str = field(compare=False, default="core")
    meta: dict[str, Any] = field(compare=False, default_factory=dict)

    # Scheduling control.
    cancelled: bool = field(compare=False, default=False)

    def __post_init__(self) -> None:
        # Structural validation only; action and payload semantics belong to
        # the target component.

        if not isinstance(self.time, int):
            raise TypeError("Event.time must be an int")

        if not isinstance(self.priority, int):
            raise TypeError("Event.priority must be an int")

        if self.event_id is not None and not isinstance(self.event_id, int):
            raise TypeError("Event.event_id must be int or None")

        if not isinstance(self.action, str):
            raise TypeError("Event.action must be a str")

        if not isinstance(self.subsystem_id, str):
            raise TypeError("Event.subsystem_id must be a str")

        if not isinstance(self.meta, dict):
            raise TypeError("Event.meta must be a dict")

        if not isinstance(self.cancelled, bool):
            raise TypeError("Event.cancelled must be a bool")

    def _mark_cancelled(self) -> None:
        """Mark this event as cancelled for lazy queue skipping.

        Notes
        -----
        This internal hook is owned by ``Timeline.cancel``. Cancellation does
        not remove the event from the queue immediately; ``EventQueue`` drops
        cancelled events when they reach the head of the heap.
        """
        object.__setattr__(self, "cancelled", True)

    def _set_event_id(self, event_id):
        """Assign the timeline-owned event identifier once.

        Parameters
        ----------
        event_id : int
            Identifier allocated by ``Timeline``.

        Raises
        ------
        RuntimeError
            If the event already has an identifier.

        Notes
        -----
        This internal hook is owned by ``Timeline.schedule``. The method does
        not validate the type because public construction already validates
        ``event_id`` when it is provided.
        """
        if self.event_id is not None:
            raise RuntimeError("Cannot re-assign events.")
        object.__setattr__(self, "event_id", event_id)
