"""
Mock Emitter Stub

Purpose: Test that Timeline correctly dispatches events and components can schedule
         new events without Timeline implementation errors.

This is NOT a production emitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.tracing.levels import LogLevel

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline

SOURCE_EMIT_START_ACTIONS = frozenset({"source.emit.start"})
QUANTUM_SIGNAL_OUT = "quantum.signal.out"


class EmitterStub(Component):
    """
    Deterministic mock emitter for Timeline execution validation.

    This stub receives EMIT_START events and schedules exactly one SIGNAL_OUT
    event with a fixed delay.

    Attributes
    ----------
    output_target : Component
        The component that will receive SIGNAL_OUT events.
        Typically another mock component for testing.

    emit_delay : int
        Fixed delay in simulation ticks between receiving EMIT_START
        and scheduling SIGNAL_OUT. Hard-coded to 100 ticks.

    emit_count : int
        Counter tracking total number of EMIT_START events handled.
        Used for generating unique signal IDs and test assertions.

    """

    # Hard-coded delay (ticks)
    emit_delay: int = 10

    def __init__(self, output_target: Component) -> None:
        """
        Initialize EmitterStub with an output target.

        Parameters
        ----------
        output_target : Component
            The component that will receive SIGNAL_OUT events.
            Must be a valid Component instance.

        Raises
        ------
        TypeError
            If output_target is not a Component instance.
        """
        if not isinstance(output_target, Component):
            raise TypeError(
                f"output_target must be a Component instance, "
                f"got {type(output_target).__name__}"
            )

        self.output_target = output_target

    def handle_event(self, event: Event, timeline: "Timeline") -> None:
        """
        Handle EMIT_START event by scheduling exactly one SIGNAL_OUT event.

        Parameters
        ----------
        event : Event
            The event to handle. Must have action="source.emit.start".
            Event is treated as read-only (not mutated).

        timeline : Timeline
            Reference to Timeline for scheduling new events.
            Used only for timeline.schedule().

        Raises
        ------
        ValueError
            If event.action is not "source.emit.start"
        """
        # Explicit event-type checking
        if event.action not in SOURCE_EMIT_START_ACTIONS:
            raise ValueError(
                f"EmitterStub only handles source.emit.start events, "
                f"received action='{event.action}'"
            )

        timeline.log(
            LogLevel.INFO,
            "component.emitter.start",
            "emitter started",
            event=event,
            source_name=type(self).__name__,
        )

        output_time = event.time + self.emit_delay

        # Schedule exactly one output event
        scheduled_event = timeline.schedule(
            Event(
                time=output_time,
                target_ref=self.output_target,
                action=QUANTUM_SIGNAL_OUT,
                payload_ref=None,
            )
        )

        timeline.log(
            LogLevel.DEBUG,
            "component.emitter.forward",
            "signal forwarded",
            event=scheduled_event,
            source_name=type(self).__name__,
            target_name=type(self.output_target).__name__,
            meta={"input_event_id": event.event_id, "delay_ticks": self.emit_delay},
        )
