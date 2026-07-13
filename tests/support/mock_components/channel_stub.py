"""
Mock Channel Stub

Purpose: Test that Timeline correctly handles event scheduling within batches,
         enforces batch boundaries, and prevents re-entry into current batch.

This is NOT a production channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.tracing.levels import LogLevel

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline

QUANTUM_SIGNAL_OUT_ACTIONS = frozenset({"quantum.signal.out"})
QUANTUM_SIGNAL_IN = "quantum.signal.in"


class ChannelStub(Component):
    """
    Deterministic mock channel for Timeline batch-closure validation.

        This stub receives SIGNAL_OUT events and schedules exactly one SIGNAL_IN
    event with a fixed propagation delay. It is used exclusively for testing
    Timeline batch execution and event scheduling semantics.

    Attributes
    ----------
    output_target : Component
        The component that will receive SIGNAL_IN events.
        Typically another mock component or detector stub.

    propagation_delay : int
        Fixed delay in simulation ticks between receiving SIGNAL_OUT
        and scheduling SIGNAL_IN. Hard-coded to 50 ticks.
    """

    # Hard-coded delay (ticks)
    propagation_delay: int = 50

    def __init__(self, output_target: Component) -> None:
        """
        Initialize ChannelStub with an output target.

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
        Handle SIGNAL_OUT event by scheduling exactly one SIGNAL_IN event.

        Parameters
        ----------
        event : Event
            The event to handle. Must have action="quantum.signal.out".
            Event is treated as read-only (not mutated).

        timeline : Timeline
            Reference to Timeline for scheduling new events.
            Used only for timeline.schedule().

        Raises
        ------
        ValueError
            If event.action is not "quantum.signal.out"
        """
        # Explicit event-type checking
        if event.action not in QUANTUM_SIGNAL_OUT_ACTIONS:
            raise ValueError(
                f"ChannelStub only handles quantum.signal.out events, "
                f"received action='{event.action}'"
            )

        # Calculate output event time (fixed propagation delay)
        output_time = event.time + self.propagation_delay

        # Pass through payload unchanged (no signal modification)
        output_payload = event.payload_ref

        # Schedule exactly one output event
        scheduled_event = timeline.schedule(
            Event(
                time=output_time,
                target_ref=self.output_target,
                action=QUANTUM_SIGNAL_IN,
                payload_ref=output_payload,
            )
        )

        timeline.log(
            LogLevel.DEBUG,
            "component.channel.forward",
            "signal forwarded",
            event=scheduled_event,
            source_name=type(self).__name__,
            target_name=type(self.output_target).__name__,
            meta={
                "input_event_id": event.event_id,
                "delay_ticks": self.propagation_delay,
            },
        )
