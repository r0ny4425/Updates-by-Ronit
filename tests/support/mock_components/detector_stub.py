"""
Mock Detector Stub

Purpose: Test that components can maintain internal state, record events
         deterministically, and optionally report results without forwarding signals.

This is NOT a production detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.messages import DeliveryReport
from simyuj.tracing.levels import LogLevel

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline

QUANTUM_SIGNAL_IN_ACTIONS = frozenset({"quantum.signal.in"})
MEASURE_DETECTION_REPORT = "measure.detection.report"


class DetectorStub(Component):
    """
    Deterministic mock detector for signal sink validation.

    This stub receives SIGNAL_IN events and records each arrival in internal
    state. It acts as a signal sink (does not forward signals). Optionally, it can
    schedule DETECTION_REPORT events to a specified target after a fixed delay.

    Attributes
    ----------
    detections : List[int]
        Ordered list of detection records. Each record contains:
        - time: Detection time (event.time)

    report_target : Optional[Component]
        Component to receive DETECTION_REPORT events.
        If None, no reports are generated.
    """

    # Hard-coded report delay (ticks)
    report_delay: int = 10

    def __init__(
        self,
        report_target: Optional[Component] = None,
    ) -> None:
        """
        Initialize DetectorStub.

        Parameters
        ----------
        report_target : Optional[Component]
            Component to receive DETECTION_REPORT events.
            If None, no reports are generated.
            If provided, must be a valid Component instance.

        Raises
        ------
        TypeError
            If report_target is not None and not a Component instance.
        """
        if report_target is not None and not isinstance(report_target, Component):
            raise TypeError(
                f"report_target must be a Component instance, "
                f"got {type(report_target).__name__}"
            )

        self.report_target = report_target

        # Internal state: ordered list of detections
        self.detections: List[int] = []

    def handle_event(self, event: Event, timeline: "Timeline") -> None:
        """
        Handle SIGNAL_IN event by recording detection.

        Parameters
        ----------
        event : Event
            The event to handle. Must have action="quantum.signal.in".
            Event is treated as read-only (not mutated).

        timeline : Timeline
            Reference to Timeline for scheduling report events.
            Used only for timeline.schedule().

        Raises
        ------
        ValueError
            If event.action is not "quantum.signal.in"
        """
        # Explicit event-type checking
        if event.action not in QUANTUM_SIGNAL_IN_ACTIONS:
            raise ValueError(
                f"DetectorStub only handles quantum.signal.in events, "
                f"received action='{event.action}'"
            )

        self.detections.append(event.time)

        timeline.log(
            LogLevel.INFO,
            "component.detector.record",
            "detector recorded detection",
            event=event,
            source_name=type(self).__name__,
            meta={"detection_count": len(self.detections)},
        )

        # Optionally schedule detection report
        if self.report_target is not None:
            report_time = event.time + self.report_delay
            report_payload = DeliveryReport(
                channel_id=type(self).__name__,
                report_time=report_time,
                delivered=True,
                payload_id=event.event_id,
            )

            # Schedule detection report
            scheduled_event = timeline.schedule(
                Event(
                    time=report_time,
                    target_ref=self.report_target,
                    action=MEASURE_DETECTION_REPORT,
                    payload_ref=report_payload,
                )
            )

            timeline.log(
                LogLevel.DEBUG,
                "component.detector.report",
                "detector scheduled report",
                event=scheduled_event,
                source_name=type(self).__name__,
                target_name=type(self.report_target).__name__,
                meta={"input_event_id": event.event_id},
            )
