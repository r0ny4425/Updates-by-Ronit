"""Immutable click and detection-report records.

Detector components use ``RawClick`` for low-level channel firings and
``DetectionReport`` for resolved user-facing outcomes. These records are
descriptive payloads; they do not mutate qstate or schedule events.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RawClick:
    """
    Immutable low-level click emitted by one detector channel.

    Parameters
    ----------
    detector_id : str
        Identifier of the detector channel that fired.
    time : int
        Simulation tick at which the click is reported after jitter and
        dead-time filtering.
    trigger : str
        Physical or model trigger label, such as ``"signal"``, ``"dark"``, or
        ``"afterpulse"``.
    outcome_label : str or None, default=None
        Logical outcome assigned by the readout layer to this detector.
    flags : tuple[str, ...], default=()
        Report flags attached to this raw click.
    meta : tuple[tuple[str, object], ...], default=()
        Immutable metadata carried from the detector window and readout
        exposure.

    Notes
    -----
    ``RawClick`` is the boundary between detector-channel physics and report
    resolution. Click resolvers may sort, select, or aggregate raw clicks, but
    the click record itself does not mutate qstate and does not schedule
    events. Built-in detector-channel triggers are ``"signal"``, ``"dark"``,
    and ``"afterpulse"``.
    """

    detector_id: str
    time: int
    trigger: str
    outcome_label: str | None = None
    flags: tuple[str, ...] = ()
    meta: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionReport:
    """
    Immutable detector outcome report.

    Parameters
    ----------
    report_id : str
        Stable identifier chosen by the producing detector component.
    device_id : str
        Identifier of the detector device that produced the report.
    time : int
        Simulation tick for the logical detection or readout decision.
    success : bool
        Whether the detector produced a usable logical outcome.
    outcome : object or None
        Reported logical outcome. ``None`` represents no click, no outcome, or
        a failed analysis depending on ``flags`` and metadata.
    raw_clicks : tuple[RawClick, ...]
        Low-level clicks used to resolve this report. Job-style qubit readout
        reports use an empty tuple because they do not model detector channels.
    qstate_result : object or None, default=None
        Result returned by the qstate measurement layer, when one was run.
    measurement_method, measurement_label : str or None
        Measurement method and label selected by the detector measurement
        primitive.
    selection_index, selection_probability, selection_label : optional
        Random-measurement selection metadata when ``Measure.random`` selected
        the measurement call.
    signal_id : object or None, default=None
        Input signal identifier, or a pair of identifiers for Bell analysis.
    flags : tuple[str, ...], default=()
        Stable status labels describing no-click, timeout, invalid payload,
        double-click, and related outcomes.
    meta : tuple[tuple[str, object], ...], default=()
        Immutable report metadata. Values are passed through without recursive
        validation by this record.

    Notes
    -----
    Detector components append reports to their local ``reports`` list and may
    emit the same report object through a classical output port. The report is
    descriptive; any qstate mutation already happened before it was created.
    ``success`` means the report has a usable logical outcome, not merely that
    a physical detector clicked. Failed double-click reports can carry raw
    clicks, while no-click reports carry none.
    """

    report_id: str
    device_id: str
    time: int
    success: bool
    outcome: object | None
    raw_clicks: tuple[RawClick, ...]

    qstate_result: object | None = None

    measurement_method: str | None = None
    measurement_label: str | None = None
    selection_index: int | None = None
    selection_probability: float | None = None
    selection_label: str | None = None

    signal_id: object | None = None
    flags: tuple[str, ...] = ()
    meta: tuple[tuple[str, object], ...] = ()


FLAG_OUTSIDE_GATE = "outside_gate"
"""Input arrived while the detector gate was closed."""
FLAG_DEAD_TIME_BLOCKED = "dead_time_blocked"
"""Detector recovery time prevented a candidate click."""
FLAG_NO_CLICK = "no_click"
"""No raw click was available for a click-resolved report."""
FLAG_NO_OUTCOME = "no_outcome"
"""Measurement/readout completed without a logical outcome label."""
FLAG_DOUBLE_CLICK = "double_click"
"""More than one raw click affected threshold-style resolution."""
FLAG_DARK_COUNT = "dark_count"
"""Raw click was produced by the dark-count model."""
FLAG_SIGNAL_CLICK = "signal_click"
"""Raw click was produced by signal-detection efficiency."""
FLAG_AFTERPULSE = "afterpulse"
"""Raw click was produced by the afterpulse model."""
FLAG_TIMEOUT = "timeout"
"""Detector component timed out while waiting for a matching input."""
FLAG_INVALID_PAYLOAD = "invalid_payload"
"""Detector component rejected the scheduled event payload."""


__all__ = [
    "DetectionReport",
    "FLAG_AFTERPULSE",
    "FLAG_DARK_COUNT",
    "FLAG_DEAD_TIME_BLOCKED",
    "FLAG_DOUBLE_CLICK",
    "FLAG_INVALID_PAYLOAD",
    "FLAG_NO_CLICK",
    "FLAG_NO_OUTCOME",
    "FLAG_OUTSIDE_GATE",
    "FLAG_SIGNAL_CLICK",
    "FLAG_TIMEOUT",
    "RawClick",
]
