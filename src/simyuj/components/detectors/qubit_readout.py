"""Explicit qstate readout component for detector-style reports.

This module models readout jobs that target known qstate subsystems directly,
without receiving a quantum signal through an ingress port. ``QubitReadoutDevice``
turns ``QubitReadoutJob`` events into ``DetectionReport`` records and can apply
classical readout distortion after the true qstate measurement result is known.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isclose
from typing import Protocol, cast, runtime_checkable

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.qstate import SubsystemId
from simyuj.runtime.binding import BindingContext
from simyuj.tracing.levels import LogLevel

from ..connections import PortConnection
from ..ports import Port, PortDirection, PortKind
from .primitives.actions import ACTION_RUN_QUBIT_READOUT
from .primitives.measurement import Measure, MeasurementCall, MeasurementContext
from .primitives.readout import run_qubit_readout
from .primitives.reports import DetectionReport


@dataclass(frozen=True, slots=True)
class QubitReadoutJob:
    """
    Immutable payload for an explicit qubit-readout event.

    Parameters
    ----------
    job_id : str
        Non-empty caller-chosen readout job identifier.
    targets : tuple[SubsystemId, ...]
        Non-empty unique qstate subsystem targets.
    measurement : object or None, default=None
        Optional per-job measurement spec. ``None`` uses the device default.
    collapse : bool or None, default=None
        Optional override for measurement collapse behavior.
    output_latency_ticks : int or None, default=None
        Optional per-job output latency override in simulation ticks.
    meta : tuple[tuple[str, object], ...], default=()
        Metadata appended to the resulting report.

    Notes
    -----
    ``QubitReadoutJob`` is the ``payload_ref`` for
    ``ACTION_RUN_QUBIT_READOUT``. It carries explicit qstate targets rather
    than a signal or port delivery. ``collapse=None`` keeps the selected
    measurement call's collapse setting; ``True`` or ``False`` overrides
    non-``none`` measurements.

    Job metadata is appended after device metadata in the resulting report.
    Duplicate metadata keys are preserved, so downstream first/last lookup
    behavior matters.
    """

    job_id: str
    targets: tuple[SubsystemId, ...]
    measurement: object | None = None
    collapse: bool | None = None
    output_latency_ticks: int | None = None
    meta: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.job_id, field_name="job_id")

        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be tuple")
        if not self.targets:
            raise ValueError("targets must be non-empty")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be unique")

        for target in self.targets:
            if not isinstance(target, SubsystemId):
                raise TypeError("targets must contain SubsystemId")

        if self.collapse is not None and type(self.collapse) is not bool:
            raise TypeError("collapse must be bool or None")

        if self.output_latency_ticks is not None:
            if type(self.output_latency_ticks) is not int:
                raise TypeError("output_latency_ticks must be int or None")
            if self.output_latency_ticks < 0:
                raise ValueError("output_latency_ticks must be non-negative")

        _validate_meta(self.meta, field_name="meta")


@runtime_checkable
class QubitReadoutModel(Protocol):
    """
    Protocol for classical readout distortion.

    Notes
    -----
    Implementations receive the true qstate result label and may return a
    reported outcome. They do not mutate qstate; measurement and optional
    collapse have already happened before this method is called.
    """

    def report_outcome(
        self,
        *,
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall,
        context: MeasurementContext,
        rng: object | None,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class IdentityQubitReadout:
    """Readout model that reports the true qstate outcome unchanged."""

    def report_outcome(
        self,
        *,
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall,
        context: MeasurementContext,
        rng: object | None,
    ) -> object | None:
        del qstate_result, measurement_call, context, rng
        return true_outcome


class _RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ConfusionMapQubitReadout:
    """
    Probabilistic classical confusion-map readout model.

    Parameters
    ----------
    mapping : Mapping[object, Mapping[object, float]]
        Rows keyed by true outcome. Each row maps reported outcomes to
        probabilities and must sum to one.

    Notes
    -----
    Outcomes absent from ``mapping`` pass through unchanged. Mapped outcomes
    consume the readout RNG stream supplied by the owning device. Rows are not
    normalized; each configured row must already sum to one within tolerance.
    """

    mapping: Mapping[object, Mapping[object, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, Mapping):
            raise TypeError("mapping must be Mapping")
        if not self.mapping:
            raise ValueError("mapping must be non-empty")

        for true_label, row in self.mapping.items():
            del true_label

            if not isinstance(row, Mapping):
                raise TypeError("confusion rows must be Mapping")
            if not row:
                raise ValueError("confusion rows must be non-empty")

            total = 0.0
            for reported_label, probability_raw in row.items():
                del reported_label

                if type(probability_raw) is bool or not isinstance(
                    probability_raw,
                    (int, float),
                ):
                    raise TypeError("confusion probability must be numeric")

                probability = float(probability_raw)

                if probability < 0.0:
                    raise ValueError("confusion probability must be non-negative")

                total += probability

            if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("confusion row probabilities must sum to 1")

    def report_outcome(
        self,
        *,
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall,
        context: MeasurementContext,
        rng: object | None,
    ) -> object | None:
        del qstate_result, measurement_call, context

        if true_outcome not in self.mapping:
            return true_outcome

        if rng is None or not hasattr(rng, "random"):
            raise ValueError("rng with random() is required for confusion-map readout")

        row = self.mapping[true_outcome]
        draw = float(cast(_RandomSource, rng).random())

        selected = None
        cumulative = 0.0

        for reported_label, probability in row.items():
            selected = reported_label
            cumulative += float(probability)
            if draw < cumulative:
                return reported_label

        return selected


@dataclass(frozen=True, slots=True)
class _CallableQubitReadout:
    """
    Adapter for positional callable readout models.

    The wrapped callable receives ``true_outcome``, ``qstate_result``,
    ``measurement_call``, ``context``, and ``rng`` in that order.
    """

    resolver: Callable[
        [
            object | None,
            object | None,
            MeasurementCall,
            MeasurementContext,
            object | None,
        ],
        object | None,
    ]

    def report_outcome(
        self,
        *,
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall,
        context: MeasurementContext,
        rng: object | None,
    ) -> object | None:
        return self.resolver(
            true_outcome,
            qstate_result,
            measurement_call,
            context,
            rng,
        )


def qubit_readout_model_from_spec(value: object) -> QubitReadoutModel:
    """
    Convert a readout-model specification to a ``QubitReadoutModel``.

    Parameters
    ----------
    value : object
        ``None`` for identity readout, a confusion mapping, an existing
        ``QubitReadoutModel``, or a callable.
    """

    if value is None:
        return IdentityQubitReadout()

    if isinstance(value, Mapping):
        return ConfusionMapQubitReadout(value)

    if isinstance(value, QubitReadoutModel):
        return value

    if callable(value):
        return _CallableQubitReadout(value)

    raise TypeError("cannot convert value to QubitReadoutModel")


@dataclass(slots=True)
class QubitReadoutDevice(Component):
    """
    Event-driven component for explicit qstate qubit readout jobs.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier.
    measurement : object, default="z"
        Default measurement spec converted through ``Measure.from_spec``.
    readout_model : object or None, default=None
        Classical readout model spec. ``None`` reports true qstate labels.
    output_latency_ticks : int, default=0
        Default delay before emitting the report through ``output_port``.
    output_priority : int, default=0
        Priority used for scheduled output-report events.
    detector_meta : tuple[tuple[str, object], ...], default=()
        Metadata appended to every report, before job metadata.

    Attributes
    ----------
    output_port : Port
        Classical egress port named ``"out"``.
    reports : list[DetectionReport]
        Stored reports in handling order.

    Notes
    -----
    ``QubitReadoutDevice`` accepts only ``ACTION_RUN_QUBIT_READOUT`` events
    with ``QubitReadoutJob`` payloads. It does not own a quantum input port and
    does not consume a signal. The job's explicit targets are measured through
    the timeline qstate manager; the readout model can distort only the
    reported outcome, not the qstate result.

    Per-job ``output_latency_ticks`` overrides the device default only for
    output emission. The stored ``DetectionReport.time`` remains the event
    handling time.

    Bound RNG streams are timeline-owned and declared during ``bind`` for
    measurement choice, qstate measurement, and readout-model sampling.
    """

    device_id: str

    measurement: object = "z"
    readout_model: object | None = None

    output_latency_ticks: int = 0
    output_priority: int = 0
    detector_meta: tuple[tuple[str, object], ...] = ()

    output_port: Port = field(init=False)
    reports: list[DetectionReport] = field(init=False, default_factory=list)

    _measurement: Measure = field(init=False, repr=False)
    _readout_model: QubitReadoutModel = field(init=False, repr=False)
    _bound_timeline_id: int | None = field(init=False, default=None, repr=False)

    _measurement_choice_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _qstate_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _readout_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")

        self._measurement = Measure.from_spec(self.measurement)
        self._readout_model = qubit_readout_model_from_spec(self.readout_model)

        if type(self.output_latency_ticks) is not int:
            raise TypeError("output_latency_ticks must be int")
        if self.output_latency_ticks < 0:
            raise ValueError("output_latency_ticks must be non-negative")

        if type(self.output_priority) is not int:
            raise TypeError("output_priority must be int")

        _validate_meta(self.detector_meta, field_name="detector_meta")

        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def bind(self, context: BindingContext) -> None:
        """
        Bind the device to one timeline and declare deterministic RNG streams.

        Binding is idempotent for the same timeline and rejects a different
        timeline.
        """

        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "qubit readout device is already bound to another timeline"
                )
            return

        self._measurement_choice_rng = timeline.rng(
            self.device_id,
            "qubit_readout",
            "measurement_choice",
        )
        self._qstate_rng = timeline.rng(
            self.device_id,
            "qubit_readout",
            "qstate_measurement",
        )
        self._readout_rng = timeline.rng(
            self.device_id,
            "qubit_readout",
            "readout_model",
        )

        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.detectors.qubit_readout.ready",
            "qubit readout ready",
            meta={
                "device_id": self.device_id,
                "readout_model": type(self._readout_model).__name__,
                "output_latency_ticks": self.output_latency_ticks,
                "output_priority": self.output_priority,
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """
        Handle one qubit-readout event dispatched by the timeline.

        Parameters
        ----------
        event : Event
            Scheduled event. Only ``ACTION_RUN_QUBIT_READOUT`` is accepted.
        timeline : Timeline
            Bound timeline executing the event.
        """

        self._validate_event_context(event=event, timeline=timeline)

        if event.action != ACTION_RUN_QUBIT_READOUT:
            raise ValueError(
                f"unsupported event action for QubitReadoutDevice: {event.action!r}"
            )

        job = self._require_readout_job(event.payload_ref)
        self._handle_readout_job(
            job=job,
            timeline=timeline,
            event_id=event.event_id,
            action=event.action,
        )

    def _validate_event_context(
        self,
        *,
        event: Event,
        timeline: Timeline,
    ) -> None:
        if not isinstance(event, Event):
            raise TypeError("event must be Event")

        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")

        if self._bound_timeline_id is None:
            raise RuntimeError(
                "qubit readout device must be bound before handling events"
            )

        if self._bound_timeline_id != id(timeline):
            raise RuntimeError(
                "qubit readout device received event for a different timeline"
            )

        if event.target_ref is not self:
            raise ValueError("event target_ref must be this QubitReadoutDevice")

    def _require_readout_job(self, payload: object) -> QubitReadoutJob:
        if not isinstance(payload, QubitReadoutJob):
            raise TypeError(
                "ACTION_RUN_QUBIT_READOUT payload_ref must be QubitReadoutJob"
            )
        return payload

    def _handle_readout_job(
        self,
        *,
        job: QubitReadoutJob,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        time = timeline.current_time
        report = run_qubit_readout(
            device_id=self.device_id,
            time=time,
            targets=job.targets,
            measurement=(
                self._measurement if job.measurement is None else job.measurement
            ),
            collapse=job.collapse,
            qstate=timeline.qstate,
            measurement_choice_rng=self._require_measurement_choice_rng(),
            qstate_rng=self._require_qstate_rng(),
            readout_rng=self._require_readout_rng(),
            readout_model=self._readout_model,
            detector_meta=(
                ("job_id", job.job_id),
                ("readout_job_id", job.job_id),
                *self.detector_meta,
                *job.meta,
            ),
            report_id=(
                f"{self.device_id}:readout:{time}:" f"{len(self.reports)}:{job.job_id}"
            ),
        )

        self._store_report(
            report=report,
            timeline=timeline,
            job=job,
            event_id=event_id,
            action=action,
        )

    def _store_report(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        job: QubitReadoutJob,
        event_id: int | None,
        action: str,
    ) -> None:
        self.reports.append(report)
        timeline.log(
            LogLevel.DEBUG if report.success else LogLevel.TRACE,
            "components.detectors.qubit_readout.readout",
            "qubit readout reported",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "job_id": job.job_id,
                "report_id": report.report_id,
                "measurement_label": report.measurement_label,
                "success": report.success,
                "outcome": report.outcome,
                "flags": report.flags,
            },
        )
        self._emit_report_if_connected(report=report, timeline=timeline, job=job)

    def _emit_report_if_connected(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        job: QubitReadoutJob,
    ) -> None:
        connection: PortConnection | None = self.output_port.connection

        if connection is None:
            return

        latency = (
            self.output_latency_ticks
            if job.output_latency_ticks is None
            else job.output_latency_ticks
        )

        connection.transmit(
            report,
            timeline,
            time=timeline.current_time + latency,
            priority=self.output_priority,
            source=self,
            subsystem_id="components",
            meta={
                "device_id": self.device_id,
                "output_port": self.output_port.name,
                "report_id": report.report_id,
                "job_id": job.job_id,
                "qubit_readout_device": self.device_id,
            },
        )

    def _require_measurement_choice_rng(self) -> object:
        if self._measurement_choice_rng is None:
            raise RuntimeError("measurement choice RNG is not bound")
        return self._measurement_choice_rng

    def _require_qstate_rng(self) -> object:
        if self._qstate_rng is None:
            raise RuntimeError("qstate measurement RNG is not bound")
        return self._qstate_rng

    def _require_readout_rng(self) -> object:
        if self._readout_rng is None:
            raise RuntimeError("readout RNG is not bound")
        return self._readout_rng


def _validate_meta(
    meta: object,
    *,
    field_name: str,
) -> None:
    if not isinstance(meta, tuple):
        raise TypeError(f"{field_name} must be tuple")

    for item in meta:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{field_name} entries must be 2-tuples")

        key, _value = item
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be str")


__all__ = [
    "ConfusionMapQubitReadout",
    "IdentityQubitReadout",
    "QubitReadoutDevice",
    "QubitReadoutJob",
    "QubitReadoutModel",
    "qubit_readout_model_from_spec",
]
