"""Detector-array component for signal-facing qstate measurement.

``DetectorArray`` is the generic quantum-signal detector component: it accepts
port-delivered ``Signal`` objects, resolves qstate targets, applies a
measurement/readout mapping, evaluates detector-channel windows, and stores or
emits a ``DetectionReport``. The helper functions in this module are scoped to
that event flow and do not belong to the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import validate_meta
from simyuj.primitives.validation import validate_bool
from simyuj.qstate import SubsystemId
from simyuj.runtime.binding import BindingContext
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel

from ..connections import PortDelivery
from ..ports import Port, PortDirection, PortKind
from ..quantum_targets import qstate_targets_from_signal
from .primitives.actions import ACTION_DETECT_SIGNAL
from .primitives.click import ClickPatternResolver, ThresholdClickResolver
from .primitives.gate import AlwaysOpenGate, GateModel
from .primitives.measurement import (
    Measure,
    MeasurementCall,
    MeasurementContext,
    execute_measurement_call,
)
from .primitives.readout import (
    DetectorExposure,
    ReadoutContext,
    ReadoutLayout,
    normalize_readout_exposures,
    readout_from_spec,
)
from .primitives.reports import FLAG_OUTSIDE_GATE, DetectionReport, RawClick
from .primitives.rng import DetectorRNGStreams
from .primitives.window import (
    active_detection_duration_at_arrival,
    bind_detector_rngs,
    evaluate_detector_windows,
    normalize_detectors,
    validate_gate_model,
)
from .single_photon import SinglePhotonDetector

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG


@dataclass(slots=True)
class DetectorArray(Component):
    """
    Event-driven detector array for qstate-backed quantum signals.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier.
    detectors : Sequence[SinglePhotonDetector]
        Non-empty ordered detector channels.
    measurement : object, default="z"
        Measurement spec converted through ``Measure.from_spec``.
    readout : object or None, default=None
        Readout spec mapping qstate results to detector exposures.
    gate_model : GateModel, optional
        Gate schedule queried at signal-arrival and exposure ticks.
    click_resolver : ClickPatternResolver, optional
        Resolver that converts raw clicks to a ``DetectionReport``.
    detection_window_ticks : int, default=1
        Positive detector observation window length in simulation ticks.
    consume_signal : bool, default=True
        Whether measured signal qstate targets are discarded after detection.
    output_latency_ticks : int, default=0
        Additional delay before emitting a report through the classical output
        port.
    output_priority : int, default=0
        Priority used for scheduled output-report events.
    detector_meta : tuple[tuple[str, object], ...], default=()
        Metadata made available to measurement selection and raw-click reports.

    Attributes
    ----------
    input_port : Port
        Quantum ingress port named ``"in"``.
    output_port : Port
        Classical egress port named ``"out"``.
    reports : list[DetectionReport]
        Stored reports in handling order.

    Notes
    -----
    ``DetectorArray`` accepts only ``ACTION_DETECT_SIGNAL`` events. The event
    payload must be a ``PortDelivery`` addressed to ``input_port`` whose payload
    is a ``Signal``. Detection resolves qstate targets from the signal, chooses
    and executes a measurement, maps the qstate result to detector exposures,
    evaluates detector windows, and resolves raw clicks into a report.

    When the gate is closed at arrival, no qstate measurement is run. If
    ``consume_signal`` is true, the signal targets are discarded and a
    ``FLAG_OUTSIDE_GATE`` report is stored.

    When the gate is open, ``consume_signal=True`` discards signal targets after
    measurement even if click resolution later reports no logical outcome. The
    qstate measurement result and detector-report success are separate
    concepts.

    Readout exposures from custom layouts are normalized into detector order.
    Missing detectors become unexposed entries and can still produce dark
    counts. Output emission uses the latest raw-click time when clicks exist;
    otherwise it uses the detector-window completion fallback, then adds
    ``output_latency_ticks``.

    Bound RNG streams are timeline-owned and declared during ``bind``:
    measurement selection, qstate measurement, click resolution, and per-channel
    detector streams. With fixed timeline seeds and configuration, detector
    behavior is reproducible.
    """

    device_id: str
    detectors: Sequence[SinglePhotonDetector]

    measurement: object = "z"
    readout: object | None = None
    gate_model: GateModel = field(default_factory=AlwaysOpenGate)
    click_resolver: ClickPatternResolver = field(default_factory=ThresholdClickResolver)

    detection_window_ticks: int = 1
    consume_signal: bool = True
    output_latency_ticks: int = 0
    output_priority: int = 0
    detector_meta: tuple[tuple[str, object], ...] = ()

    input_port: Port = field(init=False)
    output_port: Port = field(init=False)
    reports: list[DetectionReport] = field(init=False, default_factory=list)

    _measurement: Measure = field(init=False, repr=False)
    _readout: ReadoutLayout = field(init=False, repr=False)
    _bound_timeline_id: int | None = field(init=False, default=None, repr=False)

    _measurement_choice_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _qstate_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _resolver_rng: DeterministicRNG | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _detector_rngs: dict[str, DetectorRNGStreams] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")

        self.detectors = normalize_detectors(
            self.detectors,
            require_non_empty=True,
        )
        detector_ids = tuple(detector.detector_id for detector in self.detectors)
        self._readout = readout_from_spec(self.readout, detector_ids=detector_ids)

        self._measurement = Measure.from_spec(self.measurement)

        validate_gate_model(self.gate_model)
        _validate_click_resolver(self.click_resolver)
        validate_meta(
            self.detector_meta,
            field_name="detector_meta",
            require_hashable=False,
        )

        if type(self.detection_window_ticks) is not int:
            raise TypeError("detection_window_ticks must be int")
        if self.detection_window_ticks <= 0:
            raise ValueError("detection_window_ticks must be positive")

        validate_bool(self.consume_signal, field_name="consume_signal")

        if type(self.output_latency_ticks) is not int:
            raise TypeError("output_latency_ticks must be int")
        if self.output_latency_ticks < 0:
            raise ValueError("output_latency_ticks must be non-negative")

        if type(self.output_priority) is not int:
            raise TypeError("output_priority must be int")

        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def bind(self, context: BindingContext) -> None:
        """
        Bind the array to one timeline and declare deterministic RNG streams.

        Parameters
        ----------
        context : BindingContext
            Runtime binding context containing the timeline.

        Notes
        -----
        Binding is idempotent for the same timeline and rejects a different
        timeline. Detector RNG streams are keyed by ``device_id``,
        ``"detector_array"``, detector id, and stream role.
        """

        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "detector array is already bound to another timeline"
                )
            return

        self._measurement_choice_rng = timeline.rng(
            self.device_id,
            "detector_array",
            "measurement_choice",
        )

        self._qstate_rng = timeline.rng(
            self.device_id,
            "detector_array",
            "qstate_measurement",
        )

        self._resolver_rng = timeline.rng(
            self.device_id,
            "detector_array",
            "resolver",
        )

        self._detector_rngs = bind_detector_rngs(
            timeline=timeline,
            device_id=self.device_id,
            namespace="detector_array",
            detectors=cast(tuple[SinglePhotonDetector, ...], self.detectors),
        )

        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.detectors.detector_array.ready",
            "detector array ready",
            meta={
                "device_id": self.device_id,
                "detector_count": len(self.detectors),
                "detection_window_ticks": self.detection_window_ticks,
                "consume_signal": self.consume_signal,
                "output_latency_ticks": self.output_latency_ticks,
                "output_priority": self.output_priority,
            },
        )

    def _require_measurement_choice_rng(self) -> DeterministicRNG:
        if self._measurement_choice_rng is None:
            raise RuntimeError("measurement choice RNG is not bound")
        return self._measurement_choice_rng

    def _require_qstate_rng(self) -> DeterministicRNG:
        if self._qstate_rng is None:
            raise RuntimeError("qstate measurement RNG is not bound")
        return self._qstate_rng

    def _require_resolver_rng(self) -> DeterministicRNG:
        if self._resolver_rng is None:
            raise RuntimeError("resolver RNG is not bound")
        return self._resolver_rng

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """
        Handle one detector-array event dispatched by the timeline.

        Parameters
        ----------
        event : Event
            Scheduled event. Only ``ACTION_DETECT_SIGNAL`` is accepted.
        timeline : Timeline
            Bound timeline executing the event.

        Raises
        ------
        TypeError
            If the event, timeline, delivery, or signal payload has the wrong
            type.
        ValueError
            If the event targets another component, uses an unsupported action,
            or is delivered to the wrong port.
        RuntimeError
            If the array has not been bound or is invoked with another
            timeline.
        """

        self._validate_event_context(event=event, timeline=timeline)

        if event.action != ACTION_DETECT_SIGNAL:
            raise ValueError(
                f"unsupported event action for DetectorArray: {event.action!r}"
            )

        delivery = self._require_input_delivery(event.payload_ref)
        signal = self._require_signal_payload(delivery.payload)

        self._handle_detect_signal(
            signal=signal,
            delivery=delivery,
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
            raise RuntimeError("detector array must be bound before handling events")

        if self._bound_timeline_id != id(timeline):
            raise RuntimeError("detector array received event for a different timeline")

        if event.target_ref is not self:
            raise ValueError("event target_ref must be this DetectorArray")

    def _require_input_delivery(self, payload: object) -> PortDelivery:
        if not isinstance(payload, PortDelivery):
            raise TypeError("ACTION_DETECT_SIGNAL payload_ref must be PortDelivery")

        if payload.target_port is not self.input_port:
            raise ValueError("PortDelivery target_port must be this array's input_port")

        return payload

    def _require_signal_payload(self, payload: object) -> Signal:
        if not isinstance(payload, Signal):
            raise TypeError("PortDelivery payload must be Signal")

        return payload

    def _handle_detect_signal(
        self,
        *,
        signal: Signal,
        delivery: PortDelivery,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        targets = qstate_targets_from_signal(signal)
        time = timeline.current_time
        active_detection_duration_ticks = active_detection_duration_at_arrival(
            time=time,
            detection_window_ticks=self.detection_window_ticks,
            gate_model=self.gate_model,
        )

        if active_detection_duration_ticks <= 0:
            if self.consume_signal:
                timeline.qstate.discard(targets=targets)

            report = self._make_closed_gate_report(
                time=time,
                signal=signal,
            )
            self._store_report(
                report=report,
                timeline=timeline,
                emit_base_time=time,
                event_id=event_id,
                action=action,
            )
            return

        measurement_call, qstate_result = self._measure_signal(
            signal=signal,
            delivery=delivery,
            timeline=timeline,
            targets=targets,
        )

        exposures = self._resolve_detector_exposures(
            signal=signal,
            measurement_call=measurement_call,
            qstate_result=qstate_result,
        )

        raw_clicks, detection_complete_time = self._evaluate_detector_physics(
            time=time,
            exposures=exposures,
            qstate_result=qstate_result,
            measurement_call=measurement_call,
        )

        report = self.click_resolver.resolve(
            device_id=self.device_id,
            time=time,
            signal=signal,
            qstate_result=qstate_result,
            measurement_call=measurement_call,
            raw_clicks=raw_clicks,
            rng=self._require_resolver_rng(),
        )

        self._consume_signal_if_configured(
            timeline=timeline,
            targets=targets,
            measurement_call=measurement_call,
        )

        self._store_report(
            report=report,
            timeline=timeline,
            emit_base_time=detection_complete_time,
            event_id=event_id,
            action=action,
        )

    def _store_report(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        emit_base_time: int,
        event_id: int | None,
        action: str,
    ) -> None:
        self.reports.append(report)
        timeline.log(
            LogLevel.DEBUG if report.success else LogLevel.TRACE,
            "components.detectors.detector_array.detect",
            "detection reported",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "signal_id": report.signal_id,
                "report_id": report.report_id,
                "measurement_label": report.measurement_label,
                "success": report.success,
                "outcome": report.outcome,
                "click_count": len(report.raw_clicks),
                "flags": report.flags,
            },
        )
        self._emit_report_if_connected(
            report=report,
            timeline=timeline,
            emit_base_time=emit_base_time,
        )

    def _emit_report_if_connected(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        emit_base_time: int,
    ) -> None:
        connection = self.output_port.connection

        if connection is None:
            return

        ready_time = _report_ready_time(
            report=report,
            fallback_time=emit_base_time,
        )

        connection.transmit(
            report,
            timeline,
            time=max(timeline.current_time, ready_time) + self.output_latency_ticks,
            priority=self.output_priority,
            source=self,
            subsystem_id="components",
            meta={
                "device_id": self.device_id,
                "output_port": self.output_port.name,
                "report_id": report.report_id,
                "signal_id": report.signal_id,
                "detector_array": self.device_id,
            },
        )

    def _build_measurement_context(
        self,
        *,
        time: int,
        signal: Signal,
        delivery: PortDelivery,
        targets: tuple[SubsystemId, ...],
    ) -> MeasurementContext:
        return MeasurementContext(
            device_id=self.device_id,
            time=time,
            signal=signal,
            signal_targets=targets,
            input_port_name=delivery.target_port.name,
            detector_meta=self.detector_meta,
        )

    def _measure_signal(
        self,
        *,
        signal: Signal,
        delivery: PortDelivery,
        timeline: Timeline,
        targets: tuple[SubsystemId, ...],
    ) -> tuple[MeasurementCall, object | None]:
        context = self._build_measurement_context(
            time=timeline.current_time,
            signal=signal,
            delivery=delivery,
            targets=targets,
        )

        call = self._measurement.choose(
            context,
            rng=self._require_measurement_choice_rng(),
        )
        discard_after = self.consume_signal and call.targets == "signal"

        qstate_result = execute_measurement_call(
            call=call,
            context=context,
            qstate=timeline.qstate,
            rng=self._require_qstate_rng(),
            discard_after=discard_after,
        )

        return call, qstate_result

    def _resolve_detector_exposures(
        self,
        *,
        signal: Signal,
        measurement_call: MeasurementCall,
        qstate_result: object | None,
    ) -> tuple[DetectorExposure, ...]:
        detector_ids = tuple(detector.detector_id for detector in self.detectors)
        context = ReadoutContext(
            detector_ids=detector_ids,
            measurement_call=measurement_call,
            qstate_result=qstate_result,
            signal=signal,
        )

        exposures = self._readout.resolve_exposures(context)

        return normalize_readout_exposures(
            exposures,
            detector_ids=detector_ids,
        )

    def _evaluate_detector_physics(
        self,
        *,
        time: int,
        exposures: tuple[DetectorExposure, ...],
        qstate_result: object | None,
        measurement_call: MeasurementCall,
    ) -> tuple[tuple[RawClick, ...], int]:
        # Detector-channel physics only needs normalized exposures and the
        # selected measurement label; qstate_result remains report metadata.
        del qstate_result

        return evaluate_detector_windows(
            device_id=self.device_id,
            time=time,
            detectors=cast(tuple[SinglePhotonDetector, ...], self.detectors),
            exposures=exposures,
            detector_rngs=self._detector_rngs,
            detection_window_ticks=self.detection_window_ticks,
            gate_model=self.gate_model,
            measurement_label=measurement_call.label,
            fallback_complete_time=time,
        )

    def _consume_signal_if_configured(
        self,
        *,
        timeline: Timeline,
        targets: tuple[SubsystemId, ...],
        measurement_call: MeasurementCall,
    ) -> None:
        if not self.consume_signal:
            return

        # A no-measurement call with discard=True already removed the targets
        # inside execute_measurement_call().
        if measurement_call.method == "none" and measurement_call.discard:
            return
        if (
            measurement_call.method == "projective"
            and measurement_call.targets == "signal"
        ):
            return

        timeline.qstate.discard(targets=targets)

    def _make_closed_gate_report(
        self,
        *,
        time: int,
        signal: Signal,
    ) -> DetectionReport:
        return DetectionReport(
            report_id=(
                f"{self.device_id}:report:{time}:" f"outside-gate:{len(self.reports)}"
            ),
            device_id=self.device_id,
            time=time,
            success=False,
            outcome=None,
            raw_clicks=(),
            qstate_result=None,
            measurement_method=None,
            measurement_label=None,
            selection_index=None,
            selection_probability=None,
            selection_label=None,
            signal_id=_signal_id(signal),
            flags=(FLAG_OUTSIDE_GATE,),
            meta=(
                ("reason", "gate_closed"),
                ("consume_signal", self.consume_signal),
            ),
        )


def _validate_click_resolver(click_resolver: object) -> None:
    resolve = getattr(click_resolver, "resolve", None)
    if not callable(resolve):
        raise TypeError("click_resolver must provide resolve(...)")


def _signal_id(signal: Signal) -> object:
    signal_id = getattr(signal, "signal_id", None)
    if signal_id is not None:
        return signal_id
    return signal.id


def _report_ready_time(
    *,
    report: DetectionReport,
    fallback_time: int,
) -> int:
    if type(fallback_time) is not int:
        raise TypeError("fallback_time must be int")
    if fallback_time < 0:
        raise ValueError("fallback_time must be non-negative")

    if report.raw_clicks:
        return max(click.time for click in report.raw_clicks)

    # No-click reports use the active detector-window completion fallback.
    return fallback_time


__all__ = [
    "DetectorArray",
]
