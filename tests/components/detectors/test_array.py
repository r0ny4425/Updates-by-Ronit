from __future__ import annotations

from typing import Protocol, cast

import pytest

from simyuj.components.connections import PortDelivery, connect_ports
from simyuj.components.detectors.detector_array import DetectorArray
from simyuj.components.detectors.primitives.actions import ACTION_DETECT_SIGNAL
from simyuj.components.detectors.primitives.click import ThresholdClickResolver
from simyuj.components.detectors.primitives.gate import AlwaysOpenGate, PeriodicGate
from simyuj.components.detectors.primitives.measurement import (
    Measure,
    MeasurementCall,
    MeasurementContext,
)
from simyuj.components.detectors.primitives.params import SinglePhotonDetectorParams
from simyuj.components.detectors.primitives.readout import DetectorExposure
from simyuj.components.detectors.primitives.reports import (
    FLAG_DARK_COUNT,
    FLAG_DOUBLE_CLICK,
    FLAG_NO_CLICK,
    FLAG_OUTSIDE_GATE,
    DetectionReport,
)
from simyuj.components.detectors.primitives.rng import DetectorRNGStreams
from simyuj.components.detectors.primitives.window import evaluate_detector_windows
from simyuj.components.detectors.single_photon import SinglePhotonDetector
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.binding import binding_context

ACTION_RECEIVE_REPORT = "receive_report"
READOUT_ZX = {
    "z": {
        "0": "d0",
        "1": "d1",
    },
    "x": {
        "+": "d0",
        "-": "d1",
    },
}


class _LabeledResult(Protocol):
    label: str


class Result:
    def __init__(self, label: object) -> None:
        self.label = label


class OneDarkCountRNG:
    def poisson(self, lam: float) -> int:
        return 1


class DummyComponent(Component):
    def handle_event(self, event, timeline) -> None:
        raise NotImplementedError


class ReportSink(Component):
    def __init__(self, device_id: str = "sink") -> None:
        self.device_id = device_id
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )
        self.received: list[DetectionReport] = []
        self.received_times: list[int] = []
        self.received_events: list[Event] = []

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        if event.action != ACTION_RECEIVE_REPORT:
            raise ValueError("unexpected sink action")

        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("sink payload_ref must be PortDelivery")

        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("sink received delivery for wrong port")

        payload = event.payload_ref.payload
        if not isinstance(payload, DetectionReport):
            raise TypeError("sink payload must be DetectionReport")

        self.received.append(payload)
        self.received_times.append(timeline.current_time)
        self.received_events.append(event)


class ClosedGate:
    def is_open(self, time: int) -> bool:
        return False

    def window_containing(self, time: int):
        return None

    def active_duration_between(self, start: int, end: int) -> int:
        return 0


class CaptureMeasurement:
    def __init__(self) -> None:
        self.contexts: list[MeasurementContext] = []

    def __call__(
        self,
        context: MeasurementContext,
        rng: object | None,
    ) -> MeasurementCall:
        self.contexts.append(context)
        return Measure.basis("z").choose(context, rng=rng)


class CaptureRandomMeasurement:
    def __init__(self) -> None:
        self.calls: list[MeasurementCall] = []
        self.measurement = Measure.random({"z": 0.5, "x": 0.5})

    def __call__(
        self,
        context: MeasurementContext,
        rng: object | None,
    ) -> MeasurementCall:
        call = self.measurement.choose(context, rng=rng)
        self.calls.append(call)
        return call


def _detectors() -> tuple[SinglePhotonDetector, SinglePhotonDetector]:
    return (
        SinglePhotonDetector(detector_id="d0"),
        SinglePhotonDetector(detector_id="d1"),
    )


def _perfect_detectors() -> tuple[SinglePhotonDetector, SinglePhotonDetector]:
    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )
    return (
        SinglePhotonDetector(detector_id="d0", params=params),
        SinglePhotonDetector(detector_id="d1", params=params),
    )


def _readout() -> dict[str, dict[str, str]]:
    return {basis: dict(mapping) for basis, mapping in READOUT_ZX.items()}


def _empty_readout(_context: object) -> tuple[DetectorExposure, ...]:
    return ()


def _source_port() -> Port:
    source = DummyComponent()
    return Port(
        name="out",
        owner=source,
        owner_id="src",
        port_kind=PortKind.QUANTUM,
        direction=PortDirection.EGRESS,
    )


def _signal_with_qstate(
    timeline: Timeline,
    *,
    signal_id: str = "sig-0",
    subsystem_label: str = "q0",
    state: str = "|0>",
) -> tuple[Signal, SubsystemId]:
    subsystem = SubsystemId(subsystem_label)

    state_ref = timeline.qstate.prepare(
        state,
        subsystems=(subsystem,),
    )

    signal = Signal(
        id=signal_id,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=timeline.current_time,
        origin="src",
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=subsystem_label,
                kind="qubit",
                index=0,
                metadata=(("qstate_subsystem", subsystem_label),),
            ),
        ),
    )

    return signal, subsystem


def _polarized_pulse_signal(timeline: Timeline) -> Signal:
    """A polarized coherent pulse, exactly as WeakCoherentPulseSource builds one.

    ``state_ref`` and ``coherent_state`` both set, and the handle stamped
    ``kind="mode"`` -- the shape that passes a bare ``state_ref`` presence check.
    """
    subsystem_label = "wcp:mode:1"
    state_ref = timeline.qstate.prepare(
        (1 + 0j, 0j),
        rep="ket",
        subsystems=(SubsystemId(subsystem_label),),
    )

    return Signal(
        id="pulse-0",
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=timeline.current_time,
        origin="wcp",
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=subsystem_label,
                kind="mode",
                index=0,
                metadata=(("qstate_subsystem", subsystem_label),),
            ),
        ),
        coherent_state=CoherentState.from_mean_photon_number(0.1),
        polarization=(1 + 0j, 0j),
        temporal_mode_sigma_s=1e-11,
    )


def _plain_signal() -> Signal:
    return Signal(
        id="sig-plain",
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="src",
    )


def _delivery(
    *,
    array: DetectorArray,
    payload: object,
    target_port: Port | None = None,
) -> PortDelivery:
    return PortDelivery(
        payload=payload,
        source_port=_source_port(),
        target_port=array.input_port if target_port is None else target_port,
        connection_id="src.out->bob_rx.in",
    )


def _event(
    *,
    array: DetectorArray,
    action: str = ACTION_DETECT_SIGNAL,
    payload_ref: object,
    time: int = 0,
) -> Event:
    return Event(
        time=time,
        target_ref=array,
        action=action,
        payload_ref=payload_ref,
        source=None,
        subsystem_id="components",
    )


def test_detector_array_logs_ready_on_bind() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.INFO, sinks=[log_sink]))
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement="z",
        readout=_readout(),
        detection_window_ticks=2,
        consume_signal=False,
        output_latency_ticks=3,
        output_priority=4,
    )

    array.bind(binding_context(timeline))

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.detector_array.ready"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "device_id": "bob_rx",
        "detector_count": 2,
        "detection_window_ticks": 2,
        "consume_signal": False,
        "output_latency_ticks": 3,
        "output_priority": 4,
    }


def test_detector_array_accepts_sequence_and_stores_tuple() -> None:
    d0, d1 = _detectors()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=[d0, d1],
        readout=_readout(),
    )

    assert isinstance(array.detectors, tuple)
    assert array.detectors == (d0, d1)


def test_detector_array_requires_readout_for_multiple_detectors() -> None:
    with pytest.raises(ValueError, match="readout is required"):
        DetectorArray(device_id="bob_rx", detectors=_detectors())


def test_detector_array_one_detector_default_readout_exposes_detector() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=(SinglePhotonDetector(detector_id="D0"),),
        readout=None,
    )

    exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=MeasurementCall(method="none", label="none"),
        qstate_result=None,
    )

    assert exposures == (DetectorExposure(detector_id="D0"),)


def test_detector_array_rejects_empty_detectors() -> None:
    with pytest.raises(ValueError, match="detectors must be non-empty"):
        DetectorArray(device_id="bob_rx", detectors=())


def test_detector_array_rejects_duplicate_detector_ids() -> None:
    with pytest.raises(ValueError, match="detector_id values must be unique"):
        DetectorArray(
            device_id="bob_rx",
            detectors=(
                SinglePhotonDetector(detector_id="d0"),
                SinglePhotonDetector(detector_id="d0"),
            ),
            readout=_readout(),
        )


def test_detector_array_rejects_non_detector_entries() -> None:
    with pytest.raises(TypeError, match="detectors must contain SinglePhotonDetector"):
        DetectorArray(
            device_id="bob_rx",
            detectors=(object(),),  # type: ignore[arg-type]
        )


def test_detector_array_validates_measurement_spec_once() -> None:
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement=Measure.basis("z"),
        readout=_readout(),
    )

    assert array._measurement.label == "z"


def test_detector_array_rejects_invalid_measurement_spec() -> None:
    with pytest.raises(Exception):
        DetectorArray(
            device_id="bob_rx",
            detectors=_detectors(),
            measurement="not-a-basis",
            readout=_readout(),
        )


@pytest.mark.parametrize("bad_window", [0, -1])
def test_detector_array_rejects_non_positive_detection_window(
    bad_window: int,
) -> None:
    with pytest.raises(ValueError, match="detection_window_ticks must be positive"):
        DetectorArray(
            device_id="bob_rx",
            detectors=_detectors(),
            detection_window_ticks=bad_window,
            readout=_readout(),
        )


def test_detector_array_rejects_negative_output_latency() -> None:
    with pytest.raises(ValueError, match="output_latency_ticks must be non-negative"):
        DetectorArray(
            device_id="bob_rx",
            detectors=_detectors(),
            output_latency_ticks=-1,
            readout=_readout(),
        )


def test_detector_array_requires_gate_model_methods() -> None:
    with pytest.raises(TypeError, match="gate_model must provide"):
        DetectorArray(
            device_id="bob_rx",
            detectors=_detectors(),
            gate_model=object(),  # type: ignore[arg-type]
            readout=_readout(),
        )


def test_detector_array_requires_click_resolver_resolve() -> None:
    with pytest.raises(TypeError, match="click_resolver must provide resolve"):
        DetectorArray(
            device_id="bob_rx",
            detectors=_detectors(),
            click_resolver=object(),  # type: ignore[arg-type]
            readout=_readout(),
        )


def test_detector_array_bound_device_processes_scheduled_detection() -> None:
    timeline = Timeline(master_seed=123)
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        click_resolver=ThresholdClickResolver(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )

    array.bind(binding_context(timeline))
    signal, _subsystem = _signal_with_qstate(timeline, state="|0>")

    timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        )
    )
    timeline.run_until_empty()

    assert len(array.reports) == 1
    assert array.reports[0].success is True
    assert array.reports[0].outcome == "0"


def test_detector_array_bind_is_idempotent_for_same_timeline() -> None:
    timeline = Timeline(master_seed=123)
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )

    array.bind(binding_context(timeline))
    array.bind(binding_context(timeline))
    signal, _subsystem = _signal_with_qstate(timeline)

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert len(array.reports) == 1


def test_detector_array_cannot_rebind_to_different_timeline() -> None:
    first = Timeline(master_seed=1)
    second = Timeline(master_seed=2)
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )

    array.bind(binding_context(first))

    with pytest.raises(RuntimeError, match="already bound to another timeline"):
        array.bind(binding_context(second))


def test_detector_array_handle_event_requires_bind() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )

    event = Event(
        time=0,
        target_ref=array,
        action="anything",
        payload_ref=None,
        source=None,
        subsystem_id="components",
    )

    with pytest.raises(RuntimeError, match="must be bound"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_wrong_action_after_bind() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    array.bind(binding_context(timeline))

    event = Event(
        time=0,
        target_ref=array,
        action="anything",
        payload_ref=None,
        source=None,
        subsystem_id="components",
    )

    with pytest.raises(ValueError, match="unsupported event action"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_non_port_delivery_payload() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    array.bind(binding_context(timeline))

    event = _event(
        array=array,
        payload_ref=object(),
    )

    with pytest.raises(TypeError, match="payload_ref must be PortDelivery"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_delivery_to_wrong_port() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    other = DetectorArray(
        device_id="other_rx",
        detectors=_detectors(),
        readout=_readout(),
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(timeline)

    event = _event(
        array=array,
        payload_ref=_delivery(
            array=array,
            payload=signal,
            target_port=other.input_port,
        ),
    )

    with pytest.raises(ValueError, match="target_port must be this array"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_mode_role_signal() -> None:
    """A polarized coherent pulse must not be measured as a qubit carrier.

    It carries a ``state_ref``, so a presence check alone lets it through to a
    measurement that routes the whole pulse to one detector -- no double clicks
    at any mean photon number, and a click rate that ignores mu entirely.
    """
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    array.bind(binding_context(timeline))

    signal = _polarized_pulse_signal(timeline)

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    with pytest.raises(ValueError, match="kind='mode'") as excinfo:
        array.handle_event(event, timeline)

    assert "not implemented yet" in str(excinfo.value)
    assert not array.reports


def test_detector_array_rejects_non_signal_delivery_payload() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    array.bind(binding_context(timeline))

    event = _event(
        array=array,
        payload_ref=_delivery(
            array=array,
            payload=object(),
        ),
    )

    with pytest.raises(TypeError, match="payload must be Signal"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_event_for_different_target() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    other = DetectorArray(
        device_id="other_rx",
        detectors=_detectors(),
        readout=_readout(),
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(timeline)

    event = Event(
        time=0,
        target_ref=other,
        action=ACTION_DETECT_SIGNAL,
        payload_ref=_delivery(array=array, payload=signal),
        source=None,
        subsystem_id="components",
    )

    with pytest.raises(ValueError, match="target_ref must be this DetectorArray"):
        array.handle_event(event, timeline)


def test_detector_array_rejects_event_from_different_timeline() -> None:
    first = Timeline()
    second = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
    )
    array.bind(binding_context(first))

    signal, _subsystem = _signal_with_qstate(second)

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    with pytest.raises(RuntimeError, match="different timeline"):
        array.handle_event(event, second)


def test_detector_array_closed_gate_stores_failed_report() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
        gate_model=ClosedGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(timeline, signal_id="sig-closed")

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(array.reports) == 1

    report = array.reports[0]
    assert report.device_id == "bob_rx"
    assert report.time == timeline.current_time
    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert report.qstate_result is None
    assert report.signal_id == "sig-closed"
    assert FLAG_OUTSIDE_GATE in report.flags


def test_detector_array_logs_failed_detection_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
        gate_model=ClosedGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))
    signal, _subsystem = _signal_with_qstate(timeline, signal_id="sig-closed")

    scheduled = timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.detector_array.detect"
    )
    report = array.reports[0]

    assert record.level is LogLevel.TRACE
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_DETECT_SIGNAL
    assert dict(record.meta) == {
        "device_id": "bob_rx",
        "signal_id": "sig-closed",
        "report_id": report.report_id,
        "measurement_label": None,
        "success": False,
        "outcome": None,
        "click_count": 0,
        "flags": (FLAG_OUTSIDE_GATE,),
    }


def test_detector_array_closed_gate_consumes_signal_when_enabled() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
        gate_model=ClosedGate(),
        consume_signal=True,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(timeline)

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    with pytest.raises(Exception):
        timeline.qstate.state_of(subsystem)


def test_detector_array_closed_gate_keeps_signal_when_consume_disabled() -> None:
    timeline = Timeline()
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        readout=_readout(),
        gate_model=ClosedGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(timeline)

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert timeline.qstate.state_of(subsystem) is not None


def test_detector_array_open_gate_builds_measurement_context() -> None:
    timeline = Timeline()
    capture = CaptureMeasurement()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement=capture,
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-measure",
        subsystem_label="q0",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(capture.contexts) == 1

    context = capture.contexts[0]
    assert context.device_id == "bob_rx"
    assert context.time == timeline.current_time
    assert context.signal is signal
    assert context.signal_targets == (subsystem,)
    assert context.input_port_name == "in"
    assert len(array.reports) == 1


def test_detector_array_open_gate_executes_z_measurement_for_zero() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-zero",
        subsystem_label="q0",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    state = timeline.qstate.state_of(subsystem)
    assert state is not None


def test_detector_array_partial_gate_clips_detector_exposure() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0"),),
        measurement="z",
        gate_model=PeriodicGate(
            period_ticks=100,
            open_duration_ticks=2,
            first_open_tick=10,
        ),
        detection_window_ticks=10,
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-partial-gate",
        subsystem_label="q0",
    )

    timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
            time=10,
        )
    )
    timeline.run_one_step()

    report = array.reports[0]

    assert len(report.raw_clicks) == 1
    assert dict(report.raw_clicks[0].meta)["window_duration_ticks"] == 2
    assert dict(report.raw_clicks[0].meta)["configured_detection_window_ticks"] == 10


def test_detector_array_fully_open_window_keeps_configured_exposure() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0"),),
        measurement="z",
        gate_model=AlwaysOpenGate(),
        detection_window_ticks=10,
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-open-gate",
        subsystem_label="q0",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    report = array.reports[0]

    assert len(report.raw_clicks) == 1
    assert dict(report.raw_clicks[0].meta)["window_duration_ticks"] == 10
    assert dict(report.raw_clicks[0].meta)["configured_detection_window_ticks"] == 10


def test_detector_array_uses_readout_time_offset_and_meta() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )

    def offset_readout(_context: object) -> tuple[DetectorExposure, ...]:
        return (
            DetectorExposure(
                detector_id="d0",
                outcome_label="late",
                time_offset_ticks=3,
                meta=(("readout_kind", "offset"),),
            ),
        )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0", params=params),),
        measurement="z",
        readout=offset_readout,
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-offset",
        subsystem_label="q0",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    click = array.reports[0].raw_clicks[0]
    meta = dict(click.meta)

    assert click.detector_id == "d0"
    assert click.time == 3
    assert click.outcome_label == "late"
    assert meta["readout_signal_present"] is True
    assert meta["readout_time_offset_ticks"] == 3
    assert meta["readout_outcome_label"] == "late"
    assert meta["readout_kind"] == "offset"


def test_detector_array_recomputes_gate_duration_after_readout_offset() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )

    def offset_readout(_context: object) -> tuple[DetectorExposure, ...]:
        return (
            DetectorExposure(
                detector_id="d0",
                outcome_label="late",
                time_offset_ticks=3,
            ),
        )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0", params=params),),
        measurement="z",
        readout=offset_readout,
        gate_model=PeriodicGate(
            period_ticks=100,
            open_duration_ticks=5,
            first_open_tick=10,
        ),
        detection_window_ticks=10,
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-offset-gate",
        subsystem_label="q0",
    )

    timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
            time=10,
        )
    )
    timeline.run_one_step()

    click = array.reports[0].raw_clicks[0]
    meta = dict(click.meta)

    assert click.time == 13
    assert meta["window_duration_ticks"] == 2
    assert meta["configured_detection_window_ticks"] == 10


def test_detector_array_arrival_outside_gate_reports_outside_gate() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=PeriodicGate(
            period_ticks=100,
            open_duration_ticks=2,
            first_open_tick=10,
        ),
        detection_window_ticks=10,
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-outside-periodic-gate",
        subsystem_label="q0",
    )

    timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
            time=12,
        )
    )
    timeline.run_one_step()

    report = array.reports[0]

    assert report.success is False
    assert report.raw_clicks == ()
    assert FLAG_OUTSIDE_GATE in report.flags


def test_detector_array_expands_readout_exposures_for_missing_detectors() -> None:
    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement="z",
        readout={"0": "d1"},
    )

    class ResultZero:
        label = "0"

    exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=MeasurementCall(method="projective", label="z"),
        qstate_result=ResultZero(),
    )

    assert exposures == (
        DetectorExposure(detector_id="d0", signal_present=False),
        DetectorExposure(detector_id="d1", outcome_label="0"),
    )


def test_detector_window_rejects_exposures_out_of_detector_order() -> None:
    detectors = (
        SinglePhotonDetector(detector_id="d0"),
        SinglePhotonDetector(detector_id="d1"),
    )

    with pytest.raises(
        RuntimeError,
        match="exposure detector_id must match detector order",
    ):
        evaluate_detector_windows(
            device_id="rx",
            time=0,
            detectors=detectors,
            exposures=(
                DetectorExposure(detector_id="d1", signal_present=True),
                DetectorExposure(detector_id="d0", signal_present=False),
            ),
            detector_rngs={},
            detection_window_ticks=1,
            gate_model=AlwaysOpenGate(),
            measurement_label="z",
            fallback_complete_time=1,
        )


def test_detector_array_two_detector_readout_maps_zero_and_one() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=(
            SinglePhotonDetector(detector_id="D0"),
            SinglePhotonDetector(detector_id="D1"),
        ),
        readout={
            "0": "D0",
            "1": "D1",
        },
    )

    call = MeasurementCall(method="projective", label="z")

    assert array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=call,
        qstate_result=Result("0"),
    ) == (
        DetectorExposure(detector_id="D0", outcome_label="0"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="1",
        ),
    )

    assert array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=call,
        qstate_result=Result("1"),
    ) == (
        DetectorExposure(
            detector_id="D0",
            signal_present=False,
            outcome_label="0",
        ),
        DetectorExposure(detector_id="D1", outcome_label="1"),
    )


def test_detector_array_basis_aware_readout_maps_detector_labels() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=(
            SinglePhotonDetector(detector_id="D0"),
            SinglePhotonDetector(detector_id="D1"),
        ),
        readout={
            "z": {
                "0": "D0",
                "1": "D1",
            },
            "x": {
                "+": "D0",
                "-": "D1",
            },
        },
    )

    z_exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=MeasurementCall(method="projective", label="z"),
        qstate_result=Result("0"),
    )
    x_exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=MeasurementCall(method="projective", label="x"),
        qstate_result=Result("+"),
    )

    assert z_exposures == (
        DetectorExposure(detector_id="D0", outcome_label="0"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="1",
        ),
    )
    assert x_exposures == (
        DetectorExposure(detector_id="D0", outcome_label="+"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="-",
        ),
    )


def test_detector_array_four_detector_polarization_readout() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=(
            SinglePhotonDetector(detector_id="D_H"),
            SinglePhotonDetector(detector_id="D_V"),
            SinglePhotonDetector(detector_id="D_plus"),
            SinglePhotonDetector(detector_id="D_minus"),
        ),
        readout={
            "H": "D_H",
            "V": "D_V",
            "+": "D_plus",
            "-": "D_minus",
        },
    )

    call = MeasurementCall(method="projective", label="polarization")

    h_exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=call,
        qstate_result=Result("H"),
    )
    plus_exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=call,
        qstate_result=Result("+"),
    )

    assert h_exposures[0] == DetectorExposure(detector_id="D_H", outcome_label="H")
    assert h_exposures[1:] == (
        DetectorExposure(
            detector_id="D_V",
            signal_present=False,
            outcome_label="V",
        ),
        DetectorExposure(
            detector_id="D_plus",
            signal_present=False,
            outcome_label="+",
        ),
        DetectorExposure(
            detector_id="D_minus",
            signal_present=False,
            outcome_label="-",
        ),
    )

    assert plus_exposures[2] == DetectorExposure(
        detector_id="D_plus",
        outcome_label="+",
    )
    assert all(
        not exposure.signal_present
        for index, exposure in enumerate(plus_exposures)
        if index != 2
    )


def test_detector_array_three_outcome_povm_readout_maps_e2() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=(
            SinglePhotonDetector(detector_id="D0"),
            SinglePhotonDetector(detector_id="D1"),
            SinglePhotonDetector(detector_id="D2"),
        ),
        readout={
            "E0": "D0",
            "E1": "D1",
            "E2": "D2",
        },
    )

    exposures = array._resolve_detector_exposures(
        signal=_plain_signal(),
        measurement_call=MeasurementCall(method="povm", label="three-outcome"),
        qstate_result=Result("E2"),
    )

    assert exposures == (
        DetectorExposure(
            detector_id="D0",
            signal_present=False,
            outcome_label="E0",
        ),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="E1",
        ),
        DetectorExposure(detector_id="D2", outcome_label="E2"),
    )


def test_detector_array_unknown_readout_outcome_fails() -> None:
    array = DetectorArray(
        device_id="rx",
        detectors=_detectors(),
        readout={
            "0": "d0",
            "1": "d1",
        },
    )

    with pytest.raises(ValueError, match="unmapped readout outcome label"):
        array._resolve_detector_exposures(
            signal=_plain_signal(),
            measurement_call=MeasurementCall(method="projective", label="z"),
            qstate_result=Result("R"),
        )


def test_detector_array_rejects_duplicate_readout_exposures() -> None:
    def duplicate_readout(_context) -> tuple[DetectorExposure, ...]:
        return (
            DetectorExposure(detector_id="d0"),
            DetectorExposure(detector_id="d0"),
        )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement="z",
        readout=duplicate_readout,
    )

    with pytest.raises(ValueError, match="duplicate detector_id"):
        array._resolve_detector_exposures(
            signal=_plain_signal(),
            measurement_call=MeasurementCall(method="projective", label="z"),
            qstate_result=None,
        )


def test_detector_array_open_gate_random_measurement_records_selection() -> None:
    timeline = Timeline(master_seed=123)
    capture = CaptureRandomMeasurement()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_detectors(),
        measurement=capture,
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(timeline)

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(capture.calls) == 1
    assert capture.calls[0].selection_index in {0, 1}
    assert capture.calls[0].selection_probability == 0.5
    assert capture.calls[0].selection_label in {"z", "x"}


def test_detector_array_open_gate_zero_clicks_detector_zero() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-zero",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(array.reports) == 1

    report = array.reports[0]
    assert report.success is True
    assert report.outcome == "0"
    assert report.signal_id == "sig-zero"
    assert report.measurement_method == "projective"
    assert report.measurement_label == "z"
    assert report.qstate_result is not None
    assert cast(_LabeledResult, report.qstate_result).label == "0"

    assert len(report.raw_clicks) == 1
    assert report.raw_clicks[0].detector_id == "d0"
    assert report.raw_clicks[0].outcome_label == "0"

    assert timeline.qstate.state_of(subsystem) is not None


def test_detector_array_logs_successful_detection_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]))

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-zero",
        state="|0>",
    )

    scheduled = timeline.schedule(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.detector_array.detect"
    )
    report = array.reports[0]

    assert record.level is LogLevel.DEBUG
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_DETECT_SIGNAL
    assert dict(record.meta) == {
        "device_id": "bob_rx",
        "signal_id": "sig-zero",
        "report_id": report.report_id,
        "measurement_label": "z",
        "success": True,
        "outcome": "0",
        "click_count": 1,
        "flags": report.flags,
    }


def test_detector_array_open_gate_one_clicks_detector_one() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-one",
        subsystem_label="q1",
        state="|1>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    report = array.reports[0]

    assert report.success is True
    assert report.outcome == "1"
    assert cast(_LabeledResult, report.qstate_result).label == "1"
    assert len(report.raw_clicks) == 1
    assert report.raw_clicks[0].detector_id == "d1"
    assert report.raw_clicks[0].outcome_label == "1"


def test_detector_array_zero_efficiency_produces_no_click_report() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=0.0,
        dark_count_rate_hz=0.0,
    )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(
            SinglePhotonDetector(detector_id="d0", params=params),
            SinglePhotonDetector(detector_id="d1", params=params),
        ),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-zero-eff",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(array.reports) == 1

    report = array.reports[0]
    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert FLAG_NO_CLICK in report.flags
    assert cast(_LabeledResult, report.qstate_result).label == "0"


def test_detector_array_open_gate_consumes_signal_when_enabled() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=True,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-consume",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(array.reports) == 1

    with pytest.raises(Exception):
        timeline.qstate.state_of(subsystem)


def test_detector_array_open_gate_keeps_signal_when_consume_disabled() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-keep",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert timeline.qstate.state_of(subsystem) is not None


def test_detector_array_none_discard_measurement_does_not_double_discard() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement=Measure.none(discard=True),
        readout=_empty_readout,
        gate_model=AlwaysOpenGate(),
        consume_signal=True,
    )
    array.bind(binding_context(timeline))

    signal, subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-none-discard",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    assert len(array.reports) == 1

    with pytest.raises(Exception):
        timeline.qstate.state_of(subsystem)

    report = array.reports[0]
    assert report.success is False
    assert FLAG_NO_CLICK in report.flags


def test_detector_array_detector_dead_time_blocks_second_signal() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
        dead_time_ticks=5,
    )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(
            SinglePhotonDetector(detector_id="d0", params=params),
            SinglePhotonDetector(detector_id="d1", params=params),
        ),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal_1, _subsystem_1 = _signal_with_qstate(
        timeline,
        signal_id="sig-1",
        subsystem_label="q0",
        state="|0>",
    )
    signal_2, _subsystem_2 = _signal_with_qstate(
        timeline,
        signal_id="sig-2",
        subsystem_label="q1",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal_1),
        ),
        timeline,
    )
    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal_2),
        ),
        timeline,
    )

    assert len(array.reports) == 2

    assert array.reports[0].success is True
    assert array.reports[0].raw_clicks[0].detector_id == "d0"

    assert array.reports[1].success is False
    assert array.reports[1].raw_clicks == ()
    assert FLAG_NO_CLICK in array.reports[1].flags


def test_detector_array_evaluates_dark_counts_on_unexposed_detectors() -> None:
    timeline = Timeline()

    d0_params = SinglePhotonDetectorParams(
        efficiency=0.0,
        dark_count_rate_hz=0.0,
    )
    d1_params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=1.0,
    )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(
            SinglePhotonDetector(detector_id="d0", params=d0_params),
            SinglePhotonDetector(detector_id="d1", params=d1_params),
        ),
        measurement="z",
        readout={
            "z": {
                "0": "d0",
                "1": "d1",
            },
        },
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    d1_rngs = array._detector_rngs["d1"]
    array._detector_rngs["d1"] = DetectorRNGStreams(
        efficiency=d1_rngs.efficiency,
        dark=OneDarkCountRNG(),
        jitter=d1_rngs.jitter,
        afterpulse=d1_rngs.afterpulse,
    )

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-dark-unexposed",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    report = array.reports[0]

    assert report.success is True
    assert report.outcome == "1"
    assert len(report.raw_clicks) == 1

    click = report.raw_clicks[0]
    assert click.detector_id == "d1"
    assert click.trigger == "dark"
    assert click.outcome_label == "1"
    assert FLAG_DARK_COUNT in click.flags
    assert dict(click.meta)["readout_signal_present"] is False


def test_detector_array_signal_and_dark_count_double_click_fails() -> None:
    timeline = Timeline()

    d0_params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )
    d1_params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=1.0,
    )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(
            SinglePhotonDetector(detector_id="d0", params=d0_params),
            SinglePhotonDetector(detector_id="d1", params=d1_params),
        ),
        measurement="z",
        readout={
            "z": {
                "0": "d0",
                "1": "d1",
            },
        },
        gate_model=AlwaysOpenGate(),
        click_resolver=ThresholdClickResolver(double_click_policy="fail"),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    d1_rngs = array._detector_rngs["d1"]
    array._detector_rngs["d1"] = DetectorRNGStreams(
        efficiency=d1_rngs.efficiency,
        dark=OneDarkCountRNG(),
        jitter=d1_rngs.jitter,
        afterpulse=d1_rngs.afterpulse,
    )

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-double-click",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    report = array.reports[0]

    assert report.success is False
    assert report.outcome is None
    assert FLAG_DOUBLE_CLICK in report.flags
    assert tuple(click.outcome_label for click in report.raw_clicks) == ("0", "1")


def test_detector_array_report_includes_random_measurement_selection() -> None:
    timeline = Timeline(master_seed=123)

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement=Measure.random({"z": 0.5, "x": 0.5}),
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-random",
        subsystem_label="q0",
        state="|0>",
    )

    event = _event(
        array=array,
        payload_ref=_delivery(array=array, payload=signal),
    )

    array.handle_event(event, timeline)

    report = array.reports[0]

    assert report.measurement_method == "projective"
    assert report.selection_index in {0, 1}
    assert report.selection_probability == 0.5
    assert report.selection_label in {"z", "x"}


def test_detector_array_unconnected_output_only_stores_report() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-store-only",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert len(array.reports) == 1
    assert timeline.events_scheduled == 0


def test_detector_array_connected_output_transmits_open_gate_report() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-output",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert len(array.reports) == 1
    assert timeline.events_scheduled == 1

    timeline.run_one_step()

    assert sink.received == [array.reports[0]]
    assert sink.received[0] is array.reports[0]
    assert sink.received_times == [0]


def test_detector_array_output_latency_is_respected() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
        output_latency_ticks=7,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-latency",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert timeline.current_time == 0
    assert sink.received == []

    timeline.run_one_step()

    assert timeline.current_time == 7
    assert sink.received == [array.reports[0]]
    assert sink.received_times == [7]


def test_detector_array_no_click_output_waits_for_detection_window() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=0.0,
        dark_count_rate_hz=0.0,
    )
    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0", params=params),),
        measurement="z",
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
        detection_window_ticks=5,
        output_latency_ticks=2,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-no-click-latency",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert len(array.reports) == 1
    assert array.reports[0].success is False
    assert sink.received == []

    timeline.run_one_step()

    assert timeline.current_time == 7
    assert sink.received == [array.reports[0]]
    assert sink.received_times == [7]


def test_detector_array_delayed_click_output_uses_click_time() -> None:
    timeline = Timeline()

    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )

    def offset_readout(_context: object) -> tuple[DetectorExposure, ...]:
        return (
            DetectorExposure(
                detector_id="d0",
                outcome_label="late",
                time_offset_ticks=3,
            ),
        )

    array = DetectorArray(
        device_id="bob_rx",
        detectors=(SinglePhotonDetector(detector_id="d0", params=params),),
        measurement="z",
        readout=offset_readout,
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
        detection_window_ticks=5,
        output_latency_ticks=2,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-delayed-click-output",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert array.reports[0].raw_clicks[0].time == 3
    assert sink.received == []

    timeline.run_one_step()

    assert timeline.current_time == 5
    assert sink.received == [array.reports[0]]
    assert sink.received_times == [5]


def test_detector_array_connected_output_transmits_closed_gate_report() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        readout=_readout(),
        gate_model=ClosedGate(),
        consume_signal=False,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-closed-output",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    assert len(array.reports) == 1
    assert array.reports[0].success is False
    assert FLAG_OUTSIDE_GATE in array.reports[0].flags

    timeline.run_one_step()

    assert sink.received == [array.reports[0]]
    assert sink.received[0].signal_id == "sig-closed-output"


def test_detector_array_output_event_metadata_identifies_report() -> None:
    timeline = Timeline()

    array = DetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_readout(),
        gate_model=AlwaysOpenGate(),
        consume_signal=False,
    )
    sink = ReportSink()

    connect_ports(
        array.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    array.bind(binding_context(timeline))

    signal, _subsystem = _signal_with_qstate(
        timeline,
        signal_id="sig-meta",
        subsystem_label="q0",
        state="|0>",
    )

    array.handle_event(
        _event(
            array=array,
            payload_ref=_delivery(array=array, payload=signal),
        ),
        timeline,
    )

    timeline.run_one_step()

    event = sink.received_events[0]
    report = array.reports[0]

    assert event.meta["device_id"] == "bob_rx"
    assert event.meta["output_port"] == "out"
    assert event.meta["report_id"] == report.report_id
    assert event.meta["signal_id"] == "sig-meta"
    assert event.meta["detector_array"] == "bob_rx"
