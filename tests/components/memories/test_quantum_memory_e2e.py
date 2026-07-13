from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import (
    ACTION_TRANSMIT_QUANTUM,
    Port,
    PortDelivery,
    QuantumChannel,
    connect_ports,
)
from simyuj.components.detectors.detector_array import DetectorArray
from simyuj.components.detectors.primitives.actions import ACTION_DETECT_SIGNAL
from simyuj.components.detectors.primitives.params import SinglePhotonDetectorParams
from simyuj.components.detectors.single_photon import SinglePhotonDetector
from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_EMIT,
    MEMORY_MEASURE,
    MemoryAbsorbReport,
    MemoryApplyOperatorRequest,
    MemoryEmitReport,
    MemoryEmitRequest,
    MemoryMeasurementReport,
    MemoryMeasureRequest,
    MemoryOperatorReport,
    MemoryPositionStatus,
    QuantumMemory,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from simyuj.components.ports import PortDirection, PortKind
from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.components.sources import ACTION_EMIT as SOURCE_EMIT
from simyuj.components.sources import EntangledPairSource, SinglePhotonSource
from simyuj.components.sources.single_photon_source import EmissionAttempt
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.qstate import StateNotFoundError, StateSampler
from simyuj.qstate.noise import depolarizing
from simyuj.qstate.ops import unitary
from simyuj.signal import Signal
from tests.support.binding import binding_context


@dataclass(slots=True)
class NoticeSink(Component):
    device_id: str = "notice-sink"
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


@dataclass(slots=True)
class QuantumSink(Component):
    device_id: str
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


@dataclass(slots=True)
class MeasuringReceiver(Component):
    device_id: str = "measuring-receiver"
    basis: str = "z"
    received: list[Signal] = field(default_factory=list)
    outcomes: list[object] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        signal = event.payload_ref.payload
        if not isinstance(signal, Signal):
            raise TypeError("payload must be Signal")

        result = timeline.qstate.measure(
            targets=qstate_targets_from_signal(signal),
            basis=self.basis,
            collapse=True,
        )
        self.received.append(signal)
        self.outcomes.append(result.label)


@dataclass(slots=True)
class CapturingDetectorArray(DetectorArray):
    received_signals: list[Signal] = field(default_factory=list)
    received_times: list[int] = field(default_factory=list)
    pre_measure_state_refs: list[tuple[int, ...]] = field(default_factory=list)

    def handle_event(self, event, timeline) -> None:
        if isinstance(event.payload_ref, PortDelivery):
            signal = event.payload_ref.payload
            if isinstance(signal, Signal):
                targets = qstate_targets_from_signal(signal)
                self.received_signals.append(signal)
                self.received_times.append(timeline.current_time)
                self.pre_measure_state_refs.append(
                    tuple(timeline.qstate.state_of(target) for target in targets)
                )

        DetectorArray.handle_event(self, event, timeline)


class RecordingNoiseModel:
    name = "recording"
    arity = 1

    def __init__(self) -> None:
        self.durations: list[float] = []

    def resolve(self, *, duration_s: float):
        self.durations.append(duration_s)
        return depolarizing(0.0)


def _perfect_detectors() -> tuple[SinglePhotonDetector, SinglePhotonDetector]:
    params = SinglePhotonDetectorParams(
        efficiency=1.0,
        dark_count_rate_hz=0.0,
    )
    return (
        SinglePhotonDetector(detector_id="d0", params=params),
        SinglePhotonDetector(detector_id="d1", params=params),
    )


def _z_readout() -> dict[str, dict[str, str]]:
    return {"z": {"0": "d0", "1": "d1"}}


def _run_multi_hop_qstate_signal_metadata(seed: int) -> tuple[object, ...]:
    timeline = Timeline(master_seed=seed)
    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        duration_s=1e-12,
        sampler=StateSampler(
            states=("|1>",),
            probabilities=(1.0,),
            rep="ket",
            labels=("one",),
        ),
    )
    first_channel = QuantumChannel(
        channel_id="source-to-memory",
        delay_ticks=3,
        session_id="multi-hop",
    )
    memory = QuantumMemory(
        memory_id="relay.mem",
        num_positions=1,
        absorb_delay_ticks=2,
        emit_delay_ticks=1,
    )
    second_channel = QuantumChannel(
        channel_id="memory-to-detector",
        delay_ticks=4,
        session_id="multi-hop",
    )
    detector = CapturingDetectorArray(
        device_id="bob_rx",
        detectors=_perfect_detectors(),
        measurement="z",
        readout=_z_readout(),
        consume_signal=True,
    )
    detector_reports = NoticeSink(device_id="detector-reports")

    connect_ports(
        source.output_port,
        first_channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    connect_ports(
        first_channel.output_port,
        memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    connect_ports(
        memory.output_port,
        second_channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    connect_ports(
        second_channel.output_port,
        detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )
    connect_ports(
        detector.output_port,
        detector_reports.input_port,
        target_action="detector_report",
    )

    for entity in (first_channel, memory, second_channel, detector):
        entity.bind(binding_context(timeline))

    source.schedule_start(timeline)
    timeline.run_until(3)

    transported_signal = memory.positions[0].stored_signal
    assert isinstance(transported_signal, Signal)
    source_subsystem = qstate_targets_from_signal(transported_signal)[0]
    state_ref = transported_signal.state_ref
    assert state_ref is not None
    assert timeline.qstate.state_of(source_subsystem) == state_ref
    assert memory.positions[0].status is MemoryPositionStatus.ABSORBING
    assert dict(transported_signal.meta)["quantum_channel_id"] == "source-to-memory"
    transported_timing = dict(transported_signal.timing_meta)
    assert transported_timing["emission_slot_tick"] == 0
    assert transported_timing["emission_delay_ticks"] == 0
    assert transported_timing["channel_delay_ticks"] == 3
    assert transported_timing["channel_duration_ticks"] == 3
    assert transported_timing["channel_arrival_time"] == 3

    timeline.run_until(5)

    memory_subsystem = memory_subsystem_id(memory.memory_id, 0)
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].memory_subsystem == memory_subsystem
    assert timeline.qstate.state_of(memory_subsystem) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(source_subsystem)

    timeline.schedule(
        Event(
            time=6,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-to-detector",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    )
    timeline.run_until_empty()

    emitted_subsystem = emitted_photon_subsystem_id(memory.memory_id, 0, 0)
    assert detector.received_times == [11]
    assert len(detector.received_signals) == 1
    emitted_signal = detector.received_signals[0]
    assert emitted_signal.state_ref == state_ref
    assert emitted_signal.origin == memory.memory_id
    assert emitted_signal.emission_time == 7
    assert emitted_signal.state_targets[0].label == str(emitted_subsystem)
    assert emitted_signal.state_targets[0].metadata == (
        ("qstate_subsystem", str(emitted_subsystem)),
    )
    assert detector.pre_measure_state_refs == [(state_ref,)]
    emitted_meta = dict(emitted_signal.meta)
    assert emitted_meta["memory_id"] == memory.memory_id
    assert emitted_meta["position"] == 0
    assert emitted_meta["request_id"] == "emit-to-detector"
    assert emitted_meta["quantum_channel_id"] == "memory-to-detector"
    emitted_timing = dict(emitted_signal.timing_meta)
    assert emitted_timing["memory_emit_time"] == 7
    assert emitted_timing["memory_stored_time"] == 5
    assert emitted_timing["memory_storage_ticks"] == 2
    assert emitted_timing["channel_delay_ticks"] == 4
    assert emitted_timing["channel_duration_ticks"] == 4
    assert emitted_timing["channel_arrival_time"] == 11

    assert len(detector.reports) == 1
    report = detector.reports[0]
    assert report.success is True
    assert report.outcome == "1"
    assert report.signal_id == emitted_signal.id
    assert report.measurement_label == "z"
    assert report.qstate_result is not None
    assert getattr(report.qstate_result, "label") == "1"
    assert len(detector_reports.received) == 1
    assert detector_reports.received[0].payload is report
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(emitted_subsystem)
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    assert memory.positions[0].status is MemoryPositionStatus.EMPTY

    return (
        transported_signal.id,
        transported_signal.state_targets[0].label,
        transported_signal.meta,
        transported_signal.timing_meta,
        emitted_signal.id,
        emitted_signal.state_targets[0].label,
        emitted_signal.meta,
        emitted_signal.timing_meta,
        report.report_id,
        report.outcome,
        report.signal_id,
        detector.received_times,
        first_channel.received_count,
        first_channel.delivered_count,
        second_channel.received_count,
        second_channel.delivered_count,
        timeline.events_scheduled,
        timeline.events_executed,
    )


def test_multi_hop_signal_metadata_qstate_memory_detector_replays() -> None:
    first = _run_multi_hop_qstate_signal_metadata(seed=31)
    second = _run_multi_hop_qstate_signal_metadata(seed=31)

    assert first == second


def test_entangled_photon_memory_roundtrip_operator_measure_and_detector() -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    source = EntangledPairSource(device_id="eps", frequency_hz=1e6)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=1,
        emit_delay_ticks=2,
        operator_delay_ticks=1,
        measure_delay_ticks=1,
        noise_models=noise_model,
    )
    remote_sink = QuantumSink(device_id="remote")
    receiver = MeasuringReceiver(device_id="detector", basis="z")
    notice_sink = NoticeSink()
    custom_flip = unitary(
        [[0, 1], [1, 0]],
        name="custom-memory-flip",
    )

    source.bind(binding_context(timeline))
    memory.bind(binding_context(timeline))
    connect_ports(
        source.left_output_port, memory.input_port, target_action=MEMORY_ABSORB
    )
    connect_ports(
        source.right_output_port,
        remote_sink.input_port,
        target_action="receive_signal",
    )
    connect_ports(
        memory.output_port,
        receiver.input_port,
        target_action="receive_signal",
    )
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )

    timeline.schedule(
        Event(
            time=0,
            target_ref=source,
            action=SOURCE_EMIT,
            payload_ref=EmissionAttempt(
                emission_slot_tick=0,
                emission_delay_ticks=0,
            ),
        )
    )
    timeline.run_until(1)

    assert len(remote_sink.received) == 1
    right_signal = remote_sink.received[0].payload
    assert isinstance(right_signal, Signal)
    left_signal = memory.positions[0].stored_signal
    assert isinstance(left_signal, Signal)
    memory_subsystem = memory_subsystem_id(memory.memory_id, 0)
    left_source_subsystem = qstate_targets_from_signal(left_signal)[0]
    right_subsystem = qstate_targets_from_signal(right_signal)[0]
    state_ref = left_signal.state_ref

    assert right_signal.state_ref == state_ref
    assert timeline.qstate.state_of(memory_subsystem) == state_ref
    assert timeline.qstate.state_of(right_subsystem) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(left_source_subsystem)

    timeline.schedule(
        Event(
            time=3,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-flip",
                memory_id=memory.memory_id,
                positions=(0,),
                operator=custom_flip,
            ),
        )
    )
    timeline.schedule(
        Event(
            time=5,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-memory",
                memory_id=memory.memory_id,
                positions=(0,),
                measurement="z",
                collapse=True,
                destructive=False,
            ),
        )
    )
    timeline.schedule(
        Event(
            time=8,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-memory",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    )
    timeline.run_until(10)

    measurement_reports = [
        report
        for report in memory.reports
        if isinstance(report, MemoryMeasurementReport)
    ]
    assert len(measurement_reports) == 1
    memory_measurement = measurement_reports[0]
    assert memory_measurement.detection_report is not None
    memory_outcome = memory_measurement.detection_report.outcome

    assert len(receiver.received) == 1
    assert receiver.outcomes == [memory_outcome]
    emitted_signal = receiver.received[0]
    emitted_subsystem = emitted_photon_subsystem_id(memory.memory_id, 0, 0)
    assert emitted_signal is not left_signal
    assert emitted_signal.state_ref == state_ref
    assert emitted_signal.state_targets[0].label == str(emitted_subsystem)
    assert timeline.qstate.state_of(emitted_subsystem) == state_ref
    assert timeline.qstate.state_of(right_subsystem) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)

    remote_outcome = timeline.qstate.measure(
        targets=(right_subsystem,),
        basis="z",
        collapse=True,
    ).label
    assert {memory_outcome, remote_outcome} == {"0", "1"}
    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert noise_model.durations == [
        pytest.approx(2.0e-12),
        pytest.approx(2.0e-12),
        pytest.approx(5.0e-12),
    ]
    assert (
        sum(isinstance(report, MemoryOperatorReport) for report in memory.reports) == 1
    )
    assert sum(isinstance(report, MemoryEmitReport) for report in memory.reports) == 1
    assert any(
        isinstance(delivery.payload, MemoryMeasurementReport)
        for delivery in notice_sink.received
    )


def _run_entangled_pair_into_two_memories(seed: int) -> tuple[object, ...]:
    timeline = Timeline(master_seed=seed)
    source = EntangledPairSource(
        device_id="eps",
        frequency_hz=1e6,
        duration_s=1e-12,
    )
    left_memory = QuantumMemory(memory_id="alice.mem0", num_positions=1)
    right_memory = QuantumMemory(memory_id="bob.mem0", num_positions=1)
    left_notice = NoticeSink(device_id="alice-notices")
    right_notice = NoticeSink(device_id="bob-notices")

    connect_ports(
        source.left_output_port,
        left_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    connect_ports(
        source.right_output_port,
        right_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    connect_ports(
        left_memory.notice_port,
        left_notice.input_port,
        target_action="memory_notice",
    )
    connect_ports(
        right_memory.notice_port,
        right_notice.input_port,
        target_action="memory_notice",
    )

    left_memory.bind(binding_context(timeline))
    right_memory.bind(binding_context(timeline))
    source.schedule_start(timeline)

    timeline.run_until_empty()

    left_record = left_memory.positions[0]
    right_record = right_memory.positions[0]
    left_signal = left_record.stored_signal
    right_signal = right_record.stored_signal
    assert isinstance(left_signal, Signal)
    assert isinstance(right_signal, Signal)
    left_memory_subsystem = memory_subsystem_id(left_memory.memory_id, 0)
    right_memory_subsystem = memory_subsystem_id(right_memory.memory_id, 0)
    left_source_subsystem = qstate_targets_from_signal(left_signal)[0]
    right_source_subsystem = qstate_targets_from_signal(right_signal)[0]

    assert left_record.status is MemoryPositionStatus.OCCUPIED
    assert right_record.status is MemoryPositionStatus.OCCUPIED
    assert left_record.memory_subsystem == left_memory_subsystem
    assert right_record.memory_subsystem == right_memory_subsystem
    assert left_signal.state_ref == right_signal.state_ref
    assert timeline.qstate.state_of(left_memory_subsystem) == left_signal.state_ref
    assert timeline.qstate.state_of(right_memory_subsystem) == left_signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(left_source_subsystem)
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(right_source_subsystem)

    bell_result = timeline.qstate.measure_bell(
        targets=(left_memory_subsystem, right_memory_subsystem),
        collapse=False,
    )
    assert bell_result.label == "phi+"

    left_absorb_reports = [
        report
        for report in left_memory.reports
        if isinstance(report, MemoryAbsorbReport)
    ]
    right_absorb_reports = [
        report
        for report in right_memory.reports
        if isinstance(report, MemoryAbsorbReport)
    ]
    assert len(source.reports) == 1
    assert len(left_absorb_reports) == 1
    assert len(right_absorb_reports) == 1
    assert any(
        isinstance(delivery.payload, MemoryAbsorbReport)
        for delivery in left_notice.received
    )
    assert any(
        isinstance(delivery.payload, MemoryAbsorbReport)
        for delivery in right_notice.received
    )

    return (
        source.reports[0].signal_ids,
        left_signal.id,
        right_signal.id,
        left_signal.state_ref,
        right_signal.state_ref,
        str(left_record.memory_subsystem),
        str(right_record.memory_subsystem),
        left_record.status.value,
        right_record.status.value,
        bell_result.label,
        left_absorb_reports[0].status,
        right_absorb_reports[0].status,
        timeline.events_executed,
    )


def test_entangled_pair_source_stores_pair_in_two_memories_with_replay() -> None:
    first = _run_entangled_pair_into_two_memories(seed=11)
    second = _run_entangled_pair_into_two_memories(seed=11)

    assert first == second
