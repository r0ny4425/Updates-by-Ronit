"""Component-level QuantumMemory behavior tests.

This file covers public memory requests, reports, qstate ownership transitions,
and scheduled delayed operations. Private completion-token regressions live in
``test_quantum_memory_internal_regression.py``; storage-noise timing lives in
``test_quantum_memory_storage_noise.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from simyuj.components import connect_ports
from simyuj.components.detectors.primitives.measurement import Measure
from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_EXPIRE,
    MEMORY_MEASURE,
    MEMORY_UPDATE_META,
    MemoryAbsorbReport,
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardReport,
    MemoryDiscardRequest,
    MemoryEmitReport,
    MemoryEmitRequest,
    MemoryExpireReport,
    MemoryExpireRequest,
    MemoryMeasurementReport,
    MemoryMeasureRequest,
    MemoryMetaUpdateReport,
    MemoryOperatorReport,
    MemoryPositionStatus,
    MemoryUpdateMetaRequest,
    QuantumMemory,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from simyuj.components.memories.quantum_memory import _MEMORY_ABSORB_COMPLETE
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.qstate import InvalidOperationError, StateNotFoundError, SubsystemId
from simyuj.qstate.ops import CNOT, X, unitary
from simyuj.signal import Signal, SignalKind
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.components.memories._quantum_memory_support import (
    NoticeSink,
    QuantumSink,
    QuantumSource,
    _absorb_position,
    _assert_position_fields,
    _count_reports,
    _signal,
)
from tests.support.binding import binding_context

# delayed-operation contract:
# - zero-delay operations complete immediately
# - nonzero-delay start events mark affected positions busy
# - busy positions reject conflicting operations before state changes
# - completion events finalize exactly once and emit exactly one report
# - completion leaves the expected final position/qstate state
# - stale completion tokens are ignored
# - wrong-status completions are ignored or rejected deterministically


def test_quantum_memory_initializes_empty_positions() -> None:
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=3)

    assert tuple(position.position for position in memory.positions) == (0, 1, 2)
    assert all(
        position.status is MemoryPositionStatus.EMPTY for position in memory.positions
    )
    assert all(position.ready_at == 0 for position in memory.positions)


def test_quantum_memory_binds_deterministic_rng_streams() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)

    memory.bind(binding_context(timeline))
    memory.bind(binding_context(timeline))

    assert memory._absorb_rng is not None
    assert memory._emit_rng is not None
    assert memory._measurement_choice_rng is not None
    assert memory._qstate_measurement_rng is not None
    assert memory._readout_model_rng is not None

    with pytest.raises(RuntimeError, match="already bound"):
        memory.bind(binding_context(Timeline(master_seed=2)))


def test_quantum_memory_logs_ready_on_bind() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.INFO, sinks=[log_sink]))
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        storage_lifetime_ticks=100,
        recovery_ticks=3,
        absorb_delay_ticks=1,
        emit_delay_ticks=2,
        operator_delay_ticks=4,
        measure_delay_ticks=5,
        absorb_success_probability=0.9,
        emit_success_probability=0.8,
    )

    memory.bind(binding_context(timeline))

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.ready"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "memory_id": "nodeA.mem0",
        "num_positions": 2,
        "storage_lifetime_ticks": 100,
        "recovery_ticks": 3,
        "absorb_delay_ticks": 1,
        "emit_delay_ticks": 2,
        "operator_delay_ticks": 4,
        "measure_delay_ticks": 5,
        "absorb_success_probability": 0.9,
        "emit_success_probability": 0.8,
    }


def test_quantum_memory_logs_absorb_report_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]))
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        storage_lifetime_ticks=100,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline, signal_id="sig-1")
    event = timeline.schedule(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id=memory.memory_id,
                signal=signal,
                position=0,
            ),
        )
    )

    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.absorb"
    )
    report = memory.reports[0]

    assert record.level is LogLevel.DEBUG
    assert record.event_id == event.event_id
    assert record.action == MEMORY_ABSORB
    assert dict(record.meta) == {
        "memory_id": "nodeA.mem0",
        "request_id": "absorb-1",
        "report_id": report.report_id,
        "position": 0,
        "occupancy_token": 1,
        "input_signal_id": "sig-1",
        "success": True,
        "status": "occupied",
        "expires_at": 100,
    }


def test_quantum_memory_logs_measurement_report_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]))
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline, position=0, state="|0>")
    event = timeline.schedule(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id=memory.memory_id,
                positions=(0,),
                measurement="z",
                destructive=True,
            ),
        )
    )

    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.measure"
    )
    report = memory.reports[-1]

    assert isinstance(report, MemoryMeasurementReport)
    assert record.level is LogLevel.DEBUG
    assert record.event_id == event.event_id
    assert record.action == MEMORY_MEASURE
    assert dict(record.meta) == {
        "memory_id": "nodeA.mem0",
        "request_id": "measure-1",
        "report_id": report.report_id,
        "positions": (0,),
        "measurement_label": "z",
        "success": True,
        "outcome": "0",
        "destructive": True,
        "cleared_positions": (0,),
        "status": "measured",
        "flags": (),
    }


def test_quantum_memory_logs_pending_delayed_operation_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=5,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline, signal_id="sig-1")

    timeline.schedule(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id=memory.memory_id,
                signal=signal,
                position=0,
            ),
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.pending"
    )

    assert record.level is LogLevel.TRACE
    assert record.action == _MEMORY_ABSORB_COMPLETE
    assert dict(record.meta) == {
        "memory_id": "nodeA.mem0",
        "request_id": "absorb-1",
        "operation": "absorb",
        "positions": (0,),
        "occupancy_tokens": (1,),
        "status": "absorbing",
        "delay_ticks": 5,
        "completion_time": 5,
    }


def test_absorb_request_relabels_photon_into_requested_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=2)
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)
    photon_subsystem = SubsystemId("photon:s1")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 1)
    state_ref = timeline.qstate.state_of(photon_subsystem)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
                position=1,
                session_id="session-1",
                meta=(("round", 1),),
            ),
        ),
        timeline,
    )

    assert timeline.qstate.state_of(memory_subsystem) == state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(photon_subsystem)

    record = memory.positions[1]
    assert record.status is MemoryPositionStatus.OCCUPIED
    assert record.memory_subsystem == memory_subsystem
    assert record.stored_signal is signal
    assert record.stored_time == 0
    assert record.last_noise_update_time == 0
    assert record.expires_at is None
    assert record.occupancy_token == 1
    assert len(memory.reports) == 1
    assert isinstance(memory.reports[0], MemoryAbsorbReport)
    assert ("occupancy_token", 1) in memory.reports[0].meta


def test_absorb_port_delivery_chooses_first_empty_position_and_emits_notice() -> None:
    timeline = Timeline(master_seed=1)
    source = QuantumSource()
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    sink = NoticeSink()
    signal = _signal(timeline)

    memory.bind(binding_context(timeline))
    quantum_connection = connect_ports(
        source.output_port,
        memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    connect_ports(
        memory.notice_port,
        sink.input_port,
        target_action="memory_notice",
    )

    quantum_connection.transmit(signal, timeline)
    timeline.run_until(0)

    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].memory_subsystem == memory_subsystem
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    assert len(sink.received) == 1
    assert isinstance(sink.received[0].payload, MemoryAbsorbReport)


def test_absorb_schedules_expiry_with_occupancy_token() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        storage_lifetime_ticks=5,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )

    assert memory.positions[0].expires_at == 5


def test_absorb_zero_success_probability_fails_and_discards_signal() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_success_probability=0.0,
        storage_lifetime_ticks=5,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)
    photon_subsystem = SubsystemId("photon:s1")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.EMPTY,
        memory_subsystem=None,
        stored_signal=None,
        stored_time=None,
        last_noise_update_time=None,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(photon_subsystem)

    report = memory.reports[-1]
    assert isinstance(report, MemoryAbsorbReport)
    assert report.success is False
    assert report.status == "absorb_failed"
    assert report.memory_subsystem is None
    assert ("occupancy_token", 1) in report.meta
    assert ("absorb_success_probability", 0.0) in report.meta

    timeline.run_until(5)
    assert len(memory.reports) == 1


def test_delayed_absorb_zero_success_probability_clears_absorbing() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=5,
        absorb_success_probability=0.0,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.ABSORBING,
        memory_subsystem=None,
        stored_signal=signal,
        stored_time=None,
        last_noise_update_time=None,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )

    timeline.run_until(5)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.EMPTY,
        memory_subsystem=None,
        stored_signal=None,
        stored_time=None,
        last_noise_update_time=None,
        expires_at=None,
        occupancy_token=1,
        ready_at=5,
    )
    report = memory.reports[-1]
    assert isinstance(report, MemoryAbsorbReport)
    assert report.success is False
    assert report.status == "absorb_failed"


def test_absorb_rejects_occupied_requested_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s1"),
                position=0,
            ),
        ),
        timeline,
    )

    with pytest.raises(ValueError, match="not empty"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id="absorb-2",
                    memory_id="nodeA.mem0",
                    signal=_signal(timeline, "s2"),
                    position=0,
                ),
            ),
            timeline,
        )


def test_requested_absorb_rejects_recovering_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=5,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert memory.positions[0].ready_at == 5
    with pytest.raises(ValueError, match="still recovering"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id="absorb-2",
                    memory_id="nodeA.mem0",
                    signal=_signal(timeline, "s2"),
                    position=0,
                ),
            ),
            timeline,
        )


def test_automatic_absorb_skips_recovering_positions() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        recovery_ticks=5,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline, position=0)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    signal = _absorb_position(memory, timeline, signal_id="s2")

    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert memory.positions[0].ready_at == 5
    assert memory.positions[1].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[1].stored_signal is signal


def test_automatic_absorb_fails_when_all_empty_positions_are_recovering() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=5,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    with pytest.raises(RuntimeError, match="no available memory position"):
        _absorb_position(memory, timeline, signal_id="s2", request_id="absorb-2")


def test_absorb_after_recovery_sets_ready_at_to_absorb_time() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=5,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    signal = _signal(timeline, "s2")

    timeline.schedule(
        Event(
            time=8,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-2",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        )
    )
    timeline.run_until(8)

    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].stored_signal is signal
    assert memory.positions[0].stored_time == 8
    assert memory.positions[0].ready_at == 8


@pytest.mark.parametrize("ready_at", (2, 9))
def test_occupied_position_rejects_absorb_regardless_of_ready_at(
    ready_at: int,
) -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    timeline.run_until(5)
    memory._replace_position(
        0,
        replace(memory.positions[0], ready_at=ready_at),
    )

    with pytest.raises(ValueError, match="not empty"):
        memory.handle_event(
            Event(
                time=timeline.current_time,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id="absorb-2",
                    memory_id="nodeA.mem0",
                    signal=_signal(timeline, "s2"),
                    position=0,
                ),
            ),
            timeline,
        )

    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].ready_at == ready_at


def test_zero_recovery_ticks_allows_immediate_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=0,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    signal = _absorb_position(memory, timeline, signal_id="s2", request_id="absorb-2")

    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].stored_signal is signal
    assert memory.positions[0].ready_at == 0


def test_absorb_nonzero_delay_marks_absorbing_then_completes() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=4,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)
    photon_subsystem = SubsystemId("photon:s1")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    reports_before = _count_reports(memory, MemoryAbsorbReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.ABSORBING,
        stored_signal=signal,
        occupancy_token=1,
    )
    assert timeline.qstate.state_of(photon_subsystem) == signal.state_ref
    assert _count_reports(memory, MemoryAbsorbReport) == reports_before

    with pytest.raises(ValueError, match="not empty"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id="absorb-conflict",
                    memory_id="nodeA.mem0",
                    signal=_signal(timeline, "s2"),
                    position=0,
                ),
            ),
            timeline,
        )
    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.ABSORBING,
        stored_signal=signal,
        occupancy_token=1,
    )

    timeline.run_until(4)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=4,
        last_noise_update_time=4,
        expires_at=None,
        occupancy_token=1,
    )
    assert _count_reports(memory, MemoryAbsorbReport) == reports_before + 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(photon_subsystem)


def test_scheduled_absorb_conflict_measure_and_emit_workflow() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=3,
        measure_delay_ticks=1,
        emit_delay_ticks=2,
        recovery_ticks=5,
    )
    output_sink = QuantumSink()
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, output_sink.input_port, target_action="receive")
    connect_ports(memory.notice_port, notice_sink.input_port, target_action="notice")

    signal = _signal(timeline, state="|0>")
    photon_subsystem = SubsystemId("photon:s1")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    emitted_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)

    timeline.schedule(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id=memory.memory_id,
                signal=signal,
                position=0,
            ),
        )
    )
    timeline.schedule(
        Event(
            time=1,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-too-early",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    )

    with pytest.raises(ValueError, match="not occupied"):
        timeline.run_until(1)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.ABSORBING,
        stored_signal=signal,
        occupancy_token=1,
    )
    assert timeline.qstate.state_of(photon_subsystem) == signal.state_ref
    assert output_sink.received == []

    timeline.run_until(3)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=3,
        last_noise_update_time=3,
        occupancy_token=1,
    )
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(photon_subsystem)

    timeline.schedule(
        Event(
            time=4,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id=memory.memory_id,
                positions=(0,),
                measurement="z",
                destructive=False,
            ),
        )
    )
    timeline.schedule(
        Event(
            time=6,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    )
    timeline.run_until_empty()

    measurement_reports = [
        report
        for report in memory.reports
        if isinstance(report, MemoryMeasurementReport)
    ]
    emit_reports = [
        report for report in memory.reports if isinstance(report, MemoryEmitReport)
    ]

    assert len(measurement_reports) == 1
    assert measurement_reports[0].success is True
    assert measurement_reports[0].detection_report is not None
    assert measurement_reports[0].detection_report.outcome == "0"
    assert len(emit_reports) == 1
    assert emit_reports[0].success is True
    assert len(output_sink.received) == 1
    assert any(
        isinstance(delivery.payload, MemoryAbsorbReport)
        for delivery in notice_sink.received
    )
    assert any(
        isinstance(delivery.payload, MemoryMeasurementReport)
        for delivery in notice_sink.received
    )
    assert any(
        isinstance(delivery.payload, MemoryEmitReport)
        for delivery in notice_sink.received
    )

    output_signal = output_sink.received[0].payload
    assert isinstance(output_signal, Signal)
    assert output_signal.state_ref == signal.state_ref
    assert output_signal.state_targets[0].label == str(emitted_subsystem)
    assert timeline.qstate.state_of(emitted_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert memory.positions[0].ready_at == 13


def test_emit_relabels_memory_to_new_photon_and_sends_signal() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=7,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(
        memory.output_port,
        sink.input_port,
        target_action="receive_signal",
    )
    signal = _signal(timeline)
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)
    assert timeline.qstate.state_of(output_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)

    record = memory.positions[0]
    assert record.status is MemoryPositionStatus.EMPTY
    assert record.memory_subsystem is None
    assert record.stored_signal is None
    assert record.stored_time is None
    assert record.last_noise_update_time is None
    assert record.expires_at is None
    assert record.occupancy_token == 2
    assert record.ready_at == 7

    timeline.run_until(0)
    assert len(sink.received) == 1
    output_signal = sink.received[0].payload
    assert isinstance(output_signal, Signal)
    assert output_signal is not signal
    assert output_signal.signal_kind is SignalKind.PHOTON
    assert output_signal.encoding_scheme is signal.encoding_scheme
    assert output_signal.emission_time == 0
    assert output_signal.origin == "nodeA.mem0"
    assert output_signal.state_ref == signal.state_ref
    assert output_signal.state_targets[0].label == str(output_subsystem)


def test_emit_nonzero_delay_marks_emitting_then_completes() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=5,
        recovery_ticks=7,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    signal = _absorb_position(memory, timeline)
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)
    reports_before = _count_reports(memory, MemoryEmitReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.EMITTING,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    assert sink.received == []
    assert _count_reports(memory, MemoryEmitReport) == reports_before
    with pytest.raises(ValueError, match="not occupied"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_EMIT,
                payload_ref=MemoryEmitRequest(
                    request_id="emit-conflict",
                    memory_id="nodeA.mem0",
                    position=0,
                ),
            ),
            timeline,
        )

    timeline.run_until(5)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.EMPTY,
        memory_subsystem=None,
        stored_signal=None,
        stored_time=None,
        last_noise_update_time=None,
        expires_at=None,
        occupancy_token=2,
        ready_at=12,
    )
    assert _count_reports(memory, MemoryEmitReport) == reports_before + 1
    assert timeline.qstate.state_of(output_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    assert len(sink.received) == 1
    assert isinstance(sink.received[0].payload, Signal)


def test_emit_zero_success_probability_fails_without_output_connection() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_success_probability=0.0,
        recovery_ticks=7,
    )
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline)
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )
    assert memory.emit_counter == 0
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(output_subsystem)

    report = memory.reports[-1]
    assert isinstance(report, MemoryEmitReport)
    assert report.success is False
    assert report.status == "emit_failed"
    assert report.memory_subsystem == memory_subsystem
    assert report.output_signal_id is None
    assert report.output_subsystem is None
    assert ("occupancy_token", 1) in report.meta
    assert ("emit_success_probability", 0.0) in report.meta


def test_delayed_emit_zero_success_probability_restores_occupied() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=5,
        emit_success_probability=0.0,
        recovery_ticks=7,
    )
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline)
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    assert memory.positions[0].status is MemoryPositionStatus.EMITTING

    timeline.run_until(5)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=5,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )
    assert memory.emit_counter == 0
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(output_subsystem)
    report = memory.reports[-1]
    assert isinstance(report, MemoryEmitReport)
    assert report.success is False
    assert report.status == "emit_failed"

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-2",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    assert memory.positions[0].status is MemoryPositionStatus.EMITTING


def test_apply_operator_preserves_exact_position_order() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=2)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-0",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s0", state="|1>"),
                position=0,
            ),
        ),
        timeline,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s1", state="|0>"),
                position=1,
            ),
        ),
        timeline,
    )

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0, 1),
                operator=CNOT,
            ),
        ),
        timeline,
    )

    q0 = memory_subsystem_id("nodeA.mem0", 0)
    q1 = memory_subsystem_id("nodeA.mem0", 1)
    assert timeline.qstate.measure(targets=(q0, q1), basis="z").outcome == (1, 1)


def test_apply_operator_reversed_positions_change_operator_operand_order() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=2)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-0",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s0", state="|1>"),
                position=0,
            ),
        ),
        timeline,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s1", state="|0>"),
                position=1,
            ),
        ),
        timeline,
    )

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(1, 0),
                operator=CNOT,
            ),
        ),
        timeline,
    )

    q0 = memory_subsystem_id("nodeA.mem0", 0)
    q1 = memory_subsystem_id("nodeA.mem0", 1)
    assert timeline.qstate.measure(targets=(q0, q1), basis="z").outcome == (1, 0)


def test_apply_operator_does_not_broadcast_single_qubit_operator() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=2)
    memory.bind(binding_context(timeline))

    for position in (0, 1):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_ABSORB,
                payload_ref=MemoryAbsorbRequest(
                    request_id=f"absorb-{position}",
                    memory_id="nodeA.mem0",
                    signal=_signal(timeline, f"s{position}"),
                    position=position,
                ),
            ),
            timeline,
        )

    with pytest.raises(InvalidOperationError, match="target count"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_APPLY_OPERATOR,
                payload_ref=MemoryApplyOperatorRequest(
                    request_id="apply-1",
                    memory_id="nodeA.mem0",
                    positions=(0, 1),
                    operator=X,
                ),
            ),
            timeline,
        )


def test_apply_operator_passes_custom_operator_to_qstate_boundary() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )

    with pytest.raises(TypeError, match="operation must be Unitary"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_APPLY_OPERATOR,
                payload_ref=MemoryApplyOperatorRequest(
                    request_id="apply-1",
                    memory_id="nodeA.mem0",
                    positions=(0,),
                    operator=object(),
                ),
            ),
            timeline,
        )

    custom_flip = unitary([[0, 1], [1, 0]], name="custom-flip")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-2",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=custom_flip,
            ),
        ),
        timeline,
    )

    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    assert timeline.qstate.measure(targets=(memory_subsystem,), basis="z").label == "1"


def test_apply_operator_report_emits_only_when_notice_port_connected() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    reports_before = len(memory.reports)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=X,
            ),
        ),
        timeline,
    )
    assert len(memory.reports) == reports_before

    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-2",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=X,
            ),
        ),
        timeline,
    )

    timeline.run_until(0)
    assert isinstance(memory.reports[-1], MemoryOperatorReport)
    assert memory.reports[-1].positions == (0,)
    assert any(
        isinstance(delivery.payload, MemoryOperatorReport)
        for delivery in notice_sink.received
    )


def test_apply_operator_nonzero_delay_marks_busy_then_completes() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        operator_delay_ticks=3,
        recovery_ticks=5,
    )
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline)
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    reports_before = _count_reports(memory, MemoryOperatorReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=X,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.APPLYING_OPERATOR,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )
    assert _count_reports(memory, MemoryOperatorReport) == reports_before
    with pytest.raises(ValueError, match="not occupied"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_APPLY_OPERATOR,
                payload_ref=MemoryApplyOperatorRequest(
                    request_id="apply-conflict",
                    memory_id="nodeA.mem0",
                    positions=(0,),
                    operator=X,
                ),
            ),
            timeline,
        )

    timeline.run_until(3)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )
    assert _count_reports(memory, MemoryOperatorReport) == reports_before + 1
    assert len(notice_sink.received) == 1
    assert isinstance(notice_sink.received[0].payload, MemoryOperatorReport)
    assert timeline.qstate.measure(targets=(memory_subsystem,), basis="z").label == "1"


def test_measure_nonzero_delay_marks_measuring_then_completes() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        measure_delay_ticks=2,
    )
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline, state="|1>")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    reports_before = _count_reports(memory, MemoryMeasurementReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                destructive=False,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.MEASURING,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    assert _count_reports(memory, MemoryMeasurementReport) == reports_before
    with pytest.raises(ValueError, match="not occupied"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_MEASURE,
                payload_ref=MemoryMeasureRequest(
                    request_id="measure-conflict",
                    memory_id="nodeA.mem0",
                    positions=(0,),
                    destructive=False,
                ),
            ),
            timeline,
        )

    timeline.run_until(2)

    report = memory.reports[-1]
    assert isinstance(report, MemoryMeasurementReport)
    assert report.detection_report is not None
    assert report.detection_report.outcome == "1"
    assert _count_reports(memory, MemoryMeasurementReport) == reports_before + 1
    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )


def test_delayed_destructive_measure_starts_recovery_at_completion() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        measure_delay_ticks=2,
        recovery_ticks=7,
    )
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline, state="|1>")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                destructive=True,
            ),
        ),
        timeline,
    )

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.MEASURING,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
        ready_at=0,
    )

    timeline.run_until(2)

    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.EMPTY,
        memory_subsystem=None,
        stored_signal=None,
        stored_time=None,
        last_noise_update_time=None,
        expires_at=None,
        occupancy_token=2,
        ready_at=9,
    )
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)


def test_measure_destructive_clears_position_and_wraps_detection_report() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=4,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline, state="|1>")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                measurement="z",
                collapse=True,
                destructive=True,
            ),
        ),
        timeline,
    )

    report = memory.reports[-1]
    assert isinstance(report, MemoryMeasurementReport)
    assert report.positions == (0,)
    assert report.memory_subsystems == (memory_subsystem,)
    assert report.detection_report is not None
    assert report.detection_report.outcome == "1"
    assert report.destructive is True
    assert report.cleared_positions == (0,)
    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert memory.positions[0].occupancy_token == 2
    assert memory.positions[0].ready_at == 4
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)


def test_measure_non_destructive_leaves_position_occupied() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=4,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline, state="|0>")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                measurement="z",
                collapse=True,
                destructive=False,
            ),
        ),
        timeline,
    )

    report = memory.reports[-1]
    assert isinstance(report, MemoryMeasurementReport)
    assert report.destructive is False
    assert report.cleared_positions == ()
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].ready_at == 0
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref


def test_measure_collapse_false_and_non_destructive_preserves_qstate() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, state="|+>"),
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                measurement="z",
                collapse=False,
                destructive=False,
            ),
        ),
        timeline,
    )

    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert timeline.qstate.measure(targets=(memory_subsystem,), basis="x").label == "+"


def test_measure_uses_custom_measurement_callable() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, state="|+>"),
            ),
        ),
        timeline,
    )

    calls: list[int] = []

    def custom_measurement(context, rng=None):
        calls.append(context.time)
        return Measure.basis("x", label="custom-x").choose(context, rng=rng)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                measurement=custom_measurement,
                collapse=True,
                destructive=False,
            ),
        ),
        timeline,
    )

    report = memory.reports[-1]

    assert calls == [0]
    assert isinstance(report, MemoryMeasurementReport)
    assert report.detection_report is not None
    assert report.detection_report.measurement_method == "projective"
    assert report.detection_report.measurement_label == "custom-x"
    assert report.detection_report.outcome == "+"
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED


def test_measure_report_emits_through_notice_port() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
            ),
        ),
        timeline,
    )

    timeline.run_until(0)
    assert any(
        isinstance(delivery.payload, MemoryMeasurementReport)
        for delivery in notice_sink.received
    )


def test_emit_report_can_be_emitted_through_notice_port() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    quantum_sink = QuantumSink()
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(
        memory.output_port,
        quantum_sink.input_port,
        target_action="receive_signal",
    )
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    signal = _signal(timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    timeline.run_until(0)
    assert any(isinstance(report, MemoryEmitReport) for report in memory.reports)
    assert any(
        isinstance(delivery.payload, MemoryEmitReport)
        for delivery in notice_sink.received
    )


def test_discard_removes_stored_subsystem_and_clears_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=6,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-1",
                memory_id="nodeA.mem0",
                position=0,
                reason="manual",
            ),
        ),
        timeline,
    )

    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    record = memory.positions[0]
    assert record.status is MemoryPositionStatus.EMPTY
    assert record.memory_subsystem is None
    assert record.stored_signal is None
    assert record.stored_time is None
    assert record.last_noise_update_time is None
    assert record.expires_at is None
    assert record.occupancy_token == 2
    assert record.ready_at == 6
    assert isinstance(memory.reports[-1], MemoryDiscardReport)
    assert memory.reports[-1].reason == "manual"
    assert ("ready_at", 6) in memory.reports[-1].meta
    assert ("recovery_ticks", 6) in memory.reports[-1].meta


def test_expire_removes_matching_token_and_emits_report() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        recovery_ticks=8,
    )
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    signal = _signal(timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EXPIRE,
            payload_ref=MemoryExpireRequest(
                request_id="expire-1",
                memory_id="nodeA.mem0",
                position=0,
                occupancy_token=1,
            ),
        ),
        timeline,
    )

    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)
    assert memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert memory.positions[0].occupancy_token == 2
    assert memory.positions[0].ready_at == 8
    assert isinstance(memory.reports[-1], MemoryExpireReport)
    timeline.run_until(0)
    assert any(
        isinstance(delivery.payload, MemoryExpireReport)
        for delivery in notice_sink.received
    )


def test_update_meta_updates_only_classical_position_metadata() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    signal = _signal(timeline)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
                meta=(("pair_id", "old"), ("keep", "yes")),
            ),
        ),
        timeline,
    )
    before = memory.positions[0]
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    state_ref = timeline.qstate.state_of(memory_subsystem)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_UPDATE_META,
            payload_ref=MemoryUpdateMetaRequest(
                request_id="meta-1",
                memory_id="nodeA.mem0",
                position=0,
                updates=(("pair_id", "new"), ("swap_round", 1)),
                remove_keys=("pair_id",),
                expected_occupancy_token=before.occupancy_token,
            ),
        ),
        timeline,
    )

    record = memory.positions[0]
    assert record.meta == (("keep", "yes"), ("pair_id", "new"), ("swap_round", 1))
    assert record.status is before.status
    assert record.memory_subsystem == before.memory_subsystem
    assert record.stored_signal is before.stored_signal
    assert record.stored_time == before.stored_time
    assert record.last_noise_update_time == before.last_noise_update_time
    assert record.expires_at == before.expires_at
    assert record.occupancy_token == before.occupancy_token
    assert record.ready_at == before.ready_at
    assert timeline.qstate.state_of(memory_subsystem) == state_ref
    report = memory.reports[-1]
    assert isinstance(report, MemoryMetaUpdateReport)
    assert report.success is True
    assert report.status == "updated"
    assert report.updated_keys == ("pair_id", "swap_round")
    assert report.removed_keys == ("pair_id",)
    timeline.run_until(0)
    assert any(
        isinstance(delivery.payload, MemoryMetaUpdateReport)
        for delivery in notice_sink.received
    )
