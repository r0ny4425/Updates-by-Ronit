"""Internal QuantumMemory regressions for delayed-operation invariants.

These tests intentionally exercise private completion payloads/actions, direct
component dispatch, and stale-token rollback paths. They are not public memory
workflow acceptance tests.
"""

from __future__ import annotations

import pytest

from simyuj.components import connect_ports
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
    MemoryDiscardRequest,
    MemoryEmitReport,
    MemoryEmitRequest,
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
from simyuj.components.memories.quantum_memory import (
    _MEMORY_ABSORB_COMPLETE,
    _MEMORY_APPLY_OPERATOR_COMPLETE,
    _MEMORY_EMIT_COMPLETE,
    _MEMORY_MEASURE_COMPLETE,
    _AbsorbCompletion,
    _ApplyOperatorCompletion,
    _EmitCompletion,
    _MeasureCompletion,
)
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.qstate import StateNotFoundError, SubsystemId
from simyuj.qstate.ops import CNOT, X
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.components.memories._quantum_memory_support import (
    NoticeSink,
    QuantumSink,
    RecordingNoiseModel,
    _absorb_position,
    _assert_position_fields,
    _conflict_event,
    _count_reports,
    _signal,
    _start_busy_position,
)
from tests.support.binding import binding_context


def test_quantum_memory_logs_stale_completion_skip_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    scheduled = timeline.schedule(
        Event(
            time=0,
            target_ref=memory,
            action=_MEMORY_EMIT_COMPLETE,
            payload_ref=_EmitCompletion(
                request=MemoryEmitRequest(
                    request_id="emit-1",
                    memory_id=memory.memory_id,
                    position=0,
                ),
                occupancy_token=99,
            ),
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.stale_skip"
    )

    assert record.level is LogLevel.TRACE
    assert record.event_id == scheduled.event_id
    assert record.action == _MEMORY_EMIT_COMPLETE
    assert dict(record.meta) == {
        "memory_id": "nodeA.mem0",
        "operation": "emit",
        "positions": (0,),
        "occupancy_tokens": (99,),
        "expected_status": "emitting",
        "reason": "stale_completion",
    }


@pytest.mark.parametrize(
    "busy_kind",
    ("absorbing", "emitting", "applying", "measuring"),
)
@pytest.mark.parametrize(
    "conflict_action",
    (
        MEMORY_ABSORB,
        MEMORY_EMIT,
        MEMORY_APPLY_OPERATOR,
        MEMORY_MEASURE,
        MEMORY_DISCARD,
        MEMORY_EXPIRE,
    ),
)
def test_busy_position_rejects_conflicting_operation_without_mutation(
    busy_kind: str,
    conflict_action: str,
) -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=10 if busy_kind == "absorbing" else 0,
        emit_delay_ticks=10 if busy_kind == "emitting" else 0,
        operator_delay_ticks=10 if busy_kind == "applying" else 0,
        measure_delay_ticks=10 if busy_kind == "measuring" else 0,
    )
    memory.bind(binding_context(timeline))
    owner_subsystem, state_ref, absent_subsystem = _start_busy_position(
        memory,
        timeline,
        busy_kind,
    )
    event = _conflict_event(memory, timeline, conflict_action)
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)

    if conflict_action == MEMORY_EXPIRE:
        memory.handle_event(event, timeline)
    else:
        with pytest.raises(ValueError, match="not empty|not occupied"):
            memory.handle_event(event, timeline)

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert timeline.qstate.state_of(owner_subsystem) == state_ref
    if absent_subsystem is not None:
        with pytest.raises(StateNotFoundError, match="not owned"):
            timeline.qstate.state_of(absent_subsystem)


@pytest.mark.parametrize(
    "busy_kind",
    ("absorbing", "emitting", "applying", "measuring"),
)
def test_expiry_ignores_busy_position_even_with_matching_token(
    busy_kind: str,
) -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=10 if busy_kind == "absorbing" else 0,
        emit_delay_ticks=10 if busy_kind == "emitting" else 0,
        operator_delay_ticks=10 if busy_kind == "applying" else 0,
        measure_delay_ticks=10 if busy_kind == "measuring" else 0,
    )
    memory.bind(binding_context(timeline))
    owner_subsystem, state_ref, absent_subsystem = _start_busy_position(
        memory,
        timeline,
        busy_kind,
    )
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_EXPIRE,
            payload_ref=MemoryExpireRequest(
                request_id="expire-busy",
                memory_id=memory.memory_id,
                position=0,
                occupancy_token=before_record.occupancy_token,
            ),
        ),
        timeline,
    )

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert timeline.qstate.state_of(owner_subsystem) == state_ref
    if absent_subsystem is not None:
        with pytest.raises(StateNotFoundError, match="not owned"):
            timeline.qstate.state_of(absent_subsystem)


def test_absorb_completion_ignores_stale_token_and_wrong_status() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=4,
    )
    memory.bind(binding_context(timeline))
    signal = _signal(timeline)
    request = MemoryAbsorbRequest(
        request_id="absorb-1",
        memory_id="nodeA.mem0",
        signal=signal,
    )
    photon_subsystem = SubsystemId("photon:s1")
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=request,
        ),
        timeline,
    )
    reports_before = _count_reports(memory, MemoryAbsorbReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=_MEMORY_ABSORB_COMPLETE,
            payload_ref=_AbsorbCompletion(
                request=request,
                position=0,
                occupancy_token=99,
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
    assert _count_reports(memory, MemoryAbsorbReport) == reports_before
    assert timeline.qstate.state_of(photon_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)

    timeline.run_until(4)
    assert _count_reports(memory, MemoryAbsorbReport) == reports_before + 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_ABSORB_COMPLETE,
            payload_ref=_AbsorbCompletion(
                request=request,
                position=0,
                occupancy_token=1,
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
        stored_time=4,
        last_noise_update_time=4,
        expires_at=None,
        occupancy_token=1,
    )
    assert _count_reports(memory, MemoryAbsorbReport) == reports_before + 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref


def test_absorb_old_completion_is_ignored_after_public_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        absorb_delay_ticks=4,
    )
    memory.bind(binding_context(timeline))
    old_signal = _signal(timeline, "old")
    old_request = MemoryAbsorbRequest(
        request_id="absorb-old",
        memory_id="nodeA.mem0",
        signal=old_signal,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=old_request,
        ),
        timeline,
    )
    old_completion = _AbsorbCompletion(
        request=old_request,
        position=0,
        occupancy_token=1,
    )
    timeline.run_until(4)
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-old",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    new_signal = _signal(timeline, "new")
    new_request = MemoryAbsorbRequest(
        request_id="absorb-new",
        memory_id="nodeA.mem0",
        signal=new_signal,
    )
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=new_request,
        ),
        timeline,
    )
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)
    new_photon_subsystem = SubsystemId("photon:new")

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_ABSORB_COMPLETE,
            payload_ref=old_completion,
        ),
        timeline,
    )

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert timeline.qstate.state_of(new_photon_subsystem) == new_signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(memory_subsystem)


def test_emit_completion_ignores_stale_token_and_wrong_status() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=5,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    signal = _absorb_position(memory, timeline)
    request = MemoryEmitRequest(
        request_id="emit-1",
        memory_id="nodeA.mem0",
        position=0,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=request,
        ),
        timeline,
    )
    reports_before = _count_reports(memory, MemoryEmitReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=_MEMORY_EMIT_COMPLETE,
            payload_ref=_EmitCompletion(request=request, occupancy_token=99),
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
    assert _count_reports(memory, MemoryEmitReport) == reports_before
    assert sink.received == []
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(output_subsystem)

    timeline.run_until(5)
    assert _count_reports(memory, MemoryEmitReport) == reports_before + 1
    assert len(sink.received) == 1
    assert timeline.qstate.state_of(output_subsystem) == signal.state_ref

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_EMIT_COMPLETE,
            payload_ref=_EmitCompletion(request=request, occupancy_token=1),
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
        occupancy_token=2,
        ready_at=5,
    )
    assert _count_reports(memory, MemoryEmitReport) == reports_before + 1
    assert len(sink.received) == 1
    assert timeline.qstate.state_of(output_subsystem) == signal.state_ref


def test_emit_old_completion_is_ignored_after_public_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=5,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    old_signal = _absorb_position(memory, timeline, signal_id="old")
    old_request = MemoryEmitRequest(
        request_id="emit-old",
        memory_id="nodeA.mem0",
        position=0,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=old_request,
        ),
        timeline,
    )
    old_completion = _EmitCompletion(request=old_request, occupancy_token=1)
    timeline.run_until(5)
    _ = old_signal
    new_signal = _absorb_position(
        memory,
        timeline,
        signal_id="new",
        request_id="absorb-new",
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)
    before_deliveries = tuple(sink.received)

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_EMIT_COMPLETE,
            payload_ref=old_completion,
        ),
        timeline,
    )

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert tuple(sink.received) == before_deliveries
    assert timeline.qstate.state_of(memory_subsystem) == new_signal.state_ref


def test_apply_operator_completion_ignores_stale_token_and_wrong_status() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        operator_delay_ticks=3,
    )
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline)
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    request = MemoryApplyOperatorRequest(
        request_id="apply-1",
        memory_id="nodeA.mem0",
        positions=(0,),
        operator=X,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=request,
        ),
        timeline,
    )
    reports_before = _count_reports(memory, MemoryOperatorReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=_MEMORY_APPLY_OPERATOR_COMPLETE,
            payload_ref=_ApplyOperatorCompletion(
                request=request,
                occupancy_tokens=(99,),
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
    )
    assert _count_reports(memory, MemoryOperatorReport) == reports_before
    assert notice_sink.received == []
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref

    timeline.run_until(3)
    assert _count_reports(memory, MemoryOperatorReport) == reports_before + 1
    assert len(notice_sink.received) == 1
    assert timeline.qstate.measure(targets=(memory_subsystem,), basis="z").label == "1"

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_APPLY_OPERATOR_COMPLETE,
            payload_ref=_ApplyOperatorCompletion(
                request=request,
                occupancy_tokens=(1,),
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
    )
    assert _count_reports(memory, MemoryOperatorReport) == reports_before + 1
    assert len(notice_sink.received) == 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref


def test_apply_operator_old_completion_is_ignored_after_public_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        operator_delay_ticks=3,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline, signal_id="old")
    old_request = MemoryApplyOperatorRequest(
        request_id="apply-old",
        memory_id="nodeA.mem0",
        positions=(0,),
        operator=X,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=old_request,
        ),
        timeline,
    )
    old_completion = _ApplyOperatorCompletion(
        request=old_request,
        occupancy_tokens=(1,),
    )
    timeline.run_until(3)
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-old",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    new_signal = _absorb_position(
        memory,
        timeline,
        signal_id="new",
        request_id="absorb-new",
    )
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_APPLY_OPERATOR_COMPLETE,
            payload_ref=old_completion,
        ),
        timeline,
    )

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert timeline.qstate.state_of(memory_subsystem) == new_signal.state_ref


def test_apply_operator_stale_completion_is_ignored_after_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        operator_delay_ticks=5,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s1"),
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
                positions=(0,),
                operator=X,
            ),
        ),
        timeline,
    )
    old_memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    old_record = memory.positions[0]
    timeline.qstate.discard(targets=(old_memory_subsystem,))
    memory._clear_position_after_quantum_removal(
        position=0,
        timeline=timeline,
        record=old_record,
        meta=(("last_operation", "test_clear"),),
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-2",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s2"),
            ),
        ),
        timeline,
    )

    timeline.run_until(5)

    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].occupancy_token == 3
    assert timeline.qstate.measure(targets=(memory_subsystem,), basis="z").label == "0"


def test_apply_operator_partial_busy_conflict_is_atomic() -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        noise_models=noise_model,
        emit_delay_ticks=10,
    )
    sink = QuantumSink()
    notice_sink = NoticeSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    signal0 = _absorb_position(
        memory,
        timeline,
        position=0,
        signal_id="s0",
        state="|0>",
        request_id="absorb-0",
    )
    signal1 = _absorb_position(
        memory,
        timeline,
        position=1,
        signal_id="s1",
        state="|1>",
        request_id="absorb-1",
    )
    connect_ports(
        memory.notice_port,
        notice_sink.input_port,
        target_action="memory_notice",
    )
    q0 = memory_subsystem_id("nodeA.mem0", 0)
    q1 = memory_subsystem_id("nodeA.mem0", 1)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=1,
            ),
        ),
        timeline,
    )
    _assert_position_fields(
        memory,
        1,
        status=MemoryPositionStatus.EMITTING,
        memory_subsystem=q1,
        stored_signal=signal1,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )

    timeline.schedule(
        Event(
            time=3,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0, 1),
                operator=CNOT,
            ),
        )
    )
    with pytest.raises(ValueError, match="not occupied"):
        timeline.run_until(3)

    assert noise_model.durations == []
    assert _count_reports(memory, MemoryOperatorReport) == 0
    assert notice_sink.received == []
    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=q0,
        stored_signal=signal0,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    _assert_position_fields(
        memory,
        1,
        status=MemoryPositionStatus.EMITTING,
        memory_subsystem=q1,
        stored_signal=signal1,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    assert timeline.qstate.state_of(q0) == signal0.state_ref
    assert timeline.qstate.state_of(q1) == signal1.state_ref


def test_measure_completion_ignores_stale_token_and_wrong_status() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        measure_delay_ticks=2,
    )
    memory.bind(binding_context(timeline))
    signal = _absorb_position(memory, timeline, state="|1>")
    request = MemoryMeasureRequest(
        request_id="measure-1",
        memory_id="nodeA.mem0",
        positions=(0,),
        destructive=False,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=request,
        ),
        timeline,
    )
    reports_before = _count_reports(memory, MemoryMeasurementReport)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=_MEMORY_MEASURE_COMPLETE,
            payload_ref=_MeasureCompletion(
                request=request,
                occupancy_tokens=(99,),
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
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref

    timeline.run_until(2)
    assert _count_reports(memory, MemoryMeasurementReport) == reports_before + 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_MEASURE_COMPLETE,
            payload_ref=_MeasureCompletion(
                request=request,
                occupancy_tokens=(1,),
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
    )
    assert _count_reports(memory, MemoryMeasurementReport) == reports_before + 1
    assert timeline.qstate.state_of(memory_subsystem) == signal.state_ref


def test_measure_old_completion_is_ignored_after_public_reuse() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        measure_delay_ticks=2,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline, signal_id="old", state="|1>")
    old_request = MemoryMeasureRequest(
        request_id="measure-old",
        memory_id="nodeA.mem0",
        positions=(0,),
        destructive=False,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=old_request,
        ),
        timeline,
    )
    old_completion = _MeasureCompletion(
        request=old_request,
        occupancy_tokens=(1,),
    )
    timeline.run_until(2)
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_DISCARD,
            payload_ref=MemoryDiscardRequest(
                request_id="discard-old",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    new_signal = _absorb_position(
        memory,
        timeline,
        signal_id="new",
        request_id="absorb-new",
    )
    before_record = memory.positions[0]
    before_reports = tuple(memory.reports)

    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=_MEMORY_MEASURE_COMPLETE,
            payload_ref=old_completion,
        ),
        timeline,
    )

    assert memory.positions[0] == before_record
    assert tuple(memory.reports) == before_reports
    assert timeline.qstate.state_of(memory_subsystem) == new_signal.state_ref


def test_measure_partial_busy_conflict_is_atomic() -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        noise_models=noise_model,
        emit_delay_ticks=10,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    signal0 = _absorb_position(
        memory,
        timeline,
        position=0,
        signal_id="s0",
        state="|0>",
        request_id="absorb-0",
    )
    signal1 = _absorb_position(
        memory,
        timeline,
        position=1,
        signal_id="s1",
        state="|1>",
        request_id="absorb-1",
    )
    q0 = memory_subsystem_id("nodeA.mem0", 0)
    q1 = memory_subsystem_id("nodeA.mem0", 1)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=1,
            ),
        ),
        timeline,
    )
    _assert_position_fields(
        memory,
        1,
        status=MemoryPositionStatus.EMITTING,
        memory_subsystem=q1,
        stored_signal=signal1,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )

    timeline.schedule(
        Event(
            time=3,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0, 1),
                destructive=False,
            ),
        )
    )
    with pytest.raises(ValueError, match="not occupied"):
        timeline.run_until(3)

    assert noise_model.durations == []
    assert _count_reports(memory, MemoryMeasurementReport) == 0
    _assert_position_fields(
        memory,
        0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=q0,
        stored_signal=signal0,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    _assert_position_fields(
        memory,
        1,
        status=MemoryPositionStatus.EMITTING,
        memory_subsystem=q1,
        stored_signal=signal1,
        stored_time=0,
        last_noise_update_time=0,
        expires_at=None,
        occupancy_token=1,
    )
    assert timeline.qstate.state_of(q0) == signal0.state_ref
    assert timeline.qstate.state_of(q1) == signal1.state_ref


def test_stale_expiry_token_does_not_clear_reused_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    quantum_sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, quantum_sink.input_port, target_action="receive")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s1"),
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
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-2",
                memory_id="nodeA.mem0",
                signal=_signal(timeline, "s2"),
            ),
        ),
        timeline,
    )

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EXPIRE,
            payload_ref=MemoryExpireRequest(
                request_id="expire-stale",
                memory_id="nodeA.mem0",
                position=0,
                occupancy_token=1,
            ),
        ),
        timeline,
    )

    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    stored_signal = memory.positions[0].stored_signal
    assert stored_signal is not None
    assert memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert memory.positions[0].occupancy_token == 3
    assert timeline.qstate.state_of(memory_subsystem) == stored_signal.state_ref


def test_update_meta_stale_token_reports_failure_without_mutation() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline)
    before = memory.positions[0]

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_UPDATE_META,
            payload_ref=MemoryUpdateMetaRequest(
                request_id="meta-1",
                memory_id="nodeA.mem0",
                position=0,
                updates=(("pair_id", "new"),),
                expected_occupancy_token=before.occupancy_token + 1,
            ),
        ),
        timeline,
    )

    assert memory.positions[0] == before
    report = memory.reports[-1]
    assert isinstance(report, MemoryMetaUpdateReport)
    assert report.success is False
    assert report.status == "stale_occupancy_token"
    assert report.updated_keys == ()


def test_update_meta_empty_or_busy_position_reports_failure() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        absorb_delay_ticks=5,
    )
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
                position=0,
            ),
        ),
        timeline,
    )
    busy_before = memory.positions[0]
    empty_before = memory.positions[1]

    for position in (0, 1):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_UPDATE_META,
                payload_ref=MemoryUpdateMetaRequest(
                    request_id=f"meta-{position}",
                    memory_id="nodeA.mem0",
                    position=position,
                    updates=(("pair_id", "new"),),
                ),
            ),
            timeline,
        )

    assert memory.positions[0] == busy_before
    assert memory.positions[1] == empty_before
    assert isinstance(memory.reports[-2], MemoryMetaUpdateReport)
    assert memory.reports[-2].success is False
    assert memory.reports[-2].status == "not_occupied:absorbing"
    assert isinstance(memory.reports[-1], MemoryMetaUpdateReport)
    assert memory.reports[-1].success is False
    assert memory.reports[-1].status == "not_occupied:empty"
