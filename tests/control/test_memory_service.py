from __future__ import annotations

from dataclasses import dataclass

import pytest

from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_MEASURE,
    MEMORY_UPDATE_META,
    MemoryAbsorbReport,
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryMeasureRequest,
    MemoryPositionStatus,
    MemoryUpdateMetaRequest,
    QuantumMemory,
    memory_subsystem_id,
)
from simyuj.control import Agent, AgentContext, NodeAgent, SessionRuntime
from simyuj.control.memory import MemoryService
from simyuj.engine import Timeline
from simyuj.network import Network, Node
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import StateNotFoundError, SubsystemId
from simyuj.qstate.ops import X
from simyuj.signal import EncodingScheme, Signal, SignalKind


@dataclass(slots=True)
class MemoryAgent(NodeAgent):
    contexts: list[AgentContext]

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        self.contexts.append(ctx)


@dataclass(slots=True)
class TimerAbsorbAgent(NodeAgent):
    timer_ids: list[str]
    request_ids: list[str]

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.timers is not None
        ctx.timers.set("absorb-qubit", delay=1)

    def on_timer(self, timer, ctx: AgentContext) -> None:
        self.timer_ids.append(timer.timer_id)
        assert ctx.memory is not None
        source_subsystem = SubsystemId("alice:control:incoming:0")
        state_ref = ctx.timeline.qstate.prepare(
            "|0>",
            subsystems=(source_subsystem,),
        )
        signal = Signal(
            id="timer-qubit",
            signal_kind=SignalKind.PHOTON,
            encoding_scheme=EncodingScheme.POLARIZATION,
            emission_time=ctx.timeline.current_time,
            origin=self.agent_id,
            state_ref=state_ref,
            state_targets=(
                SubsystemHandle(
                    label=str(source_subsystem),
                    kind="qubit",
                    index=0,
                    metadata=(("qstate_subsystem", str(source_subsystem)),),
                ),
            ),
        )

        event = ctx.memory.absorb(
            "mem",
            signal,
            position=0,
            request_id="timer-absorb",
        )
        self.request_ids.append(event.payload_ref.request_id)


class CountingMemory(QuantumMemory):
    def __init__(self) -> None:
        super().__init__(memory_id="physical-memory", num_positions=2)
        self.handle_calls = 0

    def handle_event(self, event, timeline) -> None:
        self.handle_calls += 1
        super().handle_event(event, timeline)


def signal() -> Signal:
    return Signal(
        id=1,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="source",
    )


def service_with_memory() -> tuple[MemoryService, CountingMemory, Timeline]:
    timeline = Timeline(master_seed=1)
    node = Node("alice")
    qmem = CountingMemory()
    node.add_device("mem", qmem)
    node.add_device("plain", object())
    service = MemoryService(
        node=node,
        timeline=timeline,
        session_id="session-1",
        source_agent=Agent(agent_id="alice-agent"),
    )
    return service, qmem, timeline


def test_absorb_schedules_memory_absorb_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.absorb("mem", signal(), position=None, request_id="absorb-1")

    assert event.target_ref is qmem
    assert event.action == MEMORY_ABSORB
    assert isinstance(event.payload_ref, MemoryAbsorbRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.position is None
    assert event.payload_ref.request_id == "absorb-1"


def test_emit_schedules_memory_emit_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.emit("mem", 1, request_id="emit-1")

    assert event.target_ref is qmem
    assert event.action == MEMORY_EMIT
    assert isinstance(event.payload_ref, MemoryEmitRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.position == 1


def test_measure_schedules_memory_measure_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.measure("mem", (0,), measurement="x", request_id="measure-1")

    assert event.target_ref is qmem
    assert event.action == MEMORY_MEASURE
    assert isinstance(event.payload_ref, MemoryMeasureRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.measurement == "x"


def test_apply_schedules_memory_apply_operator_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.apply("mem", (0,), X, request_id="apply-1")

    assert event.target_ref is qmem
    assert event.action == MEMORY_APPLY_OPERATOR
    assert isinstance(event.payload_ref, MemoryApplyOperatorRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.operator is X


def test_discard_schedules_memory_discard_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.discard("mem", 0, reason="test", request_id="discard-1")

    assert event.target_ref is qmem
    assert event.action == MEMORY_DISCARD
    assert isinstance(event.payload_ref, MemoryDiscardRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.reason == "test"


def test_update_meta_schedules_memory_update_meta_to_quantum_memory() -> None:
    service, qmem, _ = service_with_memory()

    event = service.update_meta(
        "mem",
        0,
        updates=(("basis", "z"),),
        remove_keys=("old",),
        expected_occupancy_token=3,
        request_id="meta-1",
    )

    assert event.target_ref is qmem
    assert event.action == MEMORY_UPDATE_META
    assert isinstance(event.payload_ref, MemoryUpdateMetaRequest)
    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.updates == (("basis", "z"),)


def test_all_methods_use_physical_memory_id_not_node_alias() -> None:
    service, qmem, _ = service_with_memory()

    event = service.emit("mem", 0)

    assert event.payload_ref.memory_id == qmem.memory_id
    assert event.payload_ref.memory_id != "mem"
    assert event.meta["memory_alias"] == "mem"


def test_service_rejects_non_memory_device_names() -> None:
    service, _, _ = service_with_memory()

    with pytest.raises(TypeError, match="not QuantumMemory"):
        service.emit("plain", 0)


def test_service_never_calls_memory_handle_event_when_scheduling() -> None:
    service, qmem, timeline = service_with_memory()

    service.absorb("mem", signal())
    service.emit("mem", 0)

    assert qmem.handle_calls == 0
    assert timeline.events_scheduled == 2


def test_runtime_context_adds_memory_service_for_node_agent() -> None:
    timeline = Timeline(master_seed=1)
    network = Network()
    node = Node("alice")
    qmem = CountingMemory()
    node.add_device("mem", qmem)
    network.add_node(node)
    agent = MemoryAgent(agent_id="alice-agent", node_id="alice", contexts=[])
    node.add_agent(agent)
    runtime = SessionRuntime(timeline=timeline, network=network)

    runtime.run()

    assert agent.contexts[0].memory is not None
    event = agent.contexts[0].memory.emit("mem", 0)
    assert event.target_ref is qmem


def test_runtime_timer_schedules_memory_absorb_workflow() -> None:
    timeline = Timeline(master_seed=1)
    network = Network()
    node = Node("alice")
    qmem = CountingMemory()
    node.add_device("mem", qmem)
    network.add_node(node)
    agent = TimerAbsorbAgent(
        agent_id="alice-agent",
        node_id="alice",
        timer_ids=[],
        request_ids=[],
    )
    node.add_agent(agent)
    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        session_id="session-1",
    )

    runtime.run()

    memory_subsystem = memory_subsystem_id(qmem.memory_id, 0)
    assert agent.timer_ids == ["absorb-qubit"]
    assert agent.request_ids == ["timer-absorb"]
    assert qmem.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert qmem.positions[0].memory_subsystem == memory_subsystem
    assert timeline.qstate.state_of(memory_subsystem) is not None
    with pytest.raises(StateNotFoundError, match="not owned"):
        timeline.qstate.state_of(SubsystemId("alice:control:incoming:0"))

    reports = [
        report for report in qmem.reports if isinstance(report, MemoryAbsorbReport)
    ]
    assert len(reports) == 1
    assert reports[0].success is True
    assert reports[0].session_id == "session-1"
    assert reports[0].memory_subsystem == memory_subsystem
    assert reports[0].input_signal_id == "timer-qubit"
    assert qmem.handle_calls > 0
    assert timeline.events_executed >= 3
