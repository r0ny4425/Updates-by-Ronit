from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components.memories import (
    MEMORY_ABSORB,
    MemoryAbsorbReport,
    MemoryAbsorbRequest,
    MemoryPositionStatus,
    QuantumMemory,
    memory_subsystem_id,
)
from simyuj.components.ports import PortKind
from simyuj.control import AGENT_REPORT, Agent, AgentContext, SessionRuntime
from simyuj.control.resources import ResourceService
from simyuj.engine import Event, Timeline
from simyuj.entanglement import EntangledPairRecord, EntangledPairRegistry
from simyuj.network import Network, Node
from simyuj.network.routing import Route
from simyuj.network.topology import TopologyEdge
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.resources.manager import ResourceManager, UnauthorizedError
from simyuj.resources.memory import MemoryRef, MemorySlotState
from simyuj.signal import EncodingScheme, Signal, SignalKind


@dataclass(slots=True)
class ResourceAgent(Agent):
    contexts: list[AgentContext] = field(default_factory=list)

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        self.contexts.append(ctx)


@dataclass(slots=True)
class ResourcePairWorkflowAgent(Agent):
    alice_memory: QuantumMemory
    bob_memory: QuantumMemory
    reservation_id: str | None = None
    reservation_refs: tuple[tuple[str, str, int], ...] = ()
    scheduled_request_ids: list[str] = field(default_factory=list)
    absorb_reports: list[MemoryAbsorbReport] = field(default_factory=list)
    occupied_refs: list[tuple[str, str, int]] = field(default_factory=list)
    consumed_refs: list[tuple[str, str, int]] = field(default_factory=list)
    available_pair_ids: tuple[str, ...] = ()
    registered_pair_state: str | None = None
    reserved_pair_state: str | None = None
    consumed_pair_state: str | None = None

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.resources is not None

        reservation = ctx.resources.reserve_memories(
            ctx.timeline.current_time,
            {"alice": 1, "bob": 1},
            reservation_id="reservation:alice-bob",
            created_at=ctx.timeline.current_time,
            metadata=(("workflow", "resource-pair-service"),),
        )
        committed = ctx.resources.commit(reservation.reservation_id)
        self.reservation_id = committed.reservation_id
        self.reservation_refs = committed.memory_ref_keys

        for node_id, memory, state in (
            ("alice", self.alice_memory, "|0>"),
            ("bob", self.bob_memory, "|1>"),
        ):
            request_id = f"absorb:{node_id}"
            ctx.timeline.schedule(
                Event(
                    time=ctx.timeline.current_time,
                    target_ref=memory,
                    action=MEMORY_ABSORB,
                    payload_ref=MemoryAbsorbRequest(
                        request_id=request_id,
                        memory_id=memory.memory_id,
                        signal=self._signal(ctx, node_id=node_id, state=state),
                        position=0,
                        session_id=ctx.session_id,
                    ),
                    source=self,
                    subsystem_id="control",
                    meta={
                        "session_id": ctx.session_id,
                        "agent_id": self.agent_id,
                        "memory_id": memory.memory_id,
                        "request_id": request_id,
                    },
                )
            )
            self.scheduled_request_ids.append(request_id)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        if not isinstance(report, MemoryAbsorbReport):
            return

        assert ctx.resources is not None
        self.absorb_reports.append(report)

        memory_ref = self._memory_ref_for_report(report)
        occupied = ctx.resources.mark_absorb_report(report, memory_ref)
        self.occupied_refs.append(occupied.ref.key)

        if len(self.absorb_reports) == 2 and self.consumed_pair_state is None:
            self._register_and_consume_pair(ctx)

    def _register_and_consume_pair(self, ctx: AgentContext) -> None:
        assert ctx.pairs is not None
        assert ctx.resources is not None

        left = MemoryRef("alice", "mem", 0)
        right = MemoryRef("bob", "mem", 0)
        pair = ctx.pairs.register(
            EntangledPairRecord(
                pair_id="pair:alice-bob:service",
                left=left,
                right=right,
                fidelity=1.0,
                created_at=ctx.timeline.current_time,
                generation_link_id="scheduled-memory-absorbs",
                metadata=(
                    ("registered_by", self.agent_id),
                    (
                        "absorb_reports",
                        tuple(sorted(item.report_id for item in self.absorb_reports)),
                    ),
                ),
            )
        )
        self.registered_pair_state = pair.state.value
        self.available_pair_ids = tuple(
            record.pair_id for record in ctx.pairs.available_between("alice", "bob")
        )
        self.reserved_pair_state = ctx.pairs.reserve(pair.pair_id).state.value
        self.consumed_pair_state = ctx.pairs.consume(pair.pair_id).state.value

        for memory_ref in (left, right):
            consumed = ctx.resources.mark_consumed(memory_ref)
            self.consumed_refs.append(consumed.ref.key)

    def _memory_ref_for_report(self, report: MemoryAbsorbReport) -> MemoryRef:
        if report.memory_id == self.alice_memory.memory_id:
            return MemoryRef("alice", "mem", report.position)
        if report.memory_id == self.bob_memory.memory_id:
            return MemoryRef("bob", "mem", report.position)
        raise AssertionError(f"unexpected memory report {report.memory_id!r}")

    def _signal(self, ctx: AgentContext, *, node_id: str, state: str) -> Signal:
        subsystem = SubsystemId(f"{node_id}:controller:incoming:0")
        state_ref = ctx.timeline.qstate.prepare(state, subsystems=(subsystem,))
        return Signal(
            id=f"{node_id}:workflow-qubit",
            signal_kind=SignalKind.PHOTON,
            encoding_scheme=EncodingScheme.POLARIZATION,
            emission_time=ctx.timeline.current_time,
            origin=self.agent_id,
            state_ref=state_ref,
            state_targets=(
                SubsystemHandle(
                    label=str(subsystem),
                    kind="qubit",
                    index=0,
                    metadata=(("qstate_subsystem", str(subsystem)),),
                ),
            ),
        )


def manager_with_slots() -> tuple[ResourceManager, MemoryRef, MemoryRef]:
    manager = ResourceManager()
    alice_ref = manager.register_memory("alice", "mem", num_positions=1)[0]
    bob_ref = manager.register_memory("bob", "mem", num_positions=1)[0]
    return manager, alice_ref, bob_ref


def test_query_and_reserve_boundary_uses_owner_filters_and_targets() -> None:
    manager = ResourceManager()
    alice_ref = manager.register_memory("alice", "mem", num_positions=1)[0]
    relay_ref = manager.register_memory(
        "relay",
        "qmem",
        num_positions=1,
        metadata=(("link_id", "link-a"),),
    )[0]
    qmem_a = manager.register_memory("target", "qmem_a", num_positions=2)
    qmem_b = manager.register_memory("target", "qmem_b", num_positions=1)
    service = ResourceService(manager, owner_agent_id="agent")

    assert service.available_memories(10, "alice") == (alice_ref,)
    assert service.available_memories(10, "relay", link_id="link-a") == (relay_ref,)
    assert service.available_memories(10, "relay", link_id="link-b") == ()
    reservation = service.reserve_memories(10, {"alice": 1}, reservation_id="r1")
    assert reservation.owner == "agent"
    assert reservation.memory_refs == (alice_ref,)
    targeted = service.reserve_memories(
        10,
        {"target": {"qmem_b": 1, "qmem_a": 1}},
        reservation_id="r-targeted",
    )
    assert targeted.owner == "agent"
    assert targeted.memory_refs == (qmem_a[0], qmem_b[0])


def test_resource_service_rejects_other_agent_reservation_lifecycle() -> None:
    manager, _, _ = manager_with_slots()
    owner = ResourceService(manager, owner_agent_id="owner")
    intruder = ResourceService(manager, owner_agent_id="intruder")
    reservation = owner.reserve_memories(10, {"alice": 1}, reservation_id="r1")

    with pytest.raises(UnauthorizedError, match="does not own reservation"):
        intruder.commit(reservation.reservation_id)

    committed = owner.commit(reservation.reservation_id)
    with pytest.raises(UnauthorizedError, match="does not own reservation"):
        intruder.release(committed.reservation_id)


def test_reserve_for_route_delegates_to_route_helper() -> None:
    manager, alice_ref, bob_ref = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")
    edge = TopologyEdge(
        link_id="link",
        source_node_id="alice",
        target_node_id="bob",
        port_kind=PortKind.QUANTUM,
    )
    route = Route("alice", "bob", (edge,))

    reservation = service.reserve_for_route(
        10,
        route,
        node_requirements=lambda node, idx, length: 1 if idx in (0, length - 1) else 0,
        reservation_id="route-r1",
    )

    assert reservation.owner == "agent"
    assert reservation.memory_refs == (alice_ref, bob_ref)


def test_commit_and_release_delegate() -> None:
    manager, alice_ref, _ = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")
    reservation = service.reserve_memories(10, {"alice": 1}, reservation_id="r1")

    committed = service.commit(reservation.reservation_id)
    released = service.release(committed.reservation_id)

    assert committed.state.value == "committed"
    assert released.state.value == "released"
    assert manager.get_slot(alice_ref).state is MemorySlotState.FREE


def test_mark_lifecycle_methods_delegate() -> None:
    manager, alice_ref, bob_ref = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    assert service.mark_occupied(alice_ref).state is MemorySlotState.OCCUPIED
    assert service.mark_consumed(alice_ref).state is MemorySlotState.CONSUMED
    assert service.mark_failed(alice_ref).state is MemorySlotState.FAILED
    assert service.mark_free(alice_ref).state is MemorySlotState.FREE

    service.reserve_memories(10, {"bob": 1}, reservation_id="r2")
    assert service.mark_expired(bob_ref).state is MemorySlotState.EXPIRED


def test_mark_absorb_report_rejects_wrong_physical_memory_id() -> None:
    manager = ResourceManager()
    alice_ref = manager.register_memory(
        "alice",
        "mem",
        num_positions=1,
        metadata=(("memory_id", "alice.mem"),),
    )[0]
    service = ResourceService(manager, owner_agent_id="agent")
    report = MemoryAbsorbReport(
        report_id="report:wrong-memory",
        memory_id="bob.mem",
        time=0,
        success=True,
        position=0,
        input_signal_id="signal:0",
        memory_subsystem=memory_subsystem_id("bob.mem", 0),
        status="occupied",
    )

    with pytest.raises(ValueError, match="memory_id"):
        service.mark_absorb_report(report, alice_ref)


def test_mark_absorb_report_allows_matching_or_untracked_memory_id() -> None:
    manager = ResourceManager()
    tracked_ref = manager.register_memory(
        "alice",
        "mem",
        num_positions=1,
        metadata=(("memory_id", "alice.mem"),),
    )[0]
    untracked_ref = manager.register_memory("bob", "mem", num_positions=1)[0]
    service = ResourceService(manager, owner_agent_id="agent")
    tracked_report = MemoryAbsorbReport(
        report_id="report:tracked",
        memory_id="alice.mem",
        time=0,
        success=True,
        position=0,
        input_signal_id="signal:alice",
        memory_subsystem=memory_subsystem_id("alice.mem", 0),
        status="occupied",
    )
    untracked_report = MemoryAbsorbReport(
        report_id="report:untracked",
        memory_id="bob.mem",
        time=0,
        success=True,
        position=0,
        input_signal_id="signal:bob",
        memory_subsystem=memory_subsystem_id("bob.mem", 0),
        status="occupied",
    )

    assert (
        service.mark_absorb_report(tracked_report, tracked_ref).state
        is MemorySlotState.OCCUPIED
    )
    assert (
        service.mark_absorb_report(untracked_report, untracked_ref).state
        is MemorySlotState.OCCUPIED
    )


def test_resource_service_never_schedules_timeline_memory_events() -> None:
    timeline = Timeline(master_seed=1)
    manager, _, _ = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    service.reserve_memories(10, {"alice": 1}, reservation_id="r1")

    assert timeline.events_scheduled == 0


def test_runtime_context_adds_resource_service_when_manager_exists() -> None:
    manager, _, _ = manager_with_slots()
    agent = ResourceAgent(agent_id="agent")
    network = Network()
    controller = Node("controller")
    controller.add_agent(agent)
    network.add_node(controller)
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network,
        resource_manager=manager,
    )

    runtime.run()

    assert agent.contexts[0].resources is not None
    assert agent.contexts[0].resources.available_memories(0, "alice")


def test_runtime_resource_and_pair_services_follow_memory_reports() -> None:
    timeline = Timeline(master_seed=1)
    network = Network()
    alice_memory = QuantumMemory(memory_id="alice.mem", num_positions=1)
    bob_memory = QuantumMemory(memory_id="bob.mem", num_positions=1)

    alice = Node("alice")
    alice.add_device("mem", alice_memory)
    network.add_node(alice)

    bob = Node("bob")
    bob.add_device("mem", bob_memory)
    network.add_node(bob)

    agent = ResourcePairWorkflowAgent(
        agent_id="service-controller",
        alice_memory=alice_memory,
        bob_memory=bob_memory,
    )
    alice.add_agent(agent)
    network.wire_ports(
        "alice-memory-report",
        alice_memory.notice_port,
        agent.reports.port("alice_memory"),
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "bob-memory-report",
        bob_memory.notice_port,
        agent.reports.port("bob_memory"),
        target_action=AGENT_REPORT,
    )
    resource_manager = ResourceManager.from_network(network)
    pair_registry = EntangledPairRegistry()
    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        resource_manager=resource_manager,
        pair_registry=pair_registry,
        session_id="service-workflow",
    )

    runtime.run()

    alice_ref = MemoryRef("alice", "mem", 0)
    bob_ref = MemoryRef("bob", "mem", 0)
    reservation = resource_manager.get_reservation("reservation:alice-bob")
    pair = pair_registry.get("pair:alice-bob:service")

    assert reservation.owner == "service-controller"
    assert reservation.state.value == "committed"
    assert agent.reservation_id == "reservation:alice-bob"
    assert agent.reservation_refs == (alice_ref.key, bob_ref.key)
    assert agent.scheduled_request_ids == ["absorb:alice", "absorb:bob"]
    assert len(agent.absorb_reports) == 2
    assert tuple(sorted(report.memory_id for report in agent.absorb_reports)) == (
        "alice.mem",
        "bob.mem",
    )
    assert agent.occupied_refs == [alice_ref.key, bob_ref.key]
    assert agent.available_pair_ids == ("pair:alice-bob:service",)
    assert agent.registered_pair_state == "available"
    assert agent.reserved_pair_state == "reserved"
    assert agent.consumed_pair_state == "consumed"
    assert agent.consumed_refs == [alice_ref.key, bob_ref.key]
    assert pair.state.value == "consumed"
    assert pair.memory_ref_keys == (alice_ref.key, bob_ref.key)
    assert pair.metadata[0] == ("registered_by", "service-controller")
    assert resource_manager.get_slot(alice_ref).state is MemorySlotState.CONSUMED
    assert resource_manager.get_slot(bob_ref).state is MemorySlotState.CONSUMED
    assert alice_memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert bob_memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert timeline.qstate.state_of(memory_subsystem_id("alice.mem", 0)) is not None
    assert timeline.qstate.state_of(memory_subsystem_id("bob.mem", 0)) is not None


def test_mark_absorb_report_marks_reserved_and_free_slots_occupied() -> None:
    manager, alice_ref, bob_ref = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    # Reserve Alice's slot before absorb
    service.reserve_memories(10, {"alice": 1}, reservation_id="r1")
    manager.commit_reservation("r1", owner="agent")

    alice_report = MemoryAbsorbReport(
        report_id="r:1",
        memory_id="alice.mem",
        time=10,
        success=True,
        position=0,
        input_signal_id="sig:1",
        memory_subsystem=SubsystemId("alice.mem:0"),
        status="occupied",
    )

    # Bob's slot remains FREE before absorb
    bob_report = MemoryAbsorbReport(
        report_id="r:2",
        memory_id="bob.mem",
        time=10,
        success=True,
        position=0,
        input_signal_id="sig:2",
        memory_subsystem=SubsystemId("bob.mem:0"),
        status="occupied",
    )

    alice_view = service.mark_absorb_report(alice_report, alice_ref)
    bob_view = service.mark_absorb_report(bob_report, bob_ref)

    assert alice_view.state is MemorySlotState.OCCUPIED
    assert bob_view.state is MemorySlotState.OCCUPIED
    assert manager.get_slot(alice_ref).state is MemorySlotState.OCCUPIED
    assert manager.get_slot(bob_ref).state is MemorySlotState.OCCUPIED


def test_mark_absorb_report_validates_inputs() -> None:
    manager, alice_ref, _ = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    import pytest

    with pytest.raises(TypeError, match="MemoryAbsorbReport"):
        service.mark_absorb_report(None, alice_ref)  # type: ignore

    with pytest.raises(TypeError, match="MemoryRef"):
        service.mark_absorb_report(
            MemoryAbsorbReport(
                report_id="r:1",
                memory_id="alice.mem",
                time=10,
                success=True,
                position=0,
                input_signal_id="sig:1",
                memory_subsystem=SubsystemId("alice.mem:0"),
                status="occupied",
            ),
            None,  # type: ignore
        )


def test_mark_absorb_report_rejects_failed_report() -> None:
    manager, alice_ref, _ = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    failed_report = MemoryAbsorbReport(
        report_id="r:1",
        memory_id="alice.mem",
        time=10,
        success=False,
        position=0,
        input_signal_id=None,
        memory_subsystem=None,
        status="failed",
    )

    import pytest

    with pytest.raises(ValueError, match="failed absorb report"):
        service.mark_absorb_report(failed_report, alice_ref)


def test_mark_absorb_report_rejects_mismatched_position() -> None:
    manager, alice_ref, _ = manager_with_slots()
    service = ResourceService(manager, owner_agent_id="agent")

    mismatched_report = MemoryAbsorbReport(
        report_id="r:1",
        memory_id="alice.mem",
        time=10,
        success=True,
        position=1,  # Position 1 does not match alice_ref position 0
        input_signal_id="sig:1",
        memory_subsystem=SubsystemId("alice.mem:1"),
        status="occupied",
    )

    import pytest

    with pytest.raises(ValueError, match="position does not match"):
        service.mark_absorb_report(mismatched_report, alice_ref)
