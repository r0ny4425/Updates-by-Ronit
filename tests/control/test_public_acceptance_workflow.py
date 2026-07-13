from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components import (
    ACTION_TRANSMIT_QUANTUM,
    QuantumChannel,
    SinglePhotonSource,
)
from simyuj.components.detectors import (
    ACTION_RUN_BELL_ANALYSIS,
    BellStateAnalyzer,
    DetectionReport,
)
from simyuj.components.memories import (
    MEMORY_ABSORB,
    MemoryAbsorbReport,
    MemoryPositionStatus,
    QuantumMemory,
    memory_subsystem_id,
)
from simyuj.components.ports import PortKind
from simyuj.components.sources import EntangledPairSource
from simyuj.control import AGENT_REPORT, Agent, AgentContext, NodeAgent, SessionRuntime
from simyuj.control.payloads import TimerFired
from simyuj.engine import Timeline
from simyuj.entanglement import EntangledPairRecord, EntangledPairRegistry
from simyuj.network import Network, Node
from simyuj.resources import MemoryRef, MemorySlotState, ResourceManager

ALICE_REF = MemoryRef("alice", "mem", 0)
RELAY_LEFT_REF = MemoryRef("relay", "left_mem", 0)
RELAY_RIGHT_REF = MemoryRef("relay", "right_mem", 0)
BOB_REF = MemoryRef("bob", "mem", 0)

_MEMORY_REF_BY_ID = {
    "alice.mem": ALICE_REF,
    "relay.left_mem": RELAY_LEFT_REF,
    "relay.right_mem": RELAY_RIGHT_REF,
    "bob.mem": BOB_REF,
}


def _occupancy_token(report: MemoryAbsorbReport) -> int | None:
    for key, value in report.meta:
        if key == "occupancy_token" and isinstance(value, int):
            return value
    return None


@dataclass(slots=True)
class PairWorkflowAgent(Agent):
    source: EntangledPairSource
    route_link_ids: tuple[str, ...] = ()
    reservation_id: str | None = None
    reservation_refs: tuple[tuple[str, str, int], ...] = ()
    absorb_reports: list[MemoryAbsorbReport] = field(default_factory=list)
    available_pair_ids: tuple[str, ...] = ()
    reserved_pair_state: str | None = None
    bell_label: str | None = None

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        left_route = ctx.route_planner.fewest_hops_path(
            "eps",
            "alice",
            port_kind=PortKind.QUANTUM,
        )
        right_route = ctx.route_planner.fewest_hops_path(
            "eps",
            "bob",
            port_kind=PortKind.QUANTUM,
        )
        assert left_route is not None
        assert right_route is not None
        assert ctx.resources is not None

        self.route_link_ids = left_route.link_ids + right_route.link_ids
        reservation = ctx.resources.reserve_memories(
            ctx.timeline.current_time,
            {"alice": 1, "bob": 1},
            reservation_id="reservation:pair-1",
            created_at=ctx.timeline.current_time,
            metadata=(
                ("left_delivery_route", left_route.link_ids),
                ("right_delivery_route", right_route.link_ids),
            ),
        )
        committed = ctx.resources.commit(reservation.reservation_id)

        self.reservation_id = committed.reservation_id
        self.reservation_refs = committed.memory_ref_keys

        self.source.schedule_start(ctx.timeline)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        if not isinstance(report, MemoryAbsorbReport):
            return

        self.absorb_reports.append(report)

        if len(self.absorb_reports) == 2 and self.reserved_pair_state is None:
            self._register_pair(ctx)

    def _register_pair(self, ctx: AgentContext) -> None:
        assert ctx.resources is not None
        assert ctx.pairs is not None

        alice_rep = next(r for r in self.absorb_reports if r.memory_id == "alice.mem")
        bob_rep = next(r for r in self.absorb_reports if r.memory_id == "bob.mem")
        assert alice_rep.success is True
        assert bob_rep.success is True
        assert alice_rep.position == ALICE_REF.position
        assert bob_rep.position == BOB_REF.position
        pair = ctx.pairs.register(
            EntangledPairRecord(
                pair_id="pair:eps:1",
                left=ALICE_REF,
                right=BOB_REF,
                fidelity=1.0,
                created_at=ctx.timeline.current_time,
                generation_link_id="eps-left-to-alice",
                left_occupancy_token=_occupancy_token(alice_rep),
                right_occupancy_token=_occupancy_token(bob_rep),
                metadata=(
                    ("registered_by", self.agent_id),
                    ("registry_update", "explicit_after_memory_absorb_reports"),
                    ("routes", self.route_link_ids),
                    (
                        "absorb_reports",
                        tuple(
                            sorted(report.report_id for report in self.absorb_reports)
                        ),
                    ),
                ),
            )
        )
        self.available_pair_ids = tuple(
            record.pair_id for record in ctx.pairs.available_between("alice", "bob")
        )
        reserved = ctx.pairs.reserve(pair.pair_id)
        self.reserved_pair_state = reserved.state.value

        for report in self.absorb_reports:
            ctx.resources.mark_absorb_report(
                report,
                _MEMORY_REF_BY_ID[report.memory_id],
            )

        bell = ctx.timeline.qstate.measure_bell(
            targets=(
                memory_subsystem_id("alice.mem", 0),
                memory_subsystem_id("bob.mem", 0),
            ),
            collapse=False,
        )
        self.bell_label = bell.label


@dataclass(slots=True)
class ControllerManagedSwapAgent(NodeAgent):
    left_source: EntangledPairSource
    right_source: EntangledPairSource
    route_link_ids: tuple[str, ...] = ()
    reservation_id: str | None = None
    reservation_refs: tuple[tuple[str, str, int], ...] = ()
    absorb_reports: list[MemoryAbsorbReport] = field(default_factory=list)
    bsa_reports: list[DetectionReport] = field(default_factory=list)
    elementary_pair_ids: tuple[str, str] | None = None
    consumed_pair_states: tuple[str, str] | None = None
    swapped_pair_id: str | None = None
    swapped_pair_state: str | None = None
    bsa_outcome: object | None = None
    outer_bell_label: str | None = None

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.resources is not None

        route_specs = (
            ("left_eps", "alice"),
            ("left_eps", "relay"),
            ("right_eps", "relay"),
            ("right_eps", "bob"),
        )
        routes = []
        for src, dst in route_specs:
            route = ctx.route_planner.fewest_hops_path(
                src,
                dst,
                port_kind=PortKind.QUANTUM,
            )
            assert route is not None
            routes.append(route)

        self.route_link_ids = tuple(
            link_id for route in routes for link_id in route.link_ids
        )
        reservation = ctx.resources.reserve_memories(
            ctx.timeline.current_time,
            {"alice": 1, "relay": 2, "bob": 1},
            reservation_id="reservation:swap-1",
            created_at=ctx.timeline.current_time,
            metadata=(("routes", self.route_link_ids),),
        )
        committed = ctx.resources.commit(reservation.reservation_id)
        self.reservation_id = committed.reservation_id
        self.reservation_refs = committed.memory_ref_keys

        self.left_source.schedule_start(ctx.timeline)
        self.right_source.schedule_start(ctx.timeline)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        if isinstance(report, MemoryAbsorbReport):
            self.absorb_reports.append(report)

            assert ctx.resources is not None
            ctx.resources.mark_absorb_report(
                report,
                _MEMORY_REF_BY_ID[report.memory_id],
            )

            if len(self.absorb_reports) == 4 and self.elementary_pair_ids is None:
                self._register_elementary_pairs_and_run_bsa(ctx)
            return

        if isinstance(report, DetectionReport):
            assert report.success is True
            self.bsa_reports.append(report)
            self._consume_old_pairs_and_register_swapped_pair(ctx, report)

    def _register_elementary_pairs_and_run_bsa(self, ctx: AgentContext) -> None:
        assert ctx.pairs is not None
        assert ctx.memory is not None

        alice_rep = next(r for r in self.absorb_reports if r.memory_id == "alice.mem")
        relay_left_rep = next(
            r for r in self.absorb_reports if r.memory_id == "relay.left_mem"
        )
        assert alice_rep.success is True
        assert relay_left_rep.success is True
        assert alice_rep.position == ALICE_REF.position
        assert relay_left_rep.position == RELAY_LEFT_REF.position
        left_pair = ctx.pairs.register(
            EntangledPairRecord(
                pair_id="pair:alice-relay:1",
                left=ALICE_REF,
                right=RELAY_LEFT_REF,
                fidelity=1.0,
                created_at=ctx.timeline.current_time,
                generation_link_id="left-eps-to-alice",
                left_occupancy_token=_occupancy_token(alice_rep),
                right_occupancy_token=_occupancy_token(relay_left_rep),
                metadata=(
                    ("registered_by", self.agent_id),
                    ("registry_update", "explicit_after_memory_absorb_reports"),
                ),
            )
        )

        relay_right_rep = next(
            r for r in self.absorb_reports if r.memory_id == "relay.right_mem"
        )
        bob_rep = next(r for r in self.absorb_reports if r.memory_id == "bob.mem")
        assert relay_right_rep.success is True
        assert bob_rep.success is True
        assert relay_right_rep.position == RELAY_RIGHT_REF.position
        assert bob_rep.position == BOB_REF.position
        right_pair = ctx.pairs.register(
            EntangledPairRecord(
                pair_id="pair:relay-bob:1",
                left=RELAY_RIGHT_REF,
                right=BOB_REF,
                fidelity=1.0,
                created_at=ctx.timeline.current_time,
                generation_link_id="right-eps-to-bob",
                left_occupancy_token=_occupancy_token(relay_right_rep),
                right_occupancy_token=_occupancy_token(bob_rep),
                metadata=(
                    ("registered_by", self.agent_id),
                    ("registry_update", "explicit_after_memory_absorb_reports"),
                ),
            )
        )
        self.elementary_pair_ids = (left_pair.pair_id, right_pair.pair_id)

        ctx.memory.emit(
            "left_mem",
            0,
            request_id="swap:emit:relay-left",
            meta=(("controller_action", "send_to_bsa"),),
        )
        ctx.memory.emit(
            "right_mem",
            0,
            request_id="swap:emit:relay-right",
            meta=(("controller_action", "send_to_bsa"),),
        )

    def _consume_old_pairs_and_register_swapped_pair(
        self,
        ctx: AgentContext,
        report: DetectionReport,
    ) -> None:
        assert self.elementary_pair_ids is not None
        assert ctx.resources is not None
        assert ctx.pairs is not None

        self.bsa_outcome = report.outcome
        left_pair = ctx.pairs.consume(self.elementary_pair_ids[0])
        right_pair = ctx.pairs.consume(self.elementary_pair_ids[1])
        self.consumed_pair_states = (left_pair.state.value, right_pair.state.value)

        ctx.resources.mark_consumed(RELAY_LEFT_REF)
        ctx.resources.mark_consumed(RELAY_RIGHT_REF)

        swapped = ctx.pairs.register(
            EntangledPairRecord(
                pair_id="pair:alice-bob:swapped",
                left=ALICE_REF,
                right=BOB_REF,
                fidelity=1.0,
                created_at=ctx.timeline.current_time,
                left_occupancy_token=left_pair.left_occupancy_token,
                right_occupancy_token=right_pair.right_occupancy_token,
                metadata=(
                    ("registered_by", self.agent_id),
                    ("registry_update", "explicit_after_bsa_report"),
                    ("consumed_pairs", self.elementary_pair_ids),
                    ("bsa_report_id", report.report_id),
                    ("bsa_outcome", report.outcome),
                ),
            )
        )
        self.swapped_pair_id = swapped.pair_id
        self.swapped_pair_state = swapped.state.value

        outer = ctx.timeline.qstate.measure_bell(
            targets=(
                memory_subsystem_id("alice.mem", 0),
                memory_subsystem_id("bob.mem", 0),
            ),
            collapse=False,
        )
        self.outer_bell_label = outer.label
        assert self.outer_bell_label == report.outcome


@dataclass(slots=True)
class LossCleanupAgent(Agent):
    source: SinglePhotonSource
    reservation_id: str | None = None
    reservation_refs: tuple[tuple[str, str, int], ...] = ()
    absorb_reports: list[MemoryAbsorbReport] = field(default_factory=list)
    cleanup_state: str | None = None
    cleanup_time: int | None = None

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.resources is not None
        assert ctx.timers is not None

        reservation = ctx.resources.reserve_memories(
            ctx.timeline.current_time,
            {"alice": 1},
            reservation_id="reservation:lost-photon",
            created_at=ctx.timeline.current_time,
            metadata=(("cleanup", "controller_timer"),),
        )
        committed = ctx.resources.commit(reservation.reservation_id)
        self.reservation_id = committed.reservation_id
        self.reservation_refs = committed.memory_ref_keys

        self.source.schedule_start(ctx.timeline)
        ctx.timers.set("lost-photon-cleanup", 5)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        del ctx
        if isinstance(report, MemoryAbsorbReport):
            self.absorb_reports.append(report)

    def on_timer(self, timer: TimerFired, ctx: AgentContext) -> None:
        if timer.timer_id != "lost-photon-cleanup":
            return
        assert ctx.resources is not None
        assert self.reservation_id is not None

        released = ctx.resources.release(self.reservation_id)
        self.cleanup_state = released.state.value
        self.cleanup_time = ctx.timeline.current_time


@dataclass(slots=True)
class ContendingReservationAgent(Agent):
    reservation_id: str | None = None
    reservation_refs: tuple[tuple[str, str, int], ...] = ()
    error: str | None = None
    start_event_id: int | None = None

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.resources is not None
        self.start_event_id = ctx.event.event_id

        try:
            reservation = ctx.resources.reserve_memories(
                ctx.timeline.current_time,
                {"alice": 1},
                reservation_id=f"reservation:{self.agent_id}",
                created_at=ctx.timeline.current_time,
            )
            committed = ctx.resources.commit(reservation.reservation_id)
        except ValueError as exc:
            self.error = str(exc)
            return

        self.reservation_id = committed.reservation_id
        self.reservation_refs = committed.memory_ref_keys


def _build_network(
    source: EntangledPairSource,
    alice_memory: QuantumMemory,
    bob_memory: QuantumMemory,
    agent: PairWorkflowAgent,
) -> Network:
    network = Network("acceptance")

    eps = Node("eps")
    eps.add_device("source", source)
    eps.add_agent(agent)
    eps.register_port("left", source.left_output_port)
    eps.register_port("right", source.right_output_port)

    alice = Node("alice")
    alice.add_device("mem", alice_memory)
    alice.register_port("mem_in", alice_memory.input_port)

    bob = Node("bob")
    bob.add_device("mem", bob_memory)
    bob.register_port("mem_in", bob_memory.input_port)

    network.add_node(eps)
    network.add_node(alice)
    network.add_node(bob)

    network.add_quantum_link("eps-left-to-alice", "eps", "alice")
    network.add_quantum_link("eps-right-to-bob", "eps", "bob")

    network.wire_ports(
        "eps-left-to-alice",
        source.left_output_port,
        alice_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "eps-right-to-bob",
        source.right_output_port,
        bob_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
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

    return network


def _run_workflow(seed: int) -> tuple[object, ...]:
    timeline = Timeline(master_seed=seed)
    source = EntangledPairSource(
        device_id="eps",
        frequency_hz=1e6,
        duration_s=1e-12,
    )
    alice_memory = QuantumMemory(memory_id="alice.mem", num_positions=1)
    bob_memory = QuantumMemory(memory_id="bob.mem", num_positions=1)
    agent = PairWorkflowAgent(agent_id="controller", source=source)
    network = _build_network(source, alice_memory, bob_memory, agent)
    resource_manager = ResourceManager.from_network(network)
    pair_registry = EntangledPairRegistry()

    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        resource_manager=resource_manager,
        pair_registry=pair_registry,
        session_id="acceptance-session",
    )

    runtime.run()

    pair = pair_registry.get("pair:eps:1")
    assert agent.route_link_ids == ("eps-left-to-alice", "eps-right-to-bob")
    assert agent.reservation_id == "reservation:pair-1"
    assert agent.reservation_refs == (ALICE_REF.key, BOB_REF.key)
    assert agent.available_pair_ids == ("pair:eps:1",)
    assert agent.reserved_pair_state == "reserved"
    assert agent.bell_label == "phi+"
    assert pair.metadata[1] == (
        "registry_update",
        "explicit_after_memory_absorb_reports",
    )
    assert pair.state.value == "reserved"
    assert resource_manager.get_slot(ALICE_REF).state is MemorySlotState.OCCUPIED
    assert resource_manager.get_slot(BOB_REF).state is MemorySlotState.OCCUPIED
    assert alice_memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert bob_memory.positions[0].status is MemoryPositionStatus.OCCUPIED

    return (
        agent.route_link_ids,
        agent.reservation_id,
        agent.reservation_refs,
        tuple(sorted(report.report_id for report in agent.absorb_reports)),
        agent.available_pair_ids,
        agent.reserved_pair_state,
        agent.bell_label,
        pair.memory_ref_keys,
        pair.state.value,
        resource_manager.get_slot(ALICE_REF).state.value,
        resource_manager.get_slot(BOB_REF).state.value,
        tuple(position.status.value for position in alice_memory.positions),
        tuple(position.status.value for position in bob_memory.positions),
        timeline.events_scheduled,
        timeline.events_executed,
    )


def test_public_route_resource_memory_pair_workflow_replays_deterministically() -> None:
    first = _run_workflow(seed=17)
    second = _run_workflow(seed=17)

    assert first == second


def _run_lost_photon_cleanup(seed: int) -> tuple[object, ...]:
    timeline = Timeline(master_seed=seed)
    source = SinglePhotonSource(
        device_id="loss_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
    )
    channel = QuantumChannel(
        channel_id="lossy_link",
        delay_ticks=0,
        fixed_insertion_loss_db=1e9,
    )
    alice_memory = QuantumMemory(memory_id="alice.mem", num_positions=1)
    agent = LossCleanupAgent(agent_id="cleanup-controller", source=source)
    network = Network("lost-photon-cleanup")

    source_node = Node("source")
    source_node.add_device("source", source)
    source_node.add_agent(agent)
    network.add_node(source_node)

    alice = Node("alice")
    alice.add_device("mem", alice_memory)
    network.add_node(alice)

    network.add_quantum_link("lossy_link", "source", "alice", channel=channel)

    network.wire_ports(
        "source-to-lossy-link",
        source.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    network.wire_ports(
        "lossy-link-to-memory",
        channel.output_port,
        alice_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "alice-memory-report",
        alice_memory.notice_port,
        agent.reports.port("alice_memory"),
        target_action=AGENT_REPORT,
    )

    resource_manager = ResourceManager.from_network(network)
    pair_registry = EntangledPairRegistry()
    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        resource_manager=resource_manager,
        pair_registry=pair_registry,
        session_id="lost-photon-cleanup",
    )

    runtime.run()

    reservation = resource_manager.get_reservation("reservation:lost-photon")
    assert channel.lost_count == 1
    assert channel.delivered_count == 0
    assert agent.absorb_reports == []
    assert pair_registry.all_pairs() == ()
    assert resource_manager.get_slot(ALICE_REF).state is MemorySlotState.FREE
    assert resource_manager.available_memories(10, "alice") == (ALICE_REF,)
    assert alice_memory.positions[0].status is MemoryPositionStatus.EMPTY

    return (
        agent.reservation_id,
        agent.reservation_refs,
        agent.cleanup_state,
        agent.cleanup_time,
        reservation.state.value,
        channel.received_count,
        channel.lost_count,
        channel.delivered_count,
        tuple(report.report_id for report in source.reports),
        tuple(pair.pair_id for pair in pair_registry.all_pairs()),
        resource_manager.get_slot(ALICE_REF).state.value,
        tuple(position.status.value for position in alice_memory.positions),
        timeline.events_scheduled,
        timeline.events_executed,
    )


def test_controller_releases_reserved_memory_after_lost_photon_workflow() -> None:
    first = _run_lost_photon_cleanup(seed=101)
    second = _run_lost_photon_cleanup(seed=101)

    assert first == second
    assert first[2] == "released"
    assert first[4] == "released"


def test_two_agents_contending_for_one_memory_slot_allocate_deterministically() -> None:
    timeline = Timeline(master_seed=1)
    alice_memory = QuantumMemory(memory_id="alice.mem", num_positions=1)
    network = Network("contention")
    alice = Node("alice")
    alice.add_device("mem", alice_memory)
    network.add_node(alice)
    resource_manager = ResourceManager.from_network(network)
    agent_b = ContendingReservationAgent(agent_id="agent-b")
    agent_a = ContendingReservationAgent(agent_id="agent-a")
    alice.add_agent(agent_b)
    alice.add_agent(agent_a)
    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        resource_manager=resource_manager,
        session_id="contention",
    )

    runtime.run()

    assert agent_a.start_event_id is not None
    assert agent_b.start_event_id is not None
    assert agent_a.start_event_id < agent_b.start_event_id
    assert agent_a.reservation_id == "reservation:agent-a"
    assert agent_a.reservation_refs == (ALICE_REF.key,)
    assert agent_b.reservation_id is None
    assert agent_b.error is not None
    assert "0 available memory slot(s), but 1 requested" in agent_b.error
    assert resource_manager.available_memories(10, "alice") == ()
    assert resource_manager.get_slot(ALICE_REF).state is MemorySlotState.RESERVED
    holder = resource_manager.reservation_for_memory(ALICE_REF)
    assert holder is not None
    assert holder.reservation_id == "reservation:agent-a"
    assert holder.owner == "agent-a"


def _build_swapping_network(
    *,
    left_source: EntangledPairSource,
    right_source: EntangledPairSource,
    alice_memory: QuantumMemory,
    relay_left_memory: QuantumMemory,
    relay_right_memory: QuantumMemory,
    bob_memory: QuantumMemory,
    bsa: BellStateAnalyzer,
    agent: ControllerManagedSwapAgent,
) -> Network:
    network = Network("controller-managed-swapping")

    left_eps = Node("left_eps")
    left_eps.add_device("source", left_source)
    left_eps.register_port("left", left_source.left_output_port)
    left_eps.register_port("right", left_source.right_output_port)

    right_eps = Node("right_eps")
    right_eps.add_device("source", right_source)
    right_eps.register_port("left", right_source.left_output_port)
    right_eps.register_port("right", right_source.right_output_port)

    alice = Node("alice")
    alice.add_device("mem", alice_memory)
    alice.register_port("mem_in", alice_memory.input_port)

    relay = Node("relay")
    relay.add_device("left_mem", relay_left_memory)
    relay.add_device("right_mem", relay_right_memory)
    relay.add_device("bsa", bsa)
    relay.add_agent(agent)
    relay.register_port("left_mem_in", relay_left_memory.input_port)
    relay.register_port("right_mem_in", relay_right_memory.input_port)
    relay.register_port("left_mem_out", relay_left_memory.output_port)
    relay.register_port("right_mem_out", relay_right_memory.output_port)
    relay.register_port("bsa_left", bsa.left_input_port)
    relay.register_port("bsa_right", bsa.right_input_port)

    bob = Node("bob")
    bob.add_device("mem", bob_memory)
    bob.register_port("mem_in", bob_memory.input_port)

    for node in (left_eps, right_eps, alice, relay, bob):
        network.add_node(node)

    network.add_quantum_link("left-eps-to-alice", "left_eps", "alice")
    network.add_quantum_link("left-eps-to-relay-left", "left_eps", "relay")
    network.add_quantum_link("right-eps-to-relay-right", "right_eps", "relay")
    network.add_quantum_link("right-eps-to-bob", "right_eps", "bob")

    network.wire_ports(
        "left-eps-to-alice",
        left_source.left_output_port,
        alice_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "left-eps-to-relay-left",
        left_source.right_output_port,
        relay_left_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "right-eps-to-relay-right",
        right_source.left_output_port,
        relay_right_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "right-eps-to-bob",
        right_source.right_output_port,
        bob_memory.input_port,
        target_action=MEMORY_ABSORB,
    )
    network.wire_ports(
        "relay-left-memory-to-bsa",
        relay_left_memory.output_port,
        bsa.left_input_port,
        target_action=ACTION_RUN_BELL_ANALYSIS,
    )
    network.wire_ports(
        "relay-right-memory-to-bsa",
        relay_right_memory.output_port,
        bsa.right_input_port,
        target_action=ACTION_RUN_BELL_ANALYSIS,
    )

    for label, memory in (
        ("alice_memory", alice_memory),
        ("relay_left_memory", relay_left_memory),
        ("relay_right_memory", relay_right_memory),
        ("bob_memory", bob_memory),
    ):
        network.wire_ports(
            f"{label}-report",
            memory.notice_port,
            agent.reports.port(label),
            target_action=AGENT_REPORT,
        )
    network.wire_ports(
        "bsa-report",
        bsa.output_port,
        agent.reports.port("bsa"),
        target_action=AGENT_REPORT,
    )

    return network


def _run_controller_managed_swap(seed: int) -> tuple[object, ...]:
    timeline = Timeline(master_seed=seed)
    left_source = EntangledPairSource(
        device_id="left_eps",
        frequency_hz=1e6,
        duration_s=1e-12,
    )
    right_source = EntangledPairSource(
        device_id="right_eps",
        frequency_hz=1e6,
        duration_s=1e-12,
    )
    alice_memory = QuantumMemory(memory_id="alice.mem", num_positions=1)
    relay_left_memory = QuantumMemory(memory_id="relay.left_mem", num_positions=1)
    relay_right_memory = QuantumMemory(memory_id="relay.right_mem", num_positions=1)
    bob_memory = QuantumMemory(memory_id="bob.mem", num_positions=1)
    bsa = BellStateAnalyzer(device_id="relay.bsa", pairing_key=None)
    agent = ControllerManagedSwapAgent(
        agent_id="swap-controller",
        node_id="relay",
        left_source=left_source,
        right_source=right_source,
    )
    network = _build_swapping_network(
        left_source=left_source,
        right_source=right_source,
        alice_memory=alice_memory,
        relay_left_memory=relay_left_memory,
        relay_right_memory=relay_right_memory,
        bob_memory=bob_memory,
        bsa=bsa,
        agent=agent,
    )
    resource_manager = ResourceManager.from_network(network)
    pair_registry = EntangledPairRegistry()

    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        resource_manager=resource_manager,
        pair_registry=pair_registry,
        session_id="swap-acceptance-session",
    )

    runtime.run()

    elementary_left = pair_registry.get("pair:alice-relay:1")
    elementary_right = pair_registry.get("pair:relay-bob:1")
    swapped = pair_registry.get("pair:alice-bob:swapped")

    assert agent.reservation_id == "reservation:swap-1"
    assert agent.reservation_refs == (
        ALICE_REF.key,
        BOB_REF.key,
        RELAY_LEFT_REF.key,
        RELAY_RIGHT_REF.key,
    )
    assert len(agent.absorb_reports) == 4
    assert agent.elementary_pair_ids == ("pair:alice-relay:1", "pair:relay-bob:1")
    assert elementary_left.metadata[1] == (
        "registry_update",
        "explicit_after_memory_absorb_reports",
    )
    assert elementary_right.metadata[1] == (
        "registry_update",
        "explicit_after_memory_absorb_reports",
    )
    assert agent.consumed_pair_states == ("consumed", "consumed")
    assert elementary_left.state.value == "consumed"
    assert elementary_right.state.value == "consumed"
    assert len(agent.bsa_reports) == 1
    assert agent.bsa_outcome == "phi+"
    assert agent.outer_bell_label == "phi+"
    assert agent.swapped_pair_id == "pair:alice-bob:swapped"
    assert agent.swapped_pair_state == "available"
    assert swapped.metadata[1] == ("registry_update", "explicit_after_bsa_report")
    assert resource_manager.get_slot(ALICE_REF).state is MemorySlotState.OCCUPIED
    assert resource_manager.get_slot(BOB_REF).state is MemorySlotState.OCCUPIED
    assert resource_manager.get_slot(RELAY_LEFT_REF).state is MemorySlotState.CONSUMED
    assert resource_manager.get_slot(RELAY_RIGHT_REF).state is MemorySlotState.CONSUMED
    assert alice_memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert bob_memory.positions[0].status is MemoryPositionStatus.OCCUPIED
    assert relay_left_memory.positions[0].status is MemoryPositionStatus.EMPTY
    assert relay_right_memory.positions[0].status is MemoryPositionStatus.EMPTY

    return (
        agent.reservation_id,
        agent.reservation_refs,
        tuple(sorted(report.report_id for report in agent.absorb_reports)),
        agent.elementary_pair_ids,
        agent.consumed_pair_states,
        tuple(report.report_id for report in agent.bsa_reports),
        agent.bsa_outcome,
        agent.outer_bell_label,
        swapped.memory_ref_keys,
        swapped.state.value,
        resource_manager.get_slot(ALICE_REF).state.value,
        resource_manager.get_slot(RELAY_LEFT_REF).state.value,
        resource_manager.get_slot(RELAY_RIGHT_REF).state.value,
        resource_manager.get_slot(BOB_REF).state.value,
        tuple(position.status.value for position in alice_memory.positions),
        tuple(position.status.value for position in relay_left_memory.positions),
        tuple(position.status.value for position in relay_right_memory.positions),
        tuple(position.status.value for position in bob_memory.positions),
        timeline.events_scheduled,
        timeline.events_executed,
    )


def test_controller_managed_bsa_report_registers_swapped_pair() -> None:
    first = _run_controller_managed_swap(seed=29)
    second = _run_controller_managed_swap(seed=29)

    assert first == second
