from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components.ports import PortKind
from simyuj.control import Agent, AgentContext, SessionRuntime
from simyuj.control.pairs import PairService
from simyuj.engine import Component, Event, Timeline
from simyuj.entanglement.pair import EntangledPairRecord, PairState
from simyuj.entanglement.registry import EntangledPairRegistry
from simyuj.network import Network, Node
from simyuj.network.routing import Route
from simyuj.network.topology import TopologyEdge
from simyuj.resources.memory import MemoryRef


@dataclass(slots=True)
class PairAgent(Agent):
    contexts: list[AgentContext] = field(default_factory=list)

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        self.contexts.append(ctx)


class NoopTarget(Component):
    def handle_event(self, event, timeline) -> None:
        return None


def refs() -> tuple[MemoryRef, MemoryRef]:
    return MemoryRef("alice", "mem", 0), MemoryRef("bob", "mem", 0)


def pair(pair_id: str = "pair:1", *, expires_at: int | None = None):
    left, right = refs()
    return EntangledPairRecord(
        pair_id,
        left,
        right,
        fidelity=0.9,
        created_at=0,
        expires_at=expires_at,
        generation_link_id="link-1",
    )


def service_with_pair(
    *,
    expires_at: int | None = None,
) -> tuple[PairService, EntangledPairRegistry, Timeline]:
    registry = EntangledPairRegistry()
    registry.register(pair(expires_at=expires_at))
    timeline = Timeline(master_seed=1)
    return PairService(registry, timeline=timeline), registry, timeline


def route() -> Route:
    edge = TopologyEdge(
        link_id="link-1",
        source_node_id="alice",
        target_node_id="bob",
        port_kind=PortKind.QUANTUM,
    )
    return Route("alice", "bob", (edge,))


def test_query_methods_apply_registry_filters_and_route_candidates() -> None:
    service, _, _ = service_with_pair()

    assert [item.pair_id for item in service.available_between("alice", "bob")] == [
        "pair:1"
    ]
    assert [
        item.pair_id
        for item in service.available_between("alice", "bob", link_id="link-1")
    ] == ["pair:1"]
    assert service.available_between("alice", "bob", link_id="link-2") == ()

    candidates = service.route_hop_candidates(route())
    assert len(candidates) == 1
    assert candidates[0].has_pairs is True
    assert service.route_hops_ready(route()) is True


def test_pair_lifecycle_methods_delegate_to_registry() -> None:
    service, registry, _ = service_with_pair()

    assert service.reserve("pair:1").state is PairState.RESERVED
    assert service.release("pair:1").state is PairState.AVAILABLE
    assert service.consume("pair:1").state is PairState.CONSUMED

    registry.register(pair("pair:2"))
    assert service.expire("pair:2").state is PairState.EXPIRED

    registry.register(pair("pair:3"))
    assert service.fail("pair:3").state is PairState.FAILED


def test_expire_before_now_uses_timeline_current_time() -> None:
    service, _, timeline = service_with_pair(expires_at=1)
    timeline.schedule(
        Event(
            time=2,
            target_ref=NoopTarget(),
            action="noop",
            payload_ref=None,
        )
    )
    timeline.run_until(2)

    expired = service.expire_before_now()

    assert [item.pair_id for item in expired] == ["pair:1"]
    assert expired[0].state is PairState.EXPIRED


def test_pair_service_does_not_schedule_or_touch_resource_layers() -> None:
    service, _, timeline = service_with_pair()

    service.reserve("pair:1")

    assert timeline.events_scheduled == 0


def test_runtime_context_adds_pair_service_when_registry_exists() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair())
    agent = PairAgent(agent_id="agent")
    network = Network()
    controller = Node("controller")
    controller.add_agent(agent)
    network.add_node(controller)
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network,
        pair_registry=registry,
    )

    runtime.run()

    assert agent.contexts[0].pairs is not None
    assert agent.contexts[0].pairs.available_between("alice", "bob")
