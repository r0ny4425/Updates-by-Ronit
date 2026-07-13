from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.control import AGENT_EVENT, Agent, AgentContext, NodeAgent, SessionRuntime
from simyuj.engine import Component, Event, Timeline
from simyuj.network import Network, Node
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.runtime.binding import BindingContext


@dataclass(slots=True)
class RuntimeAgent(Agent):
    order: list[str] = field(default_factory=list)
    starts: list[tuple[str, AgentContext]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    def bind(self, context: BindingContext) -> None:
        self.order.append(f"agent:{self.agent_id}")

    def on_start(self, start, ctx: AgentContext) -> None:
        self.starts.append((start.agent_id, ctx))

    def on_event(self, event: Event, ctx: AgentContext) -> None:
        del ctx
        self.events.append(event)


@dataclass(slots=True)
class RuntimeNodeAgent(NodeAgent):
    starts: list[tuple[str, AgentContext]] = field(default_factory=list)

    def on_start(self, start, ctx: AgentContext) -> None:
        self.starts.append((start.agent_id, ctx))


class BindDevice:
    def __init__(self, label: str, order: list[str]) -> None:
        self.label = label
        self.order = order

    def bind(self, context: BindingContext) -> None:
        self.order.append(f"device:{self.label}:{context.component_id}")


def network_with_node(node_id: str = "alice") -> Network:
    network = Network()
    network.add_node(Node(node_id))
    return network


def network_with_agent(agent: Agent, node_id: str = "alice") -> Network:
    network = Network()
    node = Node(node_id)
    node.add_agent(agent)
    network.add_node(node)
    return network


def test_duplicate_agent_ids_rejected_across_nodes() -> None:
    network = Network()
    alice = Node("alice")
    bob = Node("bob")
    alice.add_agent(RuntimeAgent("shared"))
    bob.add_agent(RuntimeAgent("shared"))
    network.add_node(alice)
    network.add_node(bob)

    with pytest.raises(ValueError, match="duplicate agent id"):
        SessionRuntime(
            timeline=Timeline(master_seed=1),
            network=network,
        )


def test_node_add_agent_rejects_non_agent() -> None:
    node = Node("alice")

    with pytest.raises(TypeError, match="agent must be control.Agent"):
        node.add_agent(object())


def test_node_add_agent_rejects_wrong_node_agent() -> None:
    node = Node("alice")
    agent = RuntimeNodeAgent(agent_id="alice-agent", node_id="bob")

    with pytest.raises(ValueError, match="does not match node"):
        node.add_agent(agent)


def test_node_add_agent_rejects_duplicate_agent_id() -> None:
    node = Node("alice")
    node.add_agent(RuntimeAgent("alice-agent"))

    with pytest.raises(ValueError, match="already exists"):
        node.add_agent(RuntimeAgent("alice-agent"))


def test_empty_node_agent_node_id_rejected_by_construction() -> None:
    with pytest.raises(ValueError, match="node_id must be non-empty"):
        RuntimeNodeAgent(agent_id="alice-agent", node_id="")


def test_mismatched_node_agent_rejected_by_runtime() -> None:
    node = Node("bob")
    node._agents["alice-agent"] = RuntimeNodeAgent(
        agent_id="alice-agent",
        node_id="alice",
    )
    network = Network()
    network.add_node(node)

    with pytest.raises(ValueError, match="does not match node"):
        SessionRuntime(
            timeline=Timeline(master_seed=1),
            network=network,
        )


def test_topology_for_different_network_rejected() -> None:
    with pytest.raises(ValueError, match="topology must reference"):
        SessionRuntime(
            timeline=Timeline(master_seed=1),
            network=Network("runtime"),
            topology=NetworkTopology(Network("other")),
        )


def test_route_planner_for_different_topology_rejected() -> None:
    network = Network()
    topology = NetworkTopology(network)
    other = NetworkTopology(network)

    with pytest.raises(ValueError, match="route_planner must reference"):
        SessionRuntime(
            timeline=Timeline(master_seed=1),
            network=network,
            topology=topology,
            route_planner=RoutePlanner(other),
        )


def test_runtime_creates_topology_and_route_planner_if_omitted() -> None:
    network = Network()
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network,
    )

    assert isinstance(runtime.topology, NetworkTopology)
    assert runtime.topology.network is network
    assert isinstance(runtime.route_planner, RoutePlanner)
    assert runtime.route_planner.topology is runtime.topology


def test_context_provider_attached_to_every_agent() -> None:
    timeline = Timeline(master_seed=1)
    agent = RuntimeAgent("alice")
    runtime = SessionRuntime(
        timeline=timeline,
        network=network_with_agent(agent),
    )
    event = Event(time=0, target_ref=agent, action=AGENT_EVENT, payload_ref=None)

    agent.handle_event(event, timeline)

    assert agent.events == [event]
    assert runtime.session_id == "session"


def test_network_devices_bind_before_agents() -> None:
    order: list[str] = []
    network = Network()
    node = Node("alice")
    node.add_device("detector", BindDevice("detector", order))
    agent = RuntimeAgent("alice-agent", order=order)
    node.add_agent(agent)
    network.add_node(node)
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network,
    )

    runtime.bind_all()

    assert order == ["device:detector:detector", "agent:alice-agent"]


def test_agent_starts_scheduled_in_sorted_agent_id_order() -> None:
    timeline = Timeline(master_seed=1)
    bob = RuntimeAgent("bob")
    alice = RuntimeAgent("alice")
    network = Network()
    right = Node("right")
    left = Node("left")
    right.add_agent(bob)
    left.add_agent(alice)
    network.add_node(right)
    network.add_node(left)
    runtime = SessionRuntime(timeline=timeline, network=network)
    runtime.bind_all()

    events = runtime.schedule_agent_starts()

    assert [event.payload_ref.agent_id for event in events] == ["alice", "bob"]
    assert [event.target_ref for event in events] == [alice, bob]


def test_schedule_agent_starts_requires_bind_all_first() -> None:
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network_with_agent(RuntimeAgent("alice")),
    )

    with pytest.raises(RuntimeError, match="bind_all"):
        runtime.schedule_agent_starts()


def test_schedule_agent_starts_rejects_second_call() -> None:
    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network_with_agent(RuntimeAgent("alice")),
    )
    runtime.bind_all()
    runtime.schedule_agent_starts()

    with pytest.raises(RuntimeError, match="already scheduled"):
        runtime.schedule_agent_starts()


def test_run_until_empty_only_drains_existing_scheduled_events() -> None:
    timeline = Timeline(master_seed=1)
    agent = RuntimeAgent("alice")
    runtime = SessionRuntime(
        timeline=timeline,
        network=network_with_agent(agent),
    )
    timeline.schedule(
        Event(time=0, target_ref=agent, action=AGENT_EVENT, payload_ref=None)
    )

    runtime.run_until_empty()

    assert agent.events
    assert agent.starts == []
    assert timeline.events_scheduled == 1


def test_run_binds_schedules_starts_and_drains() -> None:
    timeline = Timeline(master_seed=1)
    agent = RuntimeAgent("alice")
    runtime = SessionRuntime(
        timeline=timeline,
        network=network_with_agent(agent),
        session_id="session-1",
    )

    runtime.run()

    assert agent.order == ["agent:alice"]
    assert [start_agent_id for start_agent_id, _ in agent.starts] == ["alice"]
    assert agent.starts[0][1].session_id == "session-1"
    assert timeline.events_scheduled == 1
    assert timeline.events_executed == 1


def test_runtime_does_not_auto_start_sources() -> None:
    class SourceLike(Component):
        def __init__(self) -> None:
            self.started = False

        def schedule_start(self, timeline: Timeline) -> None:
            del timeline
            self.started = True

        def handle_event(self, event, timeline) -> None:
            raise AssertionError("source should not receive events")

    source = SourceLike()
    network = Network()
    node = Node("alice")
    node.add_device("source", source)
    network.add_node(node)

    runtime = SessionRuntime(
        timeline=Timeline(master_seed=1),
        network=network,
    )

    runtime.run()

    assert source.started is False
