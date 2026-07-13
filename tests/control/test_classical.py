from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest

from simyuj.components import Port, PortDelivery, PortDirection, PortKind, connect_ports
from simyuj.control import AGENT_MESSAGE, Agent, AgentContext, SessionRuntime
from simyuj.control.classical import ClassicalEndpoint, RoutingError
from simyuj.control.payloads import AgentMessage
from simyuj.engine import Component, Event, Timeline
from simyuj.network import Network, Node
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.primitives.messages import ClassicalMessage


@dataclass(slots=True)
class ClassicalAgent(Agent):
    messages: list[AgentMessage] = field(default_factory=list)

    def on_message(self, message: AgentMessage, ctx: AgentContext) -> None:
        del ctx
        self.messages.append(message)


@dataclass(slots=True)
class SendingAgent(ClassicalAgent):
    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        self.classical.send(message(receiver_id="bob"), ctx.timeline)


@dataclass(slots=True)
class Sink(Component):
    component_id: str = "sink"
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.component_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


@dataclass(slots=True)
class Source(Component):
    component_id: str = "source"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.component_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("source should not receive events")


def message(receiver_id: str = "bob") -> ClassicalMessage:
    return ClassicalMessage(
        sender_id="alice",
        receiver_id=receiver_id,
        body="ping",
        sent_time=0,
        message_id="m1",
    )


def attach_context(agent: Agent, timeline: Timeline) -> None:
    network = Network()
    topology = NetworkTopology(network)
    planner = RoutePlanner(topology)

    def provider(event: Event, current_timeline: Timeline) -> AgentContext:
        return AgentContext(
            agent_id=agent.agent_id,
            node_id=None,
            session_id="session-1",
            timeline=current_timeline,
            event=event,
            network=network,
            topology=topology,
            route_planner=planner,
            classical=agent._classical_endpoint,
        )

    agent.attach_context_provider(provider)


def delivery(target_port: Port, payload: object) -> PortDelivery:
    return PortDelivery(
        payload=payload,
        source_port=Source().output_port,
        target_port=target_port,
        connection_id="source.out->agent.in",
    )


def test_endpoint_is_not_component() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    assert not isinstance(endpoint, Component)


def test_classical_ports_are_owned_by_agent() -> None:
    agent = ClassicalAgent(agent_id="alice")
    endpoint = agent.enable_classical()

    assert endpoint.in_port().owner is agent
    assert endpoint.in_port().owner_id == "alice"
    assert endpoint.in_port().direction is PortDirection.INGRESS
    assert endpoint.out_port().owner is agent
    assert endpoint.out_port().owner_id == "alice"
    assert endpoint.out_port().direction is PortDirection.EGRESS


def test_named_ports_are_stable() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    assert endpoint.in_port("left") is endpoint.in_port("left")
    assert endpoint.out_port("right") is endpoint.out_port("right")


def test_extract_classical_message_returns_agent_message() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))
    msg = message()

    extracted = endpoint.extract(msg, 10)
    assert isinstance(extracted, AgentMessage)
    assert extracted.message is msg
    assert extracted.receive_time == 10


def test_extract_port_delivery_returns_agent_message() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))
    msg = message()

    extracted = endpoint.extract(delivery(endpoint.in_port("left"), msg), 10)
    assert isinstance(extracted, AgentMessage)
    assert extracted.message is msg
    assert extracted.receive_time == 10
    assert extracted.target_port_name == "left"
    assert extracted.connection_id == "source.out->agent.in"


def test_extract_rejects_port_delivery_non_classical_message() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    with pytest.raises(
        TypeError, match="PortDelivery payload must be ClassicalMessage"
    ):
        endpoint.extract(delivery(endpoint.in_port(), "bad"), 10)


def test_extract_rejects_raw_non_classical_message() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    with pytest.raises(TypeError, match="payload_ref must be ClassicalMessage"):
        endpoint.extract("bad", 10)


def test_extract_rejects_unknown_target_port() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))
    other = ClassicalEndpoint(ClassicalAgent(agent_id="bob"))

    with pytest.raises(ValueError, match="target_port is not an agent classical"):
        endpoint.extract(delivery(other.in_port(), message()), 10)


def test_agent_message_uses_endpoint_validation_when_enabled() -> None:
    timeline = Timeline(master_seed=1)
    agent = ClassicalAgent(agent_id="alice")
    endpoint = agent.enable_classical()
    attach_context(agent, timeline)
    msg = message()
    event = Event(
        time=0,
        target_ref=agent,
        action=AGENT_MESSAGE,
        payload_ref=delivery(endpoint.in_port(), msg),
    )

    agent.handle_event(event, timeline)

    assert len(agent.messages) == 1
    assert agent.messages[0].message is msg


def test_send_requires_classical_message() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    with pytest.raises(TypeError, match="message must be ClassicalMessage"):
        endpoint.send(
            cast(ClassicalMessage, "bad"),
            Timeline(master_seed=1),
            port_name="classical_out",
        )


def test_send_requires_connected_output_port() -> None:
    endpoint = ClassicalEndpoint(ClassicalAgent(agent_id="alice"))

    with pytest.raises(RuntimeError, match="not connected"):
        endpoint.send(message(), Timeline(master_seed=1), port_name="classical_out")


def test_send_schedules_one_event_through_connected_output_port() -> None:
    timeline = Timeline(master_seed=1)
    agent = ClassicalAgent(agent_id="alice")
    endpoint = agent.enable_classical()
    sink = Sink()
    msg = message()
    connect_ports(
        endpoint.out_port(),
        sink.input_port,
        target_action=AGENT_MESSAGE,
    )

    event = endpoint.send(msg, timeline, port_name="classical_out", priority=4)

    assert timeline.events_scheduled == 1
    assert event.target_ref is sink
    assert not isinstance(event.target_ref, ClassicalEndpoint)
    assert event.action == AGENT_MESSAGE
    assert event.priority == 4

    timeline.run_until_empty()

    assert len(sink.received) == 1
    assert sink.received[0].payload is msg


def test_send_with_route_resolution_delivers_to_agent_message_hook() -> None:
    timeline = Timeline(master_seed=1)
    alice = SendingAgent(agent_id="alice")
    bob = ClassicalAgent(agent_id="bob")
    alice.enable_classical()
    bob.enable_classical()
    connect_ports(
        alice.classical.out_port("to_bob"),
        bob.classical.in_port("from_alice"),
        target_action=AGENT_MESSAGE,
    )
    alice.classical.add_route("bob", "to_bob")
    network = Network()
    node = Node("controller")
    network.add_node(node)
    node.add_agent(alice)
    node.add_agent(bob)
    runtime = SessionRuntime(timeline=timeline, network=network)

    runtime.bind_all()
    runtime.schedule_agent_starts()
    timeline.run_until_empty()

    assert len(bob.messages) == 1
    delivered = bob.messages[0]
    assert delivered.message.sender_id == "alice"
    assert delivered.message.receiver_id == "bob"
    assert delivered.message.body == "ping"
    assert delivered.receive_time == 0
    assert delivered.source_port_name == "to_bob"
    assert delivered.target_port_name == "from_alice"


def test_send_raises_routing_error_if_no_route() -> None:
    timeline = Timeline(master_seed=1)
    agent = ClassicalAgent(agent_id="alice")
    endpoint = agent.enable_classical()

    endpoint.out_port("to_bob")

    msg = message(receiver_id="bob")

    with pytest.raises(RoutingError, match="No route configured"):
        endpoint.send(msg, timeline)
