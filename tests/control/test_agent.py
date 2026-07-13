from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import Port, PortDelivery, PortDirection, PortKind
from simyuj.control import (
    AGENT_EVENT,
    AGENT_MESSAGE,
    AGENT_REPORT,
    AGENT_START,
    AGENT_TIMER,
    Agent,
    AgentContext,
    NodeAgent,
)
from simyuj.control.payloads import AgentMessage, AgentReport, AgentStart, TimerFired
from simyuj.engine import Component, Event, Timeline
from simyuj.network import Network
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.primitives.messages import ClassicalMessage


@dataclass(slots=True)
class RecordingAgent(Agent):
    seen: list[tuple[str, object, AgentContext]] = field(default_factory=list)

    def on_start(self, start: AgentStart, ctx: AgentContext) -> None:
        self.seen.append(("start", start, ctx))

    def on_timer(self, timer: TimerFired, ctx: AgentContext) -> None:
        self.seen.append(("timer", timer, ctx))

    def on_message(self, message: AgentMessage, ctx: AgentContext) -> None:
        self.seen.append(("message", message, ctx))

    def on_report(self, report: object, ctx: AgentContext) -> None:
        self.seen.append(("report", report, ctx))

    def on_event(self, event: Event, ctx: AgentContext) -> None:
        self.seen.append(("event", event, ctx))


@dataclass(slots=True)
class PassiveComponent(Component):
    component_id: str
    port_name: str = "out"
    direction: PortDirection = PortDirection.EGRESS
    port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.port = Port(
            name=self.port_name,
            owner=self,
            owner_id=self.component_id,
            port_kind=PortKind.CLASSICAL,
            direction=self.direction,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("passive component should not receive events")


def attach_context(agent: Agent, timeline: Timeline) -> None:
    network = Network()
    topology = NetworkTopology(network)
    planner = RoutePlanner(topology)

    def provider(event: Event, current_timeline: Timeline) -> AgentContext:
        return AgentContext(
            agent_id=agent.agent_id,
            node_id=getattr(agent, "node_id", None),
            session_id="session-1",
            timeline=current_timeline,
            event=event,
            network=network,
            topology=topology,
            route_planner=planner,
        )

    agent.attach_context_provider(provider)


def make_event(agent: Agent, action: str, payload: object) -> Event:
    return Event(
        time=0,
        target_ref=agent,
        action=action,
        payload_ref=payload,
    )


def make_message() -> ClassicalMessage:
    return ClassicalMessage(
        sender_id="alice",
        receiver_id="bob",
        body="ping",
        sent_time=0,
        message_id="m1",
    )


def make_delivery(target_port: Port, payload: object) -> PortDelivery:
    source = PassiveComponent(component_id="source")
    return PortDelivery(
        payload=payload,
        source_port=source.port,
        target_port=target_port,
        connection_id="source.out->agent.in",
    )


def test_node_agent_rejects_empty_node_id() -> None:
    with pytest.raises(ValueError, match="node_id must be non-empty"):
        NodeAgent(agent_id="alice", node_id="")


def test_agent_without_context_provider_raises_runtime_error() -> None:
    agent = RecordingAgent(agent_id="alice")
    event = make_event(
        agent,
        AGENT_START,
        AgentStart(agent_id="alice", session_id="session-1"),
    )

    with pytest.raises(RuntimeError, match="not attached to a SessionRuntime"):
        agent.handle_event(event, Timeline(master_seed=1))


def test_event_target_ref_must_be_this_agent() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    other = RecordingAgent(agent_id="bob")
    attach_context(agent, timeline)
    event = make_event(
        other,
        AGENT_START,
        AgentStart(agent_id="alice", session_id="session-1"),
    )

    with pytest.raises(ValueError, match="target_ref must be this Agent"):
        agent.handle_event(event, timeline)


def test_agent_start_dispatches_to_on_start() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    start = AgentStart(agent_id="alice", session_id="session-1")

    agent.handle_event(make_event(agent, AGENT_START, start), timeline)

    assert agent.seen[0][0] == "start"
    assert agent.seen[0][1] is start


def test_agent_start_context_exposes_timeline_and_runtime_views() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    start = AgentStart(agent_id="alice", session_id="session-1")

    agent.handle_event(make_event(agent, AGENT_START, start), timeline)
    ctx = agent.seen[0][2]

    assert ctx.agent_id == "alice"
    assert ctx.session_id == "session-1"
    assert ctx.timeline is timeline
    assert ctx.event.payload_ref is start
    assert ctx.network.nodes == {}
    assert ctx.route_planner.topology is ctx.topology


def test_agent_timer_dispatches_to_on_timer() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    timer = TimerFired(
        timer_id="retry",
        owner_agent_id="alice",
        scheduled_at=0,
        fires_at=1,
    )

    agent.handle_event(make_event(agent, AGENT_TIMER, timer), timeline)

    assert agent.seen[0][0] == "timer"
    assert agent.seen[0][1] is timer


def test_agent_report_with_port_delivery_dispatches_unwrapped_report() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    report = {"click": 1}
    delivery = make_delivery(agent.reports.port("detector"), report)

    agent.handle_event(make_event(agent, AGENT_REPORT, delivery), timeline)

    assert agent.seen[0][0] == "report"
    assert agent.seen[0][1] is report


def test_agent_report_with_agent_report_dispatches_report() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    report = object()

    agent.handle_event(
        make_event(agent, AGENT_REPORT, AgentReport(report=report)), timeline
    )

    assert agent.seen[0][0] == "report"
    assert agent.seen[0][1] is report


def test_agent_report_with_raw_payload_dispatches_raw_payload() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    report = object()

    agent.handle_event(make_event(agent, AGENT_REPORT, report), timeline)

    assert agent.seen[0][0] == "report"
    assert agent.seen[0][1] is report


def test_agent_message_with_classical_message_dispatches_to_on_message() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    message = make_message()

    agent.handle_event(make_event(agent, AGENT_MESSAGE, message), timeline)

    assert agent.seen[0][0] == "message"
    agent_message = agent.seen[0][1]
    assert isinstance(agent_message, AgentMessage)
    assert agent_message.message is message


def test_agent_message_with_port_delivery_unwraps_and_dispatches() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    message = make_message()
    sink = PassiveComponent(
        component_id="sink",
        port_name="in",
        direction=PortDirection.INGRESS,
    )
    delivery = make_delivery(sink.port, message)

    agent.handle_event(make_event(agent, AGENT_MESSAGE, delivery), timeline)

    assert agent.seen[0][0] == "message"
    agent_message = agent.seen[0][1]
    assert isinstance(agent_message, AgentMessage)
    assert agent_message.message is message


def test_agent_message_rejects_raw_non_classical_message() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)

    with pytest.raises(TypeError, match="payload_ref must be ClassicalMessage"):
        agent.handle_event(make_event(agent, AGENT_MESSAGE, "bad"), timeline)


def test_agent_message_rejects_port_delivery_non_classical_message() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    sink = PassiveComponent(
        component_id="sink",
        port_name="in",
        direction=PortDirection.INGRESS,
    )
    delivery = make_delivery(sink.port, "bad")

    with pytest.raises(
        TypeError, match="PortDelivery payload must be ClassicalMessage"
    ):
        agent.handle_event(make_event(agent, AGENT_MESSAGE, delivery), timeline)


def test_agent_event_dispatches_to_on_event() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    event = make_event(agent, AGENT_EVENT, {"custom": True})

    agent.handle_event(event, timeline)

    assert agent.seen[0] == ("event", event, agent.seen[0][2])


def test_unsupported_action_dispatches_to_on_event() -> None:
    timeline = Timeline(master_seed=1)
    agent = RecordingAgent(agent_id="alice")
    attach_context(agent, timeline)
    event = make_event(agent, "custom_action", None)

    agent.handle_event(event, timeline)

    assert agent.seen[0][0] == "event"
    assert agent.seen[0][1] is event


def test_default_agent_unsupported_action_raises() -> None:
    timeline = Timeline(master_seed=1)
    agent = Agent(agent_id="alice")
    attach_context(agent, timeline)

    with pytest.raises(ValueError, match="unsupported event action"):
        agent.handle_event(make_event(agent, "custom_action", None), timeline)
