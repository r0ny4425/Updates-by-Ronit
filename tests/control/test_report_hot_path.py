from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import Port, PortDelivery, PortDirection, PortKind
from simyuj.control import AGENT_REPORT, Agent, AgentContext
from simyuj.control.reports import AgentReportEndpoint
from simyuj.engine import Component, Event, Timeline
from simyuj.network import Network
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology


@dataclass(slots=True)
class ReportSource(Component):
    component_id: str
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
        raise AssertionError("report source should not receive events")


@dataclass(slots=True)
class ReportAgent(Agent):
    reports_seen: list[object] = field(default_factory=list)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        del ctx
        self.reports_seen.append(report)


def attach_context(agent: Agent, timeline: Timeline) -> Network:
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
        )

    agent.attach_context_provider(provider)
    return network


def test_component_report_schedules_one_agent_report_event() -> None:
    timeline = Timeline(master_seed=1)
    agent = ReportAgent(agent_id="alice")
    network = attach_context(agent, timeline)
    source = ReportSource(component_id="detector")
    report = {"click": 1}

    network.wire_ports(
        "alice_detector_report",
        source.output_port,
        agent.reports.port("detector"),
        target_action=AGENT_REPORT,
    )
    assert source.output_port.connection is not None
    event = source.output_port.connection.transmit(report, timeline)

    assert timeline.events_scheduled == 1
    assert event.target_ref is agent
    assert not isinstance(event.target_ref, AgentReportEndpoint)
    assert event.action == AGENT_REPORT
    assert isinstance(event.payload_ref, PortDelivery)
    assert event.payload_ref.payload is report

    timeline.run_until_empty()

    assert agent.reports_seen == [report]


def test_report_fan_in_uses_distinct_named_ports() -> None:
    timeline = Timeline(master_seed=1)
    agent = ReportAgent(agent_id="alice")
    network = attach_context(agent, timeline)
    detector = ReportSource(component_id="detector")
    memory = ReportSource(component_id="memory")
    detector_report = ("detector", 1)
    memory_report = ("memory", 2)

    network.wire_ports(
        "alice_detector_report",
        detector.output_port,
        agent.reports.port("detector"),
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "alice_memory_report",
        memory.output_port,
        agent.reports.port("memory"),
        target_action=AGENT_REPORT,
    )

    assert detector.output_port.connection is not None
    assert memory.output_port.connection is not None
    detector.output_port.connection.transmit(detector_report, timeline)
    memory.output_port.connection.transmit(memory_report, timeline)

    assert timeline.events_scheduled == 2
    timeline.run_until_empty()

    assert agent.reports_seen == [detector_report, memory_report]


def test_two_sources_cannot_connect_to_same_default_report_port() -> None:
    agent = ReportAgent(agent_id="alice")
    network = Network()
    detector = ReportSource(component_id="detector")
    memory = ReportSource(component_id="memory")

    network.wire_ports(
        "alice_detector_report",
        detector.output_port,
        agent.report_port,
        target_action=AGENT_REPORT,
    )

    with pytest.raises(ValueError, match="already connected"):
        network.wire_ports(
            "alice_memory_report",
            memory.output_port,
            agent.report_port,
            target_action=AGENT_REPORT,
        )
