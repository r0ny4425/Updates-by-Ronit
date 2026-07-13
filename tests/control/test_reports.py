from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import Port, PortDelivery, PortDirection, PortKind
from simyuj.control.payloads import AgentReport
from simyuj.control.reports import AgentReportEndpoint
from simyuj.engine.component import Component


@dataclass(slots=True)
class AgentStub(Component):
    agent_id: str = "agent"
    seen: list[object] = field(default_factory=list)

    def handle_event(self, event, timeline) -> None:
        self.seen.append(event.payload_ref)


@dataclass(slots=True)
class SourceStub(Component):
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


def make_delivery(target_port: Port, payload: object = "report") -> PortDelivery:
    source = SourceStub()
    return PortDelivery(
        payload=payload,
        source_port=source.output_port,
        target_port=target_port,
        connection_id="source.out->agent.report",
    )


def test_endpoint_is_not_component() -> None:
    endpoint = AgentReportEndpoint(AgentStub())

    assert not isinstance(endpoint, Component)


def test_default_report_port_is_owned_by_agent() -> None:
    agent = AgentStub(agent_id="alice")
    endpoint = AgentReportEndpoint(agent)

    port = endpoint.input_port

    assert port.owner is agent
    assert port.owner_id == "alice"
    assert port.name == "report"
    assert port.port_kind is PortKind.CLASSICAL
    assert port.direction is PortDirection.INGRESS


def test_named_report_ports_are_owned_by_agent() -> None:
    agent = AgentStub(agent_id="alice")
    endpoint = AgentReportEndpoint(agent)

    port = endpoint.port("detector")

    assert port.owner is agent
    assert port.owner_id == "alice"
    assert port.name == "detector"


def test_named_report_port_is_stable_across_calls() -> None:
    endpoint = AgentReportEndpoint(AgentStub())

    assert endpoint.port("detector") is endpoint.port("detector")


def test_named_report_ports_are_distinct() -> None:
    endpoint = AgentReportEndpoint(AgentStub())

    assert endpoint.port("detector") is not endpoint.port("memory")


def test_ports_are_returned_in_name_order() -> None:
    endpoint = AgentReportEndpoint(AgentStub())
    memory = endpoint.port("memory")
    detector = endpoint.port("detector")

    assert endpoint.ports == (detector, memory)


def test_extract_port_delivery_returns_report_for_owned_port() -> None:
    endpoint = AgentReportEndpoint(AgentStub())
    report = {"click": 1}
    delivery = make_delivery(endpoint.port("detector"), report)

    assert endpoint.extract(delivery) is report


def test_extract_port_delivery_rejects_unknown_target_port() -> None:
    endpoint = AgentReportEndpoint(AgentStub(agent_id="alice"))
    other = AgentReportEndpoint(AgentStub(agent_id="bob"))

    with pytest.raises(ValueError, match="target_port is not an agent report port"):
        endpoint.extract(make_delivery(other.port("detector")))


def test_extract_agent_report_returns_wrapped_report() -> None:
    endpoint = AgentReportEndpoint(AgentStub())
    report = {"memory": "stored"}

    assert endpoint.extract(AgentReport(report=report)) is report


def test_extract_raw_report_returns_raw_payload() -> None:
    endpoint = AgentReportEndpoint(AgentStub())
    report = object()

    assert endpoint.extract(report) is report


def test_rejects_non_component_owner() -> None:
    with pytest.raises(TypeError, match="owner_agent must be Component"):
        AgentReportEndpoint(object())  # type: ignore[arg-type]


def test_rejects_owner_without_non_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="non-empty agent_id"):
        AgentReportEndpoint(AgentStub(agent_id=""))


def test_rejects_empty_report_port_name() -> None:
    endpoint = AgentReportEndpoint(AgentStub())

    with pytest.raises(ValueError, match="report_port_name must be non-empty"):
        endpoint.port("")
