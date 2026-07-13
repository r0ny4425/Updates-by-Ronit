from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import Port, PortDelivery, PortDirection, PortKind, connect_ports
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline

ACTION_RECEIVE_TEST = "receive_test"
ACTION_OTHER_TEST = "other_test"


@dataclass(slots=True)
class SourceStub(Component):
    device_id: str = "source"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("source should not receive events in these tests")


@dataclass(slots=True)
class SinkStub(Component):
    device_id: str = "sink"
    received: list[tuple[str, PortDelivery]] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append((event.action, event.payload_ref))


def test_connect_ports_binds_both_sides() -> None:
    source = SourceStub()
    sink = SinkStub()

    assert source.output_port.is_connected is False
    assert sink.input_port.is_connected is False

    connection = connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    assert source.output_port.connection is connection
    assert sink.input_port.connection is connection
    assert source.output_port.is_connected is True
    assert sink.input_port.is_connected is True
    assert connection.source_port is source.output_port
    assert connection.target_port is sink.input_port
    assert connection.connection_id == "source.out->sink.in"
    assert connection.target_action == ACTION_RECEIVE_TEST


def test_connection_transmit_schedules_target_component_not_port() -> None:
    timeline = Timeline(master_seed=1)
    source = SourceStub()
    sink = SinkStub()
    connection = connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    event = connection.transmit("hello", timeline)

    assert event.target_ref is sink
    assert event.target_ref is not sink.input_port
    assert event.action == ACTION_RECEIVE_TEST
    assert event.meta["connection_id"] == "source.out->sink.in"
    assert event.meta["source_owner_id"] == "source"
    assert event.meta["source_port"] == "out"
    assert event.meta["target_owner_id"] == "sink"
    assert event.meta["target_port"] == "in"
    assert event.meta["target_action"] == ACTION_RECEIVE_TEST

    timeline.run_until(0)

    assert len(sink.received) == 1
    action, delivery = sink.received[0]
    assert action == ACTION_RECEIVE_TEST
    assert delivery.payload == "hello"
    assert delivery.source_port is source.output_port
    assert delivery.target_port is sink.input_port
    assert delivery.connection_id == connection.connection_id


def test_connection_transmit_can_override_default_action() -> None:
    timeline = Timeline(master_seed=1)
    source = SourceStub()
    sink = SinkStub()
    connection = connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    connection.transmit("hello", timeline, action=ACTION_OTHER_TEST)
    timeline.run_until(0)

    action, delivery = sink.received[0]
    assert action == ACTION_OTHER_TEST
    assert delivery.payload == "hello"


def test_rejects_input_to_input() -> None:
    a = SinkStub(device_id="a")
    b = SinkStub(device_id="b")

    with pytest.raises(ValueError, match="EGRESS"):
        connect_ports(
            a.input_port,
            b.input_port,
            target_action=ACTION_RECEIVE_TEST,
        )


def test_rejects_output_to_output() -> None:
    a = SourceStub(device_id="a")
    b = SourceStub(device_id="b")

    with pytest.raises(ValueError, match="INGRESS"):
        connect_ports(
            a.output_port,
            b.output_port,
            target_action=ACTION_RECEIVE_TEST,
        )


def test_rejects_kind_mismatch() -> None:
    source = SourceStub()
    sink = SinkStub()
    sink.input_port = Port(
        name="qin",
        owner=sink,
        owner_id=sink.device_id,
        port_kind=PortKind.QUANTUM,
        direction=PortDirection.INGRESS,
    )

    with pytest.raises(ValueError, match="same port_kind"):
        connect_ports(
            source.output_port,
            sink.input_port,
            target_action=ACTION_RECEIVE_TEST,
        )


def test_rejects_duplicate_source_connection() -> None:
    source = SourceStub()
    sink1 = SinkStub(device_id="sink1")
    sink2 = SinkStub(device_id="sink2")

    connect_ports(
        source.output_port,
        sink1.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    with pytest.raises(ValueError, match="already connected"):
        connect_ports(
            source.output_port,
            sink2.input_port,
            target_action=ACTION_RECEIVE_TEST,
        )


def test_rejects_duplicate_target_connection() -> None:
    source1 = SourceStub(device_id="source1")
    source2 = SourceStub(device_id="source2")
    sink = SinkStub()

    connect_ports(
        source1.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    with pytest.raises(ValueError, match="already connected"):
        connect_ports(
            source2.output_port,
            sink.input_port,
            target_action=ACTION_RECEIVE_TEST,
        )


def test_rejects_transmit_into_past() -> None:
    timeline = Timeline(master_seed=1)
    source = SourceStub()
    sink = SinkStub()
    connection = connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_TEST,
    )

    connection.transmit("first", timeline, time=5)
    timeline.run_until(5)

    with pytest.raises(ValueError, match="past"):
        connection.transmit("late", timeline, time=4)
