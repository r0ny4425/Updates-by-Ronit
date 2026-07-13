from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from simyuj.components import (
    ACTION_RECEIVE_CLASSICAL,
    ACTION_TRANSMIT_CLASSICAL,
    DEFAULT_FIBER_LIGHT_SPEED_M_PER_S,
    ClassicalChannel,
    Port,
    PortDelivery,
    PortDirection,
    PortKind,
    connect_ports,
)
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.messages.transport import ClassicalMessage
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.binding import binding_context


@dataclass(slots=True)
class ClassicalSender(Component):
    device_id: str = "sender"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        raise AssertionError("sender should not receive events in these tests")


@dataclass(slots=True)
class ClassicalSink(Component):
    device_id: str = "sink"
    received: list[tuple[int, PortDelivery]] = field(default_factory=list)
    event_meta: list[dict[str, Any]] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        if event.action != ACTION_RECEIVE_CLASSICAL:
            raise ValueError(event.action)
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("delivery arrived on unknown port")
        if not isinstance(event.payload_ref.payload, ClassicalMessage):
            raise TypeError("payload must be ClassicalMessage")
        self.received.append((timeline.current_time, event.payload_ref))
        self.event_meta.append(dict(event.meta))


def _message(
    *,
    sent_time: int = 0,
    body: str | bytes = "hello",
) -> ClassicalMessage:
    return ClassicalMessage(
        sender_id="alice",
        receiver_id="bob",
        body=body,
        sent_time=sent_time,
        message_id="m1",
        message_type="test",
    )


def _wired_classical_chain(
    *,
    delay_ticks: int | None = None,
    length_m: float = 0.0,
    fiber_speed_m_per_s: float = DEFAULT_FIBER_LIGHT_SPEED_M_PER_S,
    loss_probability: float = 0.0,
    session_id: str | None = None,
):
    sender = ClassicalSender()
    channel = ClassicalChannel(
        channel_id="alice_to_bob",
        delay_ticks=delay_ticks,
        length_m=length_m,
        fiber_speed_m_per_s=fiber_speed_m_per_s,
        loss_probability=loss_probability,
        session_id=session_id,
    )
    sink = ClassicalSink()

    input_connection = connect_ports(
        sender.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
        connection_id="sender.out->channel.in",
    )
    output_connection = connect_ports(
        channel.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_CLASSICAL,
        connection_id="channel.out->sink.in",
    )

    return sender, channel, sink, input_connection, output_connection


def test_classical_channel_ports_are_structural_classical_ports() -> None:
    channel = ClassicalChannel(channel_id="alice_to_bob")

    assert channel.input_port.owner is channel
    assert channel.input_port.port_kind is PortKind.CLASSICAL
    assert channel.input_port.direction is PortDirection.INGRESS
    assert channel.output_port.owner is channel
    assert channel.output_port.port_kind is PortKind.CLASSICAL
    assert channel.output_port.direction is PortDirection.EGRESS

    assert not hasattr(channel.input_port, "handle_event")
    assert not hasattr(channel.output_port, "handle_event")


def test_fixed_delay_delivers_to_downstream_component() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, output_connection = _wired_classical_chain(
        delay_ticks=10
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)

    timeline.run_until(9)
    assert sink.received == []
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.dropped_count == 0

    timeline.run_until(10)

    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 10
    assert delivery.payload is message
    assert delivery.source_port is channel.output_port
    assert delivery.target_port is sink.input_port
    assert delivery.connection_id == output_connection.connection_id
    assert "session_id" not in sink.event_meta[0]


def test_zero_delay_still_targets_receiver_component_not_port() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(delay_ticks=0)
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 0
    assert delivery.payload is message
    assert delivery.target_port is sink.input_port
    assert delivery.target_port is not sink
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.dropped_count == 0


def test_channel_event_metadata_is_minimal() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(
        delay_ticks=0,
        length_m=200.0,
        session_id="session-1",
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    event_meta = sink.event_meta[0]
    assert event_meta["connection_id"] == "channel.out->sink.in"
    assert event_meta["source_owner_id"] == channel.channel_id
    assert event_meta["source_port"] == "out"
    assert event_meta["target_owner_id"] == sink.device_id
    assert event_meta["target_port"] == "in"
    assert event_meta["target_action"] == ACTION_RECEIVE_CLASSICAL
    assert event_meta["channel_id"] == channel.channel_id
    assert event_meta["message_id"] == message.message_id
    assert event_meta["message_type"] == message.message_type
    assert event_meta["session_id"] == "session-1"

    for key in (
        "input_port",
        "output_port",
        "source_event_id",
        "source_connection_id",
        "delay_ticks",
        "length_m",
        "fiber_speed_m_per_s",
        "delay_source",
    ):
        assert key not in event_meta


def test_loss_probability_one_drops_message() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(
        delay_ticks=10,
        loss_probability=1.0,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(100)

    assert sink.received == []
    assert channel.received_count == 1
    assert channel.delivered_count == 0
    assert channel.dropped_count == 1


def test_loss_probability_zero_delivers_message() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(
        delay_ticks=5,
        loss_probability=0.0,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(5)

    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 5
    assert delivery.payload is message
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.dropped_count == 0


def test_classical_channel_logs_ready_and_forwarded_message() -> None:
    log_sink = MemorySink()
    timeline = Timeline(
        master_seed=1,
        logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]),
    )
    sender, channel, _, input_connection, output_connection = _wired_classical_chain(
        delay_ticks=5,
        loss_probability=0.0,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(5)

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.classical.ready"
    )
    forwarded = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.classical.message_forwarded"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "channel_id": channel.channel_id,
        "delay_ticks": 5,
        "loss_probability": 0.0,
        "delivery_priority": 0,
    }

    assert forwarded.level is LogLevel.DEBUG
    assert forwarded.action == ACTION_TRANSMIT_CLASSICAL
    assert dict(forwarded.meta) == {
        "channel_id": channel.channel_id,
        "message_id": message.message_id,
        "message_type": message.message_type,
        "received_index": 1,
        "connection_id": output_connection.connection_id,
        "delay_ticks": 5,
        "arrival_time": 5,
    }


def test_classical_channel_logs_dropped_message_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(
        master_seed=1,
        logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]),
    )
    sender, channel, _, input_connection, _ = _wired_classical_chain(
        delay_ticks=5,
        loss_probability=1.0,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)
    timeline.run_until(5)

    dropped = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.classical.message_dropped"
    )

    assert dropped.level is LogLevel.DEBUG
    assert dropped.action == ACTION_TRANSMIT_CLASSICAL
    assert dict(dropped.meta) == {
        "channel_id": channel.channel_id,
        "message_id": message.message_id,
        "message_type": message.message_type,
        "received_index": 1,
        "loss_probability": 1.0,
    }


def test_length_based_delay_uses_fiber_speed_when_delay_ticks_not_given() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(
        length_m=200.0,
        fiber_speed_m_per_s=2.0e8,
        delay_ticks=None,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)

    # 200 m / 2e8 m/s = 1e-6 s
    # 1 tick = 1 ps, so 1e-6 s = 1,000,000 ticks
    assert channel.resolved_delay_ticks == 1_000_000

    timeline.run_until(999_999)
    assert sink.received == []

    timeline.run_until(1_000_000)
    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 1_000_000
    assert delivery.payload is message


def test_delay_ticks_overrides_length_based_delay() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(
        length_m=200.0,
        fiber_speed_m_per_s=2.0e8,
        delay_ticks=7,
    )
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(message, timeline, time=0, source=sender)

    assert channel.resolved_delay_ticks == 7

    timeline.run_until(6)
    assert sink.received == []

    timeline.run_until(7)
    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 7
    assert delivery.payload is message


def test_manual_channel_event_requires_bind_before_execution() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, _, input_connection, _ = _wired_classical_chain(delay_ticks=0)
    message = _message(sent_time=0)

    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action=ACTION_TRANSMIT_CLASSICAL,
            payload_ref=PortDelivery(
                payload=message,
                source_port=sender.output_port,
                target_port=channel.input_port,
                connection_id=input_connection.connection_id,
            ),
            source=None,
            subsystem_id="components",
        )
    )

    with pytest.raises(RuntimeError, match="must be bound"):
        timeline.run_until(0)


def test_manual_channel_event_works_after_bind() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_classical_chain(delay_ticks=0)
    message = _message(sent_time=0)

    channel.bind(binding_context(timeline))
    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action=ACTION_TRANSMIT_CLASSICAL,
            payload_ref=PortDelivery(
                payload=message,
                source_port=sender.output_port,
                target_port=channel.input_port,
                connection_id=input_connection.connection_id,
            ),
            source=None,
            subsystem_id="components",
        )
    )

    timeline.run_until(0)

    assert len(sink.received) == 1
    assert sink.received[0][1].payload is message


def test_channel_rejects_raw_message_payload_ref() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, _, _, _ = _wired_classical_chain(delay_ticks=0)

    channel.bind(binding_context(timeline))
    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action=ACTION_TRANSMIT_CLASSICAL,
            payload_ref=_message(),
            source=sender,
            subsystem_id="components",
        )
    )

    with pytest.raises(TypeError, match="PortDelivery"):
        timeline.run_until(0)


def test_delivery_payload_must_be_classical_message() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, _, input_connection, _ = _wired_classical_chain(delay_ticks=0)

    channel.bind(binding_context(timeline))
    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action=ACTION_TRANSMIT_CLASSICAL,
            payload_ref=PortDelivery(
                payload="not a ClassicalMessage",
                source_port=sender.output_port,
                target_port=channel.input_port,
                connection_id=input_connection.connection_id,
            ),
            source=sender,
            subsystem_id="components",
        )
    )

    with pytest.raises(TypeError, match="ClassicalMessage"):
        timeline.run_until(0)


def test_channel_rejects_delivery_to_wrong_port() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, _, _ = _wired_classical_chain(delay_ticks=0)

    channel.bind(binding_context(timeline))

    wrong_delivery = PortDelivery(
        payload=_message(),
        source_port=sender.output_port,
        target_port=sink.input_port,
        connection_id="wrong",
    )

    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action=ACTION_TRANSMIT_CLASSICAL,
            payload_ref=wrong_delivery,
            source=sender,
            subsystem_id="components",
        )
    )

    with pytest.raises(ValueError, match="unknown port"):
        timeline.run_until(0)


def test_channel_requires_connected_output_when_message_survives() -> None:
    timeline = Timeline(master_seed=1)
    sender = ClassicalSender()
    channel = ClassicalChannel(
        channel_id="alice_to_bob",
        delay_ticks=0,
        loss_probability=0.0,
    )

    input_connection = connect_ports(
        sender.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
    )

    channel.bind(binding_context(timeline))
    input_connection.transmit(_message(), timeline, time=0, source=sender)

    with pytest.raises(RuntimeError, match="not connected"):
        timeline.run_until(0)


def test_input_connection_rejects_past_time() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, _, input_connection, _ = _wired_classical_chain(delay_ticks=0)

    channel.bind(binding_context(timeline))
    input_connection.transmit(_message(), timeline, time=5, source=sender)
    timeline.run_until(5)

    with pytest.raises(ValueError, match="past"):
        input_connection.transmit(_message(), timeline, time=4, source=sender)


def test_unsupported_action_rejected() -> None:
    timeline = Timeline(master_seed=1)
    _, channel, _, _, _ = _wired_classical_chain(delay_ticks=0)

    channel.bind(binding_context(timeline))
    timeline.schedule(
        Event(
            time=0,
            target_ref=channel,
            action="wrong_action",
            payload_ref=_message(),
            source=None,
            subsystem_id="components",
        )
    )

    with pytest.raises(ValueError, match="unsupported event action"):
        timeline.run_until(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_id": ""},
        {"delay_ticks": -1},
        {"length_m": -1.0},
        {"length_m": float("inf")},
        {"fiber_speed_m_per_s": 0.0},
        {"fiber_speed_m_per_s": -1.0},
        {"fiber_speed_m_per_s": float("inf")},
        {"loss_probability": -0.1},
        {"loss_probability": 1.1},
        {"loss_probability": float("inf")},
    ],
)
def test_invalid_channel_values_rejected(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"channel_id": "alice_to_bob"}
    base.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ClassicalChannel(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delay_ticks": True},
        {"length_m": True},
        {"fiber_speed_m_per_s": True},
        {"loss_probability": True},
        {"delivery_priority": True},
    ],
)
def test_boolean_numeric_fields_rejected(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"channel_id": "alice_to_bob"}
    base.update(kwargs)

    with pytest.raises(TypeError):
        ClassicalChannel(**base)
