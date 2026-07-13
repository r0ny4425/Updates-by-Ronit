from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components.connections import PortDelivery
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.network import Network, Node

ACTION_RECEIVE_CLASSICAL = "receive_classical"


@dataclass(slots=True)
class Sender(Component):
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

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("sender should not receive events")


@dataclass(slots=True)
class Receiver(Component):
    device_id: str = "receiver"
    received: list[tuple[int, str, PortDelivery]] = field(default_factory=list)
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
        if event.action != ACTION_RECEIVE_CLASSICAL:
            raise ValueError(event.action)
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("wrong input port")
        self.received.append((timeline.current_time, event.action, event.payload_ref))


def test_network_connection_delivers_with_target_action_and_port_delivery() -> None:
    timeline = Timeline(master_seed=1)

    network = Network("delivery")
    alice = Node("alice")
    bob = Node("bob")

    sender = Sender("alice_sender")
    receiver = Receiver("bob_receiver")

    alice.add_device("sender", sender)
    alice.register_port("c_out", sender.output_port)

    bob.add_device("receiver", receiver)
    bob.register_port("c_in", receiver.input_port)

    network.add_node(alice)
    network.add_node(bob)

    connection = network.wire_ports(
        "c_wire_ab",
        sender.output_port,
        receiver.input_port,
        target_action=ACTION_RECEIVE_CLASSICAL,
    )

    connection.transmit("hello", timeline, time=0, source=sender)

    timeline.run_until(0)

    assert len(receiver.received) == 1

    time, action, delivery = receiver.received[0]
    assert time == 0
    assert action == ACTION_RECEIVE_CLASSICAL
    assert delivery.payload == "hello"
    assert delivery.source_port is sender.output_port
    assert delivery.target_port is receiver.input_port
    assert delivery.connection_id == "c_wire_ab"
    assert network.edges == ()
