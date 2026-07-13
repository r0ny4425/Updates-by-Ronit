from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.network import Node


@dataclass(slots=True)
class StubComponent(Component):
    device_id: str
    output_port: Port = field(init=False)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        return None


def test_node_registers_devices_and_component_owned_ports() -> None:
    node = Node("alice")
    component = StubComponent("alice_device")

    assert node.add_device("device", component) is component
    assert node.get_device("device") is component

    assert node.register_port("c_out", component.output_port) is component.output_port
    assert node.get_port("c_out") is component.output_port

    assert node.get_port("c_out").owner is component
    assert node.get_port("c_out").owner_id == "alice_device"


def test_node_rejects_duplicate_device_names() -> None:
    node = Node("alice")
    node.add_device("device", object())

    with pytest.raises(ValueError, match="already exists"):
        node.add_device("device", object())


def test_node_rejects_duplicate_port_aliases() -> None:
    node = Node("alice")
    component = StubComponent("alice_device")

    node.register_port("c_out", component.output_port)

    with pytest.raises(ValueError, match="already exists"):
        node.register_port("c_out", component.input_port)


def test_node_rejects_non_port_alias() -> None:
    node = Node("alice")

    with pytest.raises(TypeError, match="Port"):
        node.register_port("bad", object())  # type: ignore[arg-type]
