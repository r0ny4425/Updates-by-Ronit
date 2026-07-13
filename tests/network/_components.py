from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component


@dataclass(slots=True)
class QuantumSource(Component):
    device_id: str = "q_source"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        return None


@dataclass(slots=True)
class QuantumSink(Component):
    device_id: str = "q_sink"
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        return None


@dataclass(slots=True)
class ClassicalSource(Component):
    device_id: str = "c_source"
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
        return None


@dataclass(slots=True)
class ClassicalSink(Component):
    device_id: str = "c_sink"
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
        return None
