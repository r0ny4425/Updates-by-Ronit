from __future__ import annotations

from dataclasses import dataclass, field

from pytest import raises

from simyuj.components import Port, PortDirection, PortKind
from simyuj.components.memories import QuantumMemory
from simyuj.control import AgentContext, NodeAgent, SessionRuntime
from simyuj.control.devices import DeviceResolver
from simyuj.engine import Component, Timeline
from simyuj.network import Network, Node


@dataclass(slots=True)
class PortOwner(Component):
    component_id: str = "owner"
    port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.port = Port(
            name="out",
            owner=self,
            owner_id=self.component_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("owner should not receive events")


@dataclass(slots=True)
class DeviceAgent(NodeAgent):
    contexts: list[AgentContext] = field(default_factory=list)

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        self.contexts.append(ctx)


def node_with_devices() -> tuple[Node, QuantumMemory, PortOwner]:
    node = Node("alice")
    qmem = QuantumMemory(memory_id="mem-a", num_positions=2)
    owner = PortOwner()
    node.add_device("memory", qmem)
    node.add_device("plain", object())
    node.register_port("out", owner.port)
    return node, qmem, owner


def test_get_returns_node_local_device() -> None:
    node, qmem, _ = node_with_devices()
    resolver = DeviceResolver(node=node)

    assert resolver.get("memory") is qmem


def test_memory_returns_quantum_memory() -> None:
    node, qmem, _ = node_with_devices()
    resolver = DeviceResolver(node=node)

    assert resolver.memory("memory") is qmem


def test_memory_rejects_non_memory_device() -> None:
    node, _, _ = node_with_devices()
    resolver = DeviceResolver(node=node)

    with raises(TypeError, match="not QuantumMemory"):
        resolver.memory("plain")


def test_port_returns_node_local_alias() -> None:
    node, _, owner = node_with_devices()
    resolver = DeviceResolver(node=node)

    assert resolver.port("out") is owner.port


def test_runtime_context_adds_devices_for_node_agent() -> None:
    timeline = Timeline(master_seed=1)
    network = Network()
    node, qmem, _ = node_with_devices()
    network.add_node(node)
    agent = DeviceAgent(agent_id="alice-agent", node_id="alice")
    node.add_agent(agent)
    runtime = SessionRuntime(timeline=timeline, network=network)

    runtime.run()

    assert agent.contexts[0].devices is not None
    assert agent.contexts[0].devices.memory("memory") is qmem
