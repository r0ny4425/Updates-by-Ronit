from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.network import Network, Node
from tests.network._components import QuantumSink, QuantumSource


def test_network_wires_direct_component_ports_without_topology_edge() -> None:
    network = Network("direct")
    source = QuantumSource()
    sink = QuantumSink()

    wire = network.wire_ports(
        "q_wire",
        source.output_port,
        sink.input_port,
        target_action="receive_signal",
    )

    assert wire.connection_id == "q_wire"
    assert wire.target_action == "receive_signal"
    assert network.get_wire("q_wire") is wire
    assert network.links == {}
    assert network.edges == ()
    assert set(network.quantum_links) == set()
    assert set(network.classical_links) == set()


def test_network_adds_explicit_quantum_link_without_node_owned_channel() -> None:
    network = Network("links")
    channel = object()

    network.add_node(Node("alice"))
    network.add_node(Node("bob"))

    link = network.add_quantum_link(
        "q_link_ab",
        "alice",
        "bob",
        channel=channel,
    )

    assert link.link_id == "q_link_ab"
    assert link.source_node_id == "alice"
    assert link.target_node_id == "bob"
    assert link.port_kind is PortKind.QUANTUM
    assert link.transport is channel
    assert network.get_link("q_link_ab") is link
    assert set(network.quantum_links) == {"q_link_ab"}
    assert set(network.classical_links) == set()


def test_network_rejects_duplicate_node_ids() -> None:
    network = Network("duplicate_nodes")

    network.add_node(Node("alice"))

    with pytest.raises(ValueError, match="already exists"):
        network.add_node(Node("alice"))


def test_network_rejects_duplicate_link_ids() -> None:
    network = Network("duplicate_links")
    network.add_node(Node("alice"))
    network.add_node(Node("bob"))

    network.add_quantum_link("q_link", "alice", "bob")

    with pytest.raises(ValueError, match="already exists"):
        network.add_quantum_link("q_link", "alice", "bob")


def test_network_rejects_duplicate_wire_ids() -> None:
    network = Network("duplicate_wires")
    source = QuantumSource()
    sink = QuantumSink()

    network.wire_ports(
        "q_wire",
        source.output_port,
        sink.input_port,
        target_action="receive_signal",
    )

    source2 = QuantumSource("source2")
    sink2 = QuantumSink("sink2")

    with pytest.raises(ValueError, match="already exists"):
        network.wire_ports(
            "q_wire",
            source2.output_port,
            sink2.input_port,
            target_action="receive_signal",
        )


def test_network_rejects_link_with_unknown_node() -> None:
    network = Network("bad_link")
    network.add_node(Node("alice"))

    with pytest.raises(KeyError):
        network.add_quantum_link("bad_link", "alice", "bob")


def test_network_rejects_invalid_link_port_kind() -> None:
    network = Network("bad_port_kind")
    network.add_node(Node("alice"))
    network.add_node(Node("bob"))

    with pytest.raises(TypeError, match="port_kind must be PortKind"):
        network.add_link(
            "bad_link",
            "alice",
            "bob",
            port_kind="quantum",  # type: ignore[arg-type]
        )
