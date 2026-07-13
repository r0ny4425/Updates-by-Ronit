from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.network import Network, Node, TopologyEdge
from simyuj.network.topology import NetworkTopology
from tests.network._components import QuantumSink, QuantumSource


def add_quantum_link(
    network: Network,
    link_id: str,
    source_node_id: str,
    target_node_id: str,
) -> None:
    network.add_quantum_link(link_id, source_node_id, target_node_id)


def add_classical_link(
    network: Network,
    link_id: str,
    source_node_id: str,
    target_node_id: str,
) -> None:
    network.add_classical_link(link_id, source_node_id, target_node_id)


def make_network(*node_ids: str) -> Network:
    network = Network("topology")
    for node_id in node_ids:
        network.add_node(Node(node_id))
    return network


def test_network_topology_exposes_explicit_links_only() -> None:
    network = make_network("bob", "alice")
    direct_source = QuantumSource("direct_source")
    direct_sink = QuantumSink("direct_sink")

    network.add_quantum_link("q_link_ab", "alice", "bob")
    network.wire_ports(
        "direct_wire",
        direct_source.output_port,
        direct_sink.input_port,
        target_action="receive_signal",
    )

    topology = NetworkTopology(network)

    assert topology.nodes() == ("alice", "bob")
    assert topology.edges == (
        TopologyEdge(
            link_id="q_link_ab",
            source_node_id="alice",
            target_node_id="bob",
            port_kind=PortKind.QUANTUM,
        ),
    )
    assert topology.outgoing_edges("alice") == topology.edges
    assert topology.incoming_edges("bob") == topology.edges
    assert topology.neighbors("alice") == ("bob",)
    assert topology.has_edge("alice", "bob")
    assert topology.has_edge("alice", "bob", port_kind=PortKind.QUANTUM)
    assert not topology.has_edge("bob", "alice")


def test_network_exposes_topology_queries_directly() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_classical_link(network, "c_ab", "alice", "bob")

    assert tuple(edge.link_id for edge in network.edges) == ("c_ab", "q_ab")
    assert tuple(
        edge.link_id
        for edge in network.outgoing_edges(
            "alice",
            port_kind=PortKind.QUANTUM,
        )
    ) == ("q_ab",)
    assert tuple(
        edge.link_id
        for edge in network.incoming_edges(
            "bob",
            port_kind=PortKind.CLASSICAL,
        )
    ) == ("c_ab",)
    assert network.neighbors("alice") == ("bob",)
    assert network.has_edge("alice", "bob", port_kind=PortKind.QUANTUM)
    assert not network.has_edge("bob", "alice")


def test_network_topology_queries_ignore_runtime_wires() -> None:
    network = make_network("alice", "bob")
    source = QuantumSource("direct_source")
    sink = QuantumSink("direct_sink")

    network.wire_ports(
        "direct",
        source.output_port,
        sink.input_port,
        target_action="receive_signal",
    )

    assert network.wires.keys() == {"direct"}
    assert network.edges == ()
    assert network.neighbors("alice") == ()
    assert not network.has_edge("alice", "bob", port_kind=PortKind.QUANTUM)


def test_network_topology_is_live_view() -> None:
    network = make_network("alice", "bob", "carol")
    topology = NetworkTopology(network)

    assert topology.edges == ()

    network.add_quantum_link("q_link_ab", "alice", "bob")

    assert tuple(edge.link_id for edge in topology.edges) == ("q_link_ab",)

    network.add_quantum_link("q_link_bc", "bob", "carol")

    assert tuple(edge.link_id for edge in topology.edges) == (
        "q_link_ab",
        "q_link_bc",
    )


def test_network_topology_filters_edges_by_port_kind() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_classical_link(network, "c_ab", "alice", "bob")
    topology = NetworkTopology(network)

    assert tuple(
        edge.link_id
        for edge in topology.outgoing_edges("alice", port_kind=PortKind.QUANTUM)
    ) == ("q_ab",)
    assert tuple(
        edge.link_id
        for edge in topology.incoming_edges("bob", port_kind=PortKind.CLASSICAL)
    ) == ("c_ab",)
    assert topology.neighbors("alice", port_kind=PortKind.CLASSICAL) == ("bob",)
    assert not topology.has_edge("bob", "alice", port_kind=PortKind.CLASSICAL)


def test_network_topology_validates_inputs() -> None:
    network = make_network("alice", "bob")
    topology = NetworkTopology(network)

    with pytest.raises(TypeError, match="network must be Network"):
        NetworkTopology(object())  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="unknown node id"):
        topology.outgoing_edges("carol")

    with pytest.raises(KeyError, match="unknown node id"):
        topology.has_edge("alice", "carol")

    with pytest.raises(TypeError, match="PortKind or None"):
        topology.incoming_edges("bob", port_kind="quantum")  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="unknown node id"):
        network.outgoing_edges("carol")

    with pytest.raises(TypeError, match="PortKind or None"):
        network.incoming_edges("bob", port_kind="quantum")  # type: ignore[arg-type]


def test_topology_edge_validates_fields() -> None:
    with pytest.raises(ValueError, match="link_id must be non-empty"):
        TopologyEdge(
            link_id="",
            source_node_id="alice",
            target_node_id="bob",
            port_kind=PortKind.QUANTUM,
        )

    with pytest.raises(TypeError, match="port_kind must be PortKind"):
        TopologyEdge(
            link_id="q_ab",
            source_node_id="alice",
            target_node_id="bob",
            port_kind="quantum",  # type: ignore[arg-type]
        )
