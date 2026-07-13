from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.network import Network, Node, Route, TopologyEdge
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology


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
    network = Network("routes")
    for node_id in node_ids:
        network.add_node(Node(node_id))
    return network


def test_fewest_hops_path_source_equals_target_returns_empty_route() -> None:
    network = make_network("alice")
    planner = RoutePlanner(NetworkTopology(network))

    route = planner.fewest_hops_path(
        "alice",
        "alice",
        port_kind=PortKind.QUANTUM,
    )

    assert route == Route(source_node_id="alice", target_node_id="alice")
    assert route.hops == 0
    assert route.link_ids == ()
    assert route.node_ids == ("alice",)
    assert route.port_kinds == ()


def test_fewest_hops_path_returns_none_when_no_path_exists() -> None:
    network = make_network("alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    assert (
        planner.fewest_hops_path(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
        )
        is None
    )


def test_fewest_hops_path_returns_fewest_hop_route() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    route = planner.fewest_hops_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
    )

    assert route is not None
    assert route.link_ids == ("q_ab",)
    assert route.node_ids == ("alice", "bob")


def test_network_fewest_hops_path_delegates_to_route_planner() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")

    route = network.fewest_hops_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
    )

    assert route is not None
    assert route.link_ids == ("q_ab",)
    assert route.node_ids == ("alice", "bob")


def test_fewest_hops_path_is_directed() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    assert (
        planner.fewest_hops_path(
            "bob",
            "alice",
            port_kind=PortKind.QUANTUM,
        )
        is None
    )


def test_fewest_hops_path_requires_port_kind_filter() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_classical_link(network, "c_ab", "alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    quantum_route = planner.fewest_hops_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
    )
    classical_route = planner.fewest_hops_path(
        "alice",
        "bob",
        port_kind=PortKind.CLASSICAL,
    )

    assert quantum_route is not None
    assert quantum_route.link_ids == ("q_ab",)
    assert classical_route is not None
    assert classical_route.link_ids == ("c_ab",)


def test_lowest_cost_path_returns_lowest_additive_cost_route() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    route = planner.lowest_cost_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        link_cost=lambda link: {
            "q_ab": 10.0,
            "q_ar": 2.0,
            "q_rb": 3.0,
        }[link.link_id],
    )

    assert route is not None
    assert route.link_ids == ("q_ar", "q_rb")
    assert route.node_ids == ("alice", "relay", "bob")


def test_network_lowest_cost_path_delegates_to_route_planner() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")

    route = network.lowest_cost_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        link_cost=lambda link: 10.0 if link.link_id == "q_ab" else 1.0,
    )

    assert route is not None
    assert route.link_ids == ("q_ar", "q_rb")


def test_lowest_cost_path_uses_port_kind_filter() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_classical_link(network, "c_ab", "alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    quantum_route = planner.lowest_cost_path(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        link_cost=lambda link: 1.0,
    )
    classical_route = planner.lowest_cost_path(
        "alice",
        "bob",
        port_kind=PortKind.CLASSICAL,
        link_cost=lambda link: 1.0,
    )

    assert quantum_route is not None
    assert quantum_route.link_ids == ("q_ab",)
    assert classical_route is not None
    assert classical_route.link_ids == ("c_ab",)


def test_lowest_cost_path_returns_none_when_no_path_exists() -> None:
    network = make_network("alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    assert (
        planner.lowest_cost_path(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            link_cost=lambda link: 1.0,
        )
        is None
    )


def test_lowest_cost_path_source_equals_target_returns_empty_route() -> None:
    network = make_network("alice")
    planner = RoutePlanner(NetworkTopology(network))

    route = planner.lowest_cost_path(
        "alice",
        "alice",
        port_kind=PortKind.QUANTUM,
        link_cost=lambda link: 1.0,
    )

    assert route == Route(source_node_id="alice", target_node_id="alice")


def test_paths_with_max_hops_excludes_longer_paths() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    assert (
        planner.paths_with_max_hops(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=1,
        )
        == ()
    )
    assert tuple(
        route.link_ids
        for route in planner.paths_with_max_hops(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=2,
        )
    ) == (("q_ar", "q_rb"),)


def test_paths_with_max_hops_returns_simple_paths_only() -> None:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_ra", "relay", "alice")
    add_quantum_link(network, "q_rb", "relay", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    routes = planner.paths_with_max_hops(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=4,
    )

    assert tuple(route.link_ids for route in routes) == (("q_ar", "q_rb"),)


def test_paths_with_max_hops_preserves_parallel_edges() -> None:
    network = make_network("alice", "bob")
    add_quantum_link(network, "q_ab_1", "alice", "bob")
    add_quantum_link(network, "q_ab_2", "alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    routes = planner.paths_with_max_hops(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=1,
    )

    assert tuple(route.link_ids for route in routes) == (
        ("q_ab_1",),
        ("q_ab_2",),
    )


def test_paths_with_max_hops_uses_deterministic_depth_first_order() -> None:
    network = make_network("alice", "r1", "r2", "bob")
    add_quantum_link(network, "q_ar1", "alice", "r1")
    add_quantum_link(network, "q_ar2", "alice", "r2")
    add_quantum_link(network, "q_r1b", "r1", "bob")
    add_quantum_link(network, "q_r2b", "r2", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    routes = planner.paths_with_max_hops(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    assert tuple(route.link_ids for route in routes) == (
        ("q_ab",),
        ("q_ar1", "q_r1b"),
        ("q_ar2", "q_r2b"),
    )


def test_network_paths_with_max_hops_delegates_to_route_planner() -> None:
    network = make_network("alice", "r1", "r2", "bob")
    add_quantum_link(network, "q_ar1", "alice", "r1")
    add_quantum_link(network, "q_ar2", "alice", "r2")
    add_quantum_link(network, "q_r1b", "r1", "bob")
    add_quantum_link(network, "q_r2b", "r2", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")

    routes = network.paths_with_max_hops(
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    assert tuple(route.link_ids for route in routes) == (
        ("q_ab",),
        ("q_ar1", "q_r1b"),
        ("q_ar2", "q_r2b"),
    )


def test_paths_with_max_hops_source_equals_target_returns_empty_route() -> None:
    network = make_network("alice")
    planner = RoutePlanner(NetworkTopology(network))

    assert planner.paths_with_max_hops(
        "alice",
        "alice",
        port_kind=PortKind.QUANTUM,
        max_hops=0,
    ) == (Route(source_node_id="alice", target_node_id="alice"),)


def test_route_planner_validates_inputs() -> None:
    network = make_network("alice", "bob")
    planner = RoutePlanner(NetworkTopology(network))

    with pytest.raises(TypeError, match="topology must be NetworkTopology"):
        RoutePlanner(object())  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="unknown node id"):
        planner.fewest_hops_path("alice", "carol", port_kind=PortKind.QUANTUM)

    with pytest.raises(TypeError, match="port_kind must be PortKind"):
        planner.fewest_hops_path(
            "alice",
            "bob",
            port_kind="quantum",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="link_cost must be callable"):
        planner.lowest_cost_path(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            link_cost=1.0,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="link_cost for 'q_ab' must be non-negative"):
        network.add_quantum_link("q_ab", "alice", "bob")
        planner.lowest_cost_path(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            link_cost=lambda link: -1.0,
        )

    with pytest.raises(TypeError, match="max_hops must be int"):
        planner.paths_with_max_hops(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=1.0,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="max_hops must be non-negative"):
        planner.paths_with_max_hops(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=-1,
        )

    with pytest.raises(KeyError, match="unknown node id"):
        network.fewest_hops_path("alice", "carol", port_kind=PortKind.QUANTUM)

    with pytest.raises(TypeError, match="port_kind must be PortKind"):
        network.fewest_hops_path(
            "alice",
            "bob",
            port_kind="quantum",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="max_hops must be non-negative"):
        network.paths_with_max_hops(
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=-1,
        )


def test_route_rejects_non_contiguous_edges() -> None:
    left = TopologyEdge(
        link_id="q_ab",
        source_node_id="alice",
        target_node_id="bob",
        port_kind=PortKind.QUANTUM,
    )
    right = TopologyEdge(
        link_id="q_cd",
        source_node_id="carol",
        target_node_id="dave",
        port_kind=PortKind.QUANTUM,
    )

    with pytest.raises(ValueError, match="contiguous path"):
        Route(
            source_node_id="alice",
            target_node_id="dave",
            edges=(left, right),
        )


def test_route_rejects_empty_route_for_different_nodes() -> None:
    with pytest.raises(ValueError, match="empty route"):
        Route(source_node_id="alice", target_node_id="bob")
