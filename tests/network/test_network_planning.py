from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.network import Network, Node, Route
from simyuj.network.planning import (
    RankedRoute,
    best_candidate_route,
    best_candidate_route_by_link_cost,
    best_planned_route,
    best_route_by_link_cost,
    candidate_routes,
    rank_routes,
)
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from tests.network.test_network_routing import add_quantum_link


def make_network(*node_ids: str) -> Network:
    network = Network("planning")
    for node_id in node_ids:
        network.add_node(Node(node_id))
    return network


def make_planner() -> RoutePlanner:
    network = make_network("alice", "relay", "bob")
    add_quantum_link(network, "q_ab", "alice", "bob")
    add_quantum_link(network, "q_ar", "alice", "relay")
    add_quantum_link(network, "q_rb", "relay", "bob")
    return RoutePlanner(NetworkTopology(network))


def test_ranked_route_validates_score_and_route() -> None:
    route = Route(source_node_id="alice", target_node_id="alice")

    assert RankedRoute(route, 1).score == 1.0

    with pytest.raises(TypeError, match="route must be Route"):
        RankedRoute("not-route", 1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="score must be non-negative"):
        RankedRoute(route, -1.0)


def test_candidate_routes_delegates_to_planner() -> None:
    routes = candidate_routes(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    assert tuple(route.link_ids for route in routes) == (
        ("q_ab",),
        ("q_ar", "q_rb"),
    )


def test_rank_routes_sorts_by_score_and_preserves_ties() -> None:
    routes = candidate_routes(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    ranked = rank_routes(routes, lambda route: 1.0)

    assert tuple(item.route for item in ranked) == routes
    assert tuple(item.score for item in ranked) == (1.0, 1.0)


def test_best_planned_route_returns_lowest_score_or_none() -> None:
    routes = candidate_routes(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    best = best_planned_route(routes, lambda route: route.hops)
    assert best is not None
    assert best.link_ids == ("q_ab",)
    assert best_planned_route((), lambda route: route.hops) is None


def test_best_route_by_link_cost_uses_additive_link_costs() -> None:
    routes = candidate_routes(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
    )

    best = best_route_by_link_cost(
        routes,
        {"q_ab": 5.0, "q_ar": 1.0, "q_rb": 1.0},
    )
    assert best is not None
    assert best.link_ids == ("q_ar", "q_rb")


def test_best_candidate_route_generates_and_ranks_routes() -> None:
    route = best_candidate_route(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
        metric=lambda route: route.hops,
    )

    assert route is not None
    assert route.link_ids == ("q_ab",)


def test_best_candidate_route_by_link_cost_generates_and_ranks_routes() -> None:
    route = best_candidate_route_by_link_cost(
        make_planner(),
        "alice",
        "bob",
        port_kind=PortKind.QUANTUM,
        max_hops=2,
        link_costs={"q_ab": 5.0, "q_ar": 1.0, "q_rb": 1.0},
    )

    assert route is not None
    assert route.link_ids == ("q_ar", "q_rb")


def test_planning_helpers_validate_inputs() -> None:
    planner = make_planner()
    route = Route(source_node_id="alice", target_node_id="alice")

    with pytest.raises(TypeError, match="planner must be RoutePlanner"):
        candidate_routes(
            object(),  # type: ignore[arg-type]
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=1,
        )

    with pytest.raises(TypeError, match="routes must be tuple"):
        rank_routes([route], lambda route: 0.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Route entries"):
        rank_routes(("not-route",), lambda route: 0.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="metric must be callable"):
        rank_routes((route,), 1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="score must be non-negative"):
        best_candidate_route(
            planner,
            "alice",
            "bob",
            port_kind=PortKind.QUANTUM,
            max_hops=1,
            metric=lambda route: -1.0,
        )
