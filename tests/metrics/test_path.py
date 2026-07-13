from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.metrics import (
    best_route,
    hop_count,
    route_score,
    route_success_probability,
    total_link_cost,
    total_link_delay,
    total_link_metric,
)
from simyuj.network import Route, TopologyEdge


def make_edge(
    link_id: str,
    source_node_id: str,
    target_node_id: str,
) -> TopologyEdge:
    return TopologyEdge(
        link_id=link_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        port_kind=PortKind.QUANTUM,
    )


def make_route(*link_ids: str) -> Route:
    if not link_ids:
        return Route(source_node_id="alice", target_node_id="alice")

    edges: list[TopologyEdge] = []
    source = "alice"

    for index, link_id in enumerate(link_ids):
        target = "bob" if index == len(link_ids) - 1 else f"relay_{index}"
        edges.append(make_edge(link_id, source, target))
        source = target

    return Route(
        source_node_id="alice",
        target_node_id="bob",
        edges=tuple(edges),
    )


def test_hop_count_returns_route_hops() -> None:
    assert hop_count(make_route("q_ab")) == 1
    assert hop_count(make_route("q_ar", "q_rb")) == 2


def test_total_link_metric_sums_values() -> None:
    route = make_route("q_ar", "q_rb")

    assert total_link_metric(route, {"q_ar": 1.0, "q_rb": 2.5}) == 3.5
    assert total_link_cost(route, {"q_ar": 1.0, "q_rb": 2.5}) == 3.5
    assert total_link_delay(route, {"q_ar": 4.0, "q_rb": 5.0}) == 9.0


def test_total_link_metric_returns_float_for_zero_hop_route() -> None:
    assert total_link_metric(make_route(), {}) == 0.0


def test_total_link_metric_uses_default_for_missing_links() -> None:
    route = make_route("q_ar", "q_rb")

    assert total_link_metric(route, {"q_ar": 1.0}, default=2.0) == 3.0


def test_route_success_probability_multiplies_link_values() -> None:
    route = make_route("q_ar", "q_rb")

    assert route_success_probability(route, {"q_ar": 0.5, "q_rb": 0.8}) == 0.4


def test_route_success_probability_returns_float_for_zero_hop_route() -> None:
    assert route_success_probability(make_route(), {}) == 1.0


def test_route_score_sums_custom_edge_scores() -> None:
    route = make_route("q_ar", "q_rb")

    assert route_score(route, lambda edge: len(edge.link_id)) == 8.0

    with pytest.raises(ValueError, match="non-negative"):
        route_score(route, lambda _edge: -1.0)


def test_best_route_returns_minimum_metric_preserving_ties() -> None:
    direct = make_route("q_ab")
    via_relay = make_route("q_ar", "q_rb")
    routes = (via_relay, direct)

    assert best_route(routes, hop_count) == direct
    assert best_route(routes, lambda _route: 1.0) == via_relay
    assert best_route((), hop_count) is None


def test_path_helpers_validate_route_inputs() -> None:
    with pytest.raises(TypeError, match="route must be Route"):
        hop_count("not-route")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="routes must be tuple"):
        best_route([make_route("q_ab")], hop_count)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="metric must be callable"):
        best_route((make_route("q_ab"),), 3.0)  # type: ignore[arg-type]
