from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.metrics import (
    edge_metric,
    edge_success_probability,
    link_metric,
    link_success_probability,
)
from simyuj.network import TopologyEdge


def make_edge(link_id: str = "q_ab") -> TopologyEdge:
    return TopologyEdge(
        link_id=link_id,
        source_node_id="alice",
        target_node_id="bob",
        port_kind=PortKind.QUANTUM,
    )


def test_link_metric_returns_value_or_default() -> None:
    assert link_metric("q_ab", {"q_ab": 1.5}) == 1.5
    assert link_metric("q_missing", {"q_ab": 1.5}, default=2.0) == 2.0


def test_link_metric_validates_inputs() -> None:
    with pytest.raises(KeyError, match="missing delay"):
        link_metric("q_missing", {}, field_name="delay")

    with pytest.raises(ValueError, match="non-negative"):
        link_metric("q_ab", {"q_ab": -1.0})

    with pytest.raises(TypeError, match="mapping"):
        link_metric("q_ab", [("q_ab", 1.0)])  # type: ignore[arg-type]


def test_edge_metric_uses_edge_link_id() -> None:
    edge = make_edge("q_ab")

    assert edge_metric(edge, {"q_ab": 3.0}) == 3.0

    with pytest.raises(TypeError, match="TopologyEdge"):
        edge_metric("q_ab", {"q_ab": 3.0})  # type: ignore[arg-type]


def test_link_success_probability_returns_value_or_default() -> None:
    assert link_success_probability("q_ab", {"q_ab": 0.75}) == 0.75
    assert link_success_probability("q_missing", {}, default=0.25) == 0.25


def test_link_success_probability_validates_inputs() -> None:
    with pytest.raises(KeyError, match="missing success probability"):
        link_success_probability("q_missing", {})

    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        link_success_probability("q_ab", {"q_ab": 1.5})

    with pytest.raises(TypeError, match="mapping"):
        link_success_probability("q_ab", [("q_ab", 0.9)])  # type: ignore[arg-type]


def test_edge_success_probability_uses_edge_link_id() -> None:
    edge = make_edge("q_ab")

    assert edge_success_probability(edge, {"q_ab": 0.9}) == 0.9

    with pytest.raises(TypeError, match="TopologyEdge"):
        edge_success_probability("q_ab", {"q_ab": 0.9})  # type: ignore[arg-type]
