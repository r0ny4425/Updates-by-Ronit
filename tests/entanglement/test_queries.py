from __future__ import annotations

import pytest

from simyuj.components.ports import PortKind
from simyuj.entanglement import EntangledPairRecord, EntangledPairRegistry, PairState
from simyuj.entanglement.queries import (
    RouteHopPairs,
    available_pairs_for_route_hops,
    available_pairs_touching_node,
    ordered_node_pair,
    pairs_by_node_pair,
    pairs_using_any_memory,
    route_hops_have_available_pairs,
)
from simyuj.network import Route, TopologyEdge
from simyuj.resources import MemoryRef


def ref(node_id: str, position: int = 0) -> MemoryRef:
    return MemoryRef(node_id, "qmem", position)


def pair(
    pair_id: str,
    left: MemoryRef,
    right: MemoryRef,
    *,
    state: PairState = PairState.AVAILABLE,
    fidelity: float | None = None,
    generation_link_id: str | None = None,
) -> EntangledPairRecord:
    return EntangledPairRecord(
        pair_id=pair_id,
        left=left,
        right=right,
        state=state,
        fidelity=fidelity,
        generation_link_id=generation_link_id,
    )


def edge(
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


def route(*node_ids: str) -> Route:
    if len(node_ids) == 1:
        return Route(source_node_id=node_ids[0], target_node_id=node_ids[0])

    edges = tuple(
        edge(f"q_{left}_{right}", left, right)
        for left, right in zip(node_ids, node_ids[1:])
    )
    return Route(
        source_node_id=node_ids[0],
        target_node_id=node_ids[-1],
        edges=edges,
    )


def test_route_hop_pairs_validates_pairs() -> None:
    entangled_pair = pair("pair:0", ref("alice"), ref("bob"))
    hop_pairs = RouteHopPairs("alice", "bob", (entangled_pair,))

    assert hop_pairs.has_pairs

    with pytest.raises(TypeError, match="tuple"):
        RouteHopPairs("alice", "bob", [entangled_pair])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="EntangledPairRecord"):
        RouteHopPairs("alice", "bob", ("bad",))  # type: ignore[arg-type]


def test_available_pairs_touching_node_filters_available_and_fidelity() -> None:
    registry = EntangledPairRegistry()
    good = registry.register(pair("pair:0", ref("alice"), ref("bob"), fidelity=0.9))
    registry.register(pair("pair:1", ref("alice", 1), ref("carol"), fidelity=0.8))
    registry.register(
        pair(
            "pair:2",
            ref("alice", 2),
            ref("dave"),
            state=PairState.RESERVED,
            fidelity=1.0,
        )
    )

    assert available_pairs_touching_node(
        registry,
        "alice",
        min_fidelity=0.9,
    ) == (good,)


def test_available_pairs_for_route_hops_returns_pairs_per_hop() -> None:
    registry = EntangledPairRegistry()
    first = registry.register(
        pair(
            "pair:0",
            ref("alice"),
            ref("relay"),
            generation_link_id="q_alice_relay",
        )
    )
    second = registry.register(
        pair(
            "pair:1",
            ref("relay", 1),
            ref("bob"),
            generation_link_id="q_relay_bob",
        )
    )

    hop_pairs = available_pairs_for_route_hops(
        registry,
        route("alice", "relay", "bob"),
    )

    assert hop_pairs == (
        RouteHopPairs("alice", "relay", (first,)),
        RouteHopPairs("relay", "bob", (second,)),
    )
    assert registry.get("pair:0").state is PairState.AVAILABLE
    assert registry.get("pair:1").state is PairState.AVAILABLE


def test_available_pairs_for_route_hops_filters_by_route_edge_link_id() -> None:
    registry = EntangledPairRegistry()
    matching = registry.register(
        pair(
            "pair:link-a",
            ref("alice"),
            ref("bob"),
            generation_link_id="link-a",
        )
    )
    registry.register(
        pair(
            "pair:link-b",
            ref("alice", 1),
            ref("bob", 1),
            generation_link_id="link-b",
        )
    )
    route = Route(
        source_node_id="alice",
        target_node_id="bob",
        edges=(edge("link-a", "alice", "bob"),),
    )

    assert available_pairs_for_route_hops(registry, route) == (
        RouteHopPairs("alice", "bob", (matching,)),
    )


def test_route_hops_have_available_pairs_handles_missing_and_zero_hop_routes() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0", ref("alice"), ref("relay")))

    assert not route_hops_have_available_pairs(
        registry,
        route("alice", "relay", "bob"),
    )
    assert route_hops_have_available_pairs(registry, route("alice"))
    assert available_pairs_for_route_hops(registry, route("alice")) == ()


def test_pairs_using_any_memory_returns_pair_id_order_independent_of_input() -> None:
    registry = EntangledPairRegistry()
    alice = ref("alice")
    bob = ref("bob")
    carol = ref("carol")
    pair_b = registry.register(pair("pair:b", alice, bob, state=PairState.CONSUMED))
    pair_a = registry.register(pair("pair:a", bob, carol, state=PairState.CONSUMED))

    assert pairs_using_any_memory(registry, (carol, alice)) == (pair_a, pair_b)


def test_pairs_by_node_pair_groups_by_unordered_node_key() -> None:
    registry = EntangledPairRegistry()
    ab = registry.register(pair("pair:0", ref("alice"), ref("bob"), fidelity=0.9))
    ba = registry.register(
        pair("pair:1", ref("bob", 1), ref("alice", 1), fidelity=0.95)
    )
    registry.register(pair("pair:2", ref("alice", 2), ref("carol"), fidelity=0.5))

    assert pairs_by_node_pair(registry, min_fidelity=0.9) == {
        ("alice", "bob"): (ab, ba),
    }


def test_ordered_node_pair_returns_two_item_key() -> None:
    key = ordered_node_pair("bob", "alice")

    assert key == ("alice", "bob")
    assert len(key) == 2


def test_query_helpers_validate_inputs() -> None:
    registry = EntangledPairRegistry()

    with pytest.raises(TypeError, match="registry"):
        available_pairs_touching_node(object(), "alice")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="route must be Route"):
        available_pairs_for_route_hops(
            registry,
            object(),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="memory_refs must be tuple"):
        pairs_using_any_memory(registry, [ref("alice")])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="MemoryRef"):
        pairs_using_any_memory(registry, ("bad",))  # type: ignore[arg-type]
