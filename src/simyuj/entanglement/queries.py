"""Read-only query helpers for entangled-pair registries.

These helpers derive candidate views from ``EntangledPairRegistry`` and
``Route`` objects.  They do not mutate pair state, reserve pairs, consume
pairs, or submit runtime protocol operations.
"""

from __future__ import annotations

from dataclasses import dataclass

from simyuj.network.routing import Route
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import require_optional_probability
from simyuj.resources.memory import MemoryRef

from .pair import EntangledPairRecord, PairState
from .registry import EntangledPairRegistry


@dataclass(frozen=True, slots=True)
class RouteHopPairs:
    """Available entangled pairs for one route hop.

    This is query output only. It does not reserve or consume the pairs.

    Parameters
    ----------
    source_node_id, target_node_id : str
        Route-hop endpoints in route order.
    pairs : tuple[EntangledPairRecord, ...]
        Available candidate pairs connecting the hop.

    Examples
    --------
    >>> hop = RouteHopPairs("alice", "bob", ())
    >>> hop.pairs
    ()
    """

    source_node_id: str
    target_node_id: str
    pairs: tuple[EntangledPairRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_node_id",
            ensure_nonempty_id(self.source_node_id, field_name="source_node_id"),
        )
        object.__setattr__(
            self,
            "target_node_id",
            ensure_nonempty_id(self.target_node_id, field_name="target_node_id"),
        )

        if not isinstance(self.pairs, tuple):
            raise TypeError("pairs must be tuple[EntangledPairRecord, ...]")

        for pair in self.pairs:
            if not isinstance(pair, EntangledPairRecord):
                raise TypeError("pairs must contain only EntangledPairRecord entries")

    @property
    def has_pairs(self) -> bool:
        """Return whether at least one candidate pair is available."""

        return bool(self.pairs)


def available_pairs_touching_node(
    registry: EntangledPairRegistry,
    node_id: str,
    *,
    min_fidelity: float | None = None,
) -> tuple[EntangledPairRecord, ...]:
    """Return available pairs with one endpoint at ``node_id``.

    Pairs without fidelity estimates are excluded when ``min_fidelity`` is
    provided.

    Parameters
    ----------
    registry : EntangledPairRegistry
        Registry to query.
    node_id : str
        Node that must host one endpoint.
    min_fidelity : float or None, optional
        Optional minimum fidelity in ``[0, 1]``.

    Returns
    -------
    tuple[EntangledPairRecord, ...]
        Available matching pairs in registry order.
    """

    resolved_registry = _require_registry(registry)
    resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")
    resolved_min_fidelity = require_optional_probability(
        min_fidelity,
        field_name="min_fidelity",
    )

    return tuple(
        pair
        for pair in resolved_registry.available_pairs()
        if pair.has_node(resolved_node_id)
        and _passes_min_fidelity(pair, resolved_min_fidelity)
    )


def available_pairs_for_route_hops(
    registry: EntangledPairRegistry,
    route: Route,
    *,
    min_fidelity: float | None = None,
) -> tuple[RouteHopPairs, ...]:
    """Return available pair candidates for every hop in a route.

    A zero-hop route returns an empty tuple.

    Parameters
    ----------
    registry : EntangledPairRegistry
        Registry to query.
    route : Route
        Route whose adjacent node pairs become hop queries.
    min_fidelity : float or None, optional
        Optional minimum fidelity in ``[0, 1]``.

    Returns
    -------
    tuple[RouteHopPairs, ...]
        One query result per route hop, in route order.

    Notes
    -----
    Results are independent candidates per hop. This helper does not reserve
    pairs, choose a route-wide assignment, or guarantee disjoint selected pairs
    across hops.
    """

    resolved_registry = _require_registry(registry)
    resolved_route = _require_route(route)
    resolved_min_fidelity = require_optional_probability(
        min_fidelity,
        field_name="min_fidelity",
    )

    hop_pairs: list[RouteHopPairs] = []

    for edge in resolved_route.edges:
        pairs = resolved_registry.available_between(
            edge.source_node_id,
            edge.target_node_id,
            min_fidelity=resolved_min_fidelity,
            link_id=edge.link_id,
        )
        hop_pairs.append(
            RouteHopPairs(
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                pairs=pairs,
            )
        )

    return tuple(hop_pairs)


def route_hops_have_available_pairs(
    registry: EntangledPairRegistry,
    route: Route,
    *,
    min_fidelity: float | None = None,
) -> bool:
    """Return whether every route hop has at least one available pair.

    A zero-hop route returns True because there are no route hops to satisfy.
    """

    return all(
        hop.has_pairs
        for hop in available_pairs_for_route_hops(
            registry,
            route,
            min_fidelity=min_fidelity,
        )
    )


def pairs_using_any_memory(
    registry: EntangledPairRegistry,
    memory_refs: tuple[MemoryRef, ...],
) -> tuple[EntangledPairRecord, ...]:
    """Return historical pair records using any memory ref in ``memory_refs``.

    Duplicate pair records are collapsed by pair ID, and the result is sorted by
    pair ID for deterministic callers.
    """

    resolved_registry = _require_registry(registry)
    resolved_refs = _require_memory_refs(memory_refs)

    pairs_by_id: dict[str, EntangledPairRecord] = {}

    for memory_ref in resolved_refs:
        for pair in resolved_registry.pairs_using_memory(memory_ref):
            pairs_by_id.setdefault(pair.pair_id, pair)

    return tuple(pairs_by_id[pair_id] for pair_id in sorted(pairs_by_id))


def pairs_by_node_pair(
    registry: EntangledPairRegistry,
    *,
    state: PairState | None = None,
    min_fidelity: float | None = None,
) -> dict[tuple[str, str], tuple[EntangledPairRecord, ...]]:
    """Group pairs by unordered node-pair key.

    Parameters
    ----------
    registry : EntangledPairRegistry
        Registry to query.
    state : PairState or None, optional
        Optional lifecycle-state filter.
    min_fidelity : float or None, optional
        Optional minimum fidelity in ``[0, 1]``. Pairs without fidelity are
        excluded when this filter is supplied.

    Returns
    -------
    dict[tuple[str, str], tuple[EntangledPairRecord, ...]]
        Mapping keyed by sorted two-node tuples such as ``("alice", "bob")``.

    Notes
    -----
    The grouping starts from ``registry.all_pairs(state=state)``. Terminal
    records are included unless a state filter excludes them.
    """

    resolved_registry = _require_registry(registry)
    resolved_min_fidelity = require_optional_probability(
        min_fidelity,
        field_name="min_fidelity",
    )

    buckets: dict[tuple[str, str], list[EntangledPairRecord]] = {}

    for pair in resolved_registry.all_pairs(state=state):
        if not _passes_min_fidelity(pair, resolved_min_fidelity):
            continue

        key = ordered_node_pair(pair.left.node_id, pair.right.node_id)
        buckets.setdefault(key, []).append(pair)

    return {
        key: tuple(pairs)
        for key, pairs in sorted(buckets.items(), key=lambda item: item[0])
    }


def ordered_node_pair(
    first_node_id: str,
    second_node_id: str,
) -> tuple[str, str]:
    """Return a deterministic unordered node-pair key.

    Parameters
    ----------
    first_node_id, second_node_id : str
        Node IDs to canonicalize.

    Returns
    -------
    tuple[str, str]
        The two IDs sorted lexicographically.

    Examples
    --------
    >>> ordered_node_pair("bob", "alice")
    ('alice', 'bob')
    """

    first = ensure_nonempty_id(first_node_id, field_name="first_node_id")
    second = ensure_nonempty_id(second_node_id, field_name="second_node_id")

    return (first, second) if first <= second else (second, first)


def _passes_min_fidelity(
    pair: EntangledPairRecord,
    min_fidelity: float | None,
) -> bool:
    if min_fidelity is None:
        return True

    return pair.fidelity is not None and pair.fidelity >= min_fidelity


def _require_registry(registry: EntangledPairRegistry) -> EntangledPairRegistry:
    if not isinstance(registry, EntangledPairRegistry):
        raise TypeError("registry must be EntangledPairRegistry")

    return registry


def _require_route(route: Route) -> Route:
    if not isinstance(route, Route):
        raise TypeError("route must be Route")

    return route


def _require_memory_refs(
    memory_refs: tuple[MemoryRef, ...],
) -> tuple[MemoryRef, ...]:
    if not isinstance(memory_refs, tuple):
        raise TypeError("memory_refs must be tuple[MemoryRef, ...]")

    for memory_ref in memory_refs:
        if not isinstance(memory_ref, MemoryRef):
            raise TypeError("memory_refs must contain only MemoryRef entries")

    return memory_refs


__all__ = [
    "RouteHopPairs",
    "available_pairs_for_route_hops",
    "available_pairs_touching_node",
    "ordered_node_pair",
    "pairs_by_node_pair",
    "pairs_using_any_memory",
    "route_hops_have_available_pairs",
]
