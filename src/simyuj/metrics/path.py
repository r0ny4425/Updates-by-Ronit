"""Route-level metric helpers built from per-link quantities.

These helpers aggregate caller-provided metrics across ``Route`` objects.  They
remain protocol-neutral: no repeater, purification, swapping, or QKD-specific
assumptions are baked into the scoring rules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import prod

from simyuj.network.routing import Route
from simyuj.network.topology import TopologyEdge
from simyuj.primitives.validation import require_non_negative_real

from .link import edge_metric, edge_success_probability


def hop_count(route: Route) -> int:
    """Return the number of directed links in a route.

    Parameters
    ----------
    route : Route
        Route whose hop count is returned.

    Returns
    -------
    int
        Number of edges in the route. A single-node route has zero hops.
    """

    return _require_route(route).hops


def total_link_metric(
    route: Route,
    values: Mapping[str, float],
    *,
    field_name: str = "metric",
    default: float | None = None,
) -> float:
    """Sum a non-negative per-link metric over a route.

    This can represent generic additive quantities such as cost, delay, or
    length. It is intentionally not protocol-specific.

    Parameters
    ----------
    route : Route
        Route whose edges are scored.
    values : Mapping[str, float]
        Mapping from link ID to finite non-negative metric value.
    field_name : str, optional
        Name used in validation and missing-value errors.
    default : float or None, optional
        Non-negative fallback used for route links absent from ``values``.

    Returns
    -------
    float
        Sum of per-link metric values. A zero-hop route returns ``0.0``.

    Examples
    --------
    >>> from simyuj.components import PortKind
    >>> from simyuj.network.routing import Route
    >>> from simyuj.network.topology import TopologyEdge
    >>> route = Route(
    ...     "alice",
    ...     "charlie",
    ...     (
    ...         TopologyEdge("l_ab", "alice", "bob", PortKind.QUANTUM),
    ...         TopologyEdge("l_bc", "bob", "charlie", PortKind.QUANTUM),
    ...     ),
    ... )
    >>> total_link_metric(route, {"l_ab": 1.0, "l_bc": 2.5})
    3.5
    """

    resolved_route = _require_route(route)

    return sum(
        (
            edge_metric(
                edge,
                values,
                field_name=field_name,
                default=default,
            )
            for edge in resolved_route.edges
        ),
        0.0,
    )


def total_link_cost(
    route: Route,
    link_costs: Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Sum per-link costs over a route.

    This is a convenience wrapper around :func:`total_link_metric` using the
    validation field name ``"cost"``.
    """

    return total_link_metric(
        route,
        link_costs,
        field_name="cost",
        default=default,
    )


def total_link_delay(
    route: Route,
    link_delays: Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Sum per-link delays over a route.

    This is a convenience wrapper around :func:`total_link_metric` using the
    validation field name ``"delay"``.
    """

    return total_link_metric(
        route,
        link_delays,
        field_name="delay",
        default=default,
    )


def route_success_probability(
    route: Route,
    link_probabilities: Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Multiply independent per-link success probabilities over a route.

    This is still generic. It does not model a specific repeater, purification,
    swapping, or QKD protocol.

    Parameters
    ----------
    route : Route
        Route whose edge probabilities are multiplied.
    link_probabilities : Mapping[str, float]
        Mapping from link ID to probability in ``[0, 1]``.
    default : float or None, optional
        Probability fallback used for route links absent from the mapping.

    Returns
    -------
    float
        Product of per-link probabilities. A zero-hop route returns ``1.0``.

    Notes
    -----
    Multiplication assumes the caller wants independent per-link success
    composition. This helper does not model correlated failures, shared
    hardware, scheduling contention, purification, swapping, protocol retries,
    or other higher-level effects.
    """

    resolved_route = _require_route(route)

    return prod(
        (
            edge_success_probability(
                edge,
                link_probabilities,
                default=default,
            )
            for edge in resolved_route.edges
        ),
        start=1.0,
    )


def route_score(
    route: Route,
    edge_score: Callable[[TopologyEdge], float],
) -> float:
    """Sum a caller-provided non-negative edge score over a route.

    Parameters
    ----------
    route : Route
        Route whose edges are scored.
    edge_score : Callable[[TopologyEdge], float]
        Callable returning a finite non-negative score for each edge.

    Returns
    -------
    float
        Sum of edge scores.

    Raises
    ------
    TypeError
        If ``route`` is not a ``Route`` or ``edge_score`` is not callable.
    ValueError
        If any edge score is negative or non-finite.

    Notes
    -----
    Scores are costs or penalties: lower totals remain meaningful to callers
    such as ``best_route``. Negative reward-style scoring is intentionally
    rejected.
    """

    resolved_route = _require_route(route)

    if not callable(edge_score):
        raise TypeError("edge_score must be callable")

    total = 0.0

    for edge in resolved_route.edges:
        total += require_non_negative_real(
            edge_score(edge),
            field_name="edge_score",
        )

    return total


def best_route(
    routes: tuple[Route, ...],
    metric: Callable[[Route], float],
) -> Route | None:
    """Return the route with the minimum metric value.

    Ties preserve input order. Returns None for an empty route tuple.

    Parameters
    ----------
    routes : tuple[Route, ...]
        Candidate routes in deterministic tie-break order.
    metric : Callable[[Route], float]
        Callable returning a finite non-negative value for each route.

    Returns
    -------
    Route or None
        Route with the smallest metric, or ``None`` when no routes are given.

    Notes
    -----
    ``metric`` is minimized. Transform success or reward values before passing
    them here; for example, do not pass ``route_success_probability`` directly
    unless the lowest success probability is actually desired.

    Examples
    --------
    >>> from simyuj.components import PortKind
    >>> from simyuj.network.routing import Route
    >>> from simyuj.network.topology import TopologyEdge
    >>> via_bob = Route(
    ...     "alice",
    ...     "charlie",
    ...     (
    ...         TopologyEdge("l_ab", "alice", "bob", PortKind.QUANTUM),
    ...         TopologyEdge("l_bc", "bob", "charlie", PortKind.QUANTUM),
    ...     ),
    ... )
    >>> direct = Route(
    ...     "alice",
    ...     "charlie",
    ...     (TopologyEdge("l_ac", "alice", "charlie", PortKind.QUANTUM),),
    ... )
    >>> selected = best_route((via_bob, direct), hop_count)
    >>> selected.link_ids
    ('l_ac',)
    """

    if not isinstance(routes, tuple):
        raise TypeError("routes must be tuple[Route, ...]")

    if not callable(metric):
        raise TypeError("metric must be callable")

    best: Route | None = None
    best_value: float | None = None

    for route in routes:
        resolved_route = _require_route(route)
        value = require_non_negative_real(
            metric(resolved_route),
            field_name="route_metric",
        )

        if best is None or best_value is None or value < best_value:
            best = resolved_route
            best_value = value

    return best


def _require_route(route: Route) -> Route:
    if not isinstance(route, Route):
        raise TypeError("route must be Route")

    return route


__all__ = [
    "best_route",
    "hop_count",
    "route_score",
    "route_success_probability",
    "total_link_cost",
    "total_link_delay",
    "total_link_metric",
]
