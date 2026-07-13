"""Route candidate generation and ranking helpers.

Planning helpers sit above ``RoutePlanner`` and below protocol code. They rank
generic routes by caller-provided non-negative metrics and do not reserve
resources or mutate the network.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simyuj.components.ports import PortKind
from simyuj.metrics.path import total_link_cost
from simyuj.network.routing import Route, RoutePlanner
from simyuj.primitives.validation import require_non_negative_real


@dataclass(frozen=True, slots=True)
class RankedRoute:
    """Route plus generic non-negative score.

    Lower score is considered better by the helpers in this module.

    Parameters
    ----------
    route : Route
        Candidate route.
    score : float
        Finite non-negative score. Lower is better.

    Raises
    ------
    TypeError
        If ``route`` is not a ``Route`` or ``score`` has an unsupported type.
    ValueError
        If ``score`` is negative or non-finite.
    """

    route: Route
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.route, Route):
            raise TypeError("route must be Route")

        object.__setattr__(
            self,
            "score",
            require_non_negative_real(self.score, field_name="score"),
        )


def candidate_routes(
    planner: RoutePlanner,
    source_node_id: str,
    target_node_id: str,
    *,
    port_kind: PortKind,
    max_hops: int,
) -> tuple[Route, ...]:
    """Return deterministic simple candidate routes up to ``max_hops``.

    This is a thin validation wrapper around
    ``RoutePlanner.paths_with_max_hops``.

    Parameters
    ----------
    planner : RoutePlanner
        Planner used to generate candidates.
    source_node_id, target_node_id : str
        Endpoint node IDs.
    port_kind : PortKind
        Edge kind to traverse.
    max_hops : int
        Non-negative maximum hop count.

    Returns
    -------
    tuple[Route, ...]
        Candidate routes in planner order.
    """

    if not isinstance(planner, RoutePlanner):
        raise TypeError("planner must be RoutePlanner")

    return planner.paths_with_max_hops(
        source_node_id,
        target_node_id,
        port_kind=port_kind,
        max_hops=max_hops,
    )


def rank_routes(
    routes: tuple[Route, ...],
    metric: Callable[[Route], float],
) -> tuple[RankedRoute, ...]:
    """Rank routes by a caller-provided non-negative metric.

    Ties preserve input order.
    Metrics should be read-only. Use them to inspect current state and score a
    route; perform reservations or other mutations after a route is selected.

    Parameters
    ----------
    routes : tuple[Route, ...]
        Candidate routes.
    metric : Callable[[Route], float]
        Callable returning a finite non-negative score.

    Returns
    -------
    tuple[RankedRoute, ...]
        Ranked records sorted by score.

    Examples
    --------
    >>> rank_routes((Route("alice", "alice"),), lambda route: route.hops)
    (...,)
    """

    resolved_routes = _require_routes(routes)

    if not callable(metric):
        raise TypeError("metric must be callable")

    ranked = tuple(
        RankedRoute(
            route=route,
            score=metric(route),
        )
        for route in resolved_routes
    )

    return tuple(sorted(ranked, key=lambda ranked_route: ranked_route.score))


def best_planned_route(
    routes: tuple[Route, ...],
    metric: Callable[[Route], float],
) -> Route | None:
    """Return the lowest-score route, or ``None`` when ``routes`` is empty."""

    ranked = rank_routes(routes, metric)

    if not ranked:
        return None

    return ranked[0].route


def best_route_by_link_cost(
    routes: tuple[Route, ...],
    link_costs: Mapping[str, float],
    *,
    default: float | None = None,
) -> Route | None:
    """Return the lowest-score route by additive link cost.

    Parameters
    ----------
    routes : tuple[Route, ...]
        Candidate routes.
    link_costs : Mapping[str, float]
        Mapping from link ID to non-negative cost.
    default : float or None, optional
        Optional non-negative fallback for missing link IDs.
        If ``default`` is ``None``, every traversed link ID must be present in
        ``link_costs``.
    """

    return best_planned_route(
        routes,
        lambda route: total_link_cost(
            route,
            link_costs,
            default=default,
        ),
    )


def best_candidate_route(
    planner: RoutePlanner,
    source_node_id: str,
    target_node_id: str,
    *,
    port_kind: PortKind,
    max_hops: int,
    metric: Callable[[Route], float],
) -> Route | None:
    """Generate candidates and return the lowest-score route.

    This combines ``candidate_routes`` and ``best_planned_route`` without
    changing the planner's deterministic candidate order.
    """

    routes = candidate_routes(
        planner,
        source_node_id,
        target_node_id,
        port_kind=port_kind,
        max_hops=max_hops,
    )

    return best_planned_route(routes, metric)


def best_candidate_route_by_link_cost(
    planner: RoutePlanner,
    source_node_id: str,
    target_node_id: str,
    *,
    port_kind: PortKind,
    max_hops: int,
    link_costs: Mapping[str, float],
    default: float | None = None,
) -> Route | None:
    """Generate candidates and return the lowest-score route by link cost.

    Link costs are summed with ``simyuj.metrics.path.total_link_cost``.
    """

    return best_candidate_route(
        planner,
        source_node_id,
        target_node_id,
        port_kind=port_kind,
        max_hops=max_hops,
        metric=lambda route: total_link_cost(
            route,
            link_costs,
            default=default,
        ),
    )


def _require_routes(routes: tuple[Route, ...]) -> tuple[Route, ...]:
    if not isinstance(routes, tuple):
        raise TypeError("routes must be tuple[Route, ...]")

    for route in routes:
        if not isinstance(route, Route):
            raise TypeError("routes must contain only Route entries")

    return routes


__all__ = [
    "RankedRoute",
    "best_candidate_route",
    "best_candidate_route_by_link_cost",
    "best_planned_route",
    "best_route_by_link_cost",
    "candidate_routes",
    "rank_routes",
]
