"""Protocol-neutral route records and graph search.

Routing operates on ``NetworkTopology`` edge metadata only. It does not reserve
resources, transmit signals, create entanglement, or inspect protocol state.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count

from simyuj.components.ports import PortKind
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import require_non_negative_real

from .link import NetworkLink
from .topology import NetworkTopology, TopologyEdge


@dataclass(frozen=True, slots=True)
class Route:
    """Directed path through a ``NetworkTopology``.

    A Route is graph metadata only. It does not reserve resources, allocate
    memories, create entanglement, or transmit anything.

    Parameters
    ----------
    source_node_id, target_node_id : str
        Route endpoints.
    edges : tuple[TopologyEdge, ...], optional
        Directed contiguous edges from source to target.

    Raises
    ------
    TypeError
        If ``edges`` is not a tuple of ``TopologyEdge`` instances.
    ValueError
        If identifiers are empty, an empty route has different endpoints, or
        the edges do not form a contiguous directed path.

    Examples
    --------
    >>> route = Route("alice", "alice")
    >>> route.hops
    0
    """

    source_node_id: str
    target_node_id: str
    edges: tuple[TopologyEdge, ...] = ()

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.source_node_id, field_name="source_node_id")
        ensure_nonempty_id(self.target_node_id, field_name="target_node_id")

        if not isinstance(self.edges, tuple):
            raise TypeError("edges must be tuple[TopologyEdge, ...]")

        for edge in self.edges:
            if not isinstance(edge, TopologyEdge):
                raise TypeError("edges must contain only TopologyEdge instances")

        if not self.edges:
            if self.source_node_id != self.target_node_id:
                raise ValueError(
                    "empty route is only valid when source_node_id "
                    "equals target_node_id"
                )
            return

        if self.edges[0].source_node_id != self.source_node_id:
            raise ValueError("first edge source does not match route source")

        if self.edges[-1].target_node_id != self.target_node_id:
            raise ValueError("last edge target does not match route target")

        for left, right in zip(self.edges, self.edges[1:]):
            if left.target_node_id != right.source_node_id:
                raise ValueError("route edges must form a contiguous path")

    @property
    def hops(self) -> int:
        """Return the number of edges in the route."""

        return len(self.edges)

    @property
    def link_ids(self) -> tuple[str, ...]:
        """Return route link IDs in traversal order."""

        return tuple(edge.link_id for edge in self.edges)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Return node IDs visited by the route in traversal order."""

        if not self.edges:
            return (self.source_node_id,)

        return (self.source_node_id,) + tuple(
            edge.target_node_id for edge in self.edges
        )

    @property
    def port_kinds(self) -> tuple[PortKind, ...]:
        """Return the port kind of each edge in traversal order."""

        return tuple(edge.port_kind for edge in self.edges)


class RoutePlanner:
    """Generic path search over ``NetworkTopology``.

    This class is intentionally protocol-neutral. It does not know about
    repeaters, BB84, purification, swapping, memory slots, or entangled pairs.
    The planner searches the supplied live topology view, so later link
    additions are visible through ``NetworkTopology``.

    Parameters
    ----------
    topology : NetworkTopology
        Topology view to search.
    """

    __slots__ = ("_topology",)

    def __init__(self, topology: NetworkTopology) -> None:
        if not isinstance(topology, NetworkTopology):
            raise TypeError("topology must be NetworkTopology")

        self._topology = topology

    @property
    def topology(self) -> NetworkTopology:
        """Return the topology searched by this planner."""

        return self._topology

    def fewest_hops_path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
    ) -> Route | None:
        """Return one deterministic path with the fewest hops.

        Returns None when no path exists.

        The search is directed and filtered by ``port_kind``. When multiple
        fewest-hop paths are possible, the path discovered first from
        deterministic link-ID edge order is returned.

        Parameters
        ----------
        source_node_id, target_node_id : str
            Endpoint node IDs.
        port_kind : PortKind
            Edge kind to traverse.

        Returns
        -------
        Route or None
            Fewest-hop route, zero-hop route for identical endpoints, or
            ``None`` if the target is unreachable.
        """

        source = self._require_node(source_node_id, field_name="source_node_id")
        target = self._require_node(target_node_id, field_name="target_node_id")
        resolved_port_kind = self._require_port_kind(port_kind)

        if source == target:
            return Route(
                source_node_id=source,
                target_node_id=target,
                edges=(),
            )

        edges_by_source = self._outgoing_edges_by_source(
            port_kind=resolved_port_kind,
        )

        visited: set[str] = {source}
        queue: deque[tuple[str, tuple[TopologyEdge, ...]]] = deque()
        queue.append((source, ()))

        while queue:
            current_node_id, path_edges = queue.popleft()

            for edge in edges_by_source.get(current_node_id, ()):
                next_node_id = edge.target_node_id

                if next_node_id in visited:
                    continue

                next_path_edges = path_edges + (edge,)

                if next_node_id == target:
                    return Route(
                        source_node_id=source,
                        target_node_id=target,
                        edges=next_path_edges,
                    )

                visited.add(next_node_id)
                queue.append((next_node_id, next_path_edges))

        return None

    def lowest_cost_path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
        link_cost: Callable[[NetworkLink], float],
    ) -> Route | None:
        """Return one deterministic route with the lowest additive link cost.

        Costs are supplied by the caller so routing remains protocol-neutral.
        The callable receives the underlying ``NetworkLink`` for each traversed
        edge and must return a finite non-negative number. Dijkstra's algorithm
        is used, so negative costs are rejected.
        """

        source = self._require_node(source_node_id, field_name="source_node_id")
        target = self._require_node(target_node_id, field_name="target_node_id")
        resolved_port_kind = self._require_port_kind(port_kind)

        if not callable(link_cost):
            raise TypeError("link_cost must be callable")

        if source == target:
            return Route(
                source_node_id=source,
                target_node_id=target,
                edges=(),
            )

        edges_by_source = self._outgoing_edges_by_source(
            port_kind=resolved_port_kind,
        )

        best_cost_by_node: dict[str, float] = {source: 0.0}
        sequence = count()
        pending: list[tuple[float, int, int, str, tuple[TopologyEdge, ...]]] = [
            (0.0, 0, next(sequence), source, ())
        ]

        while pending:
            current_cost, _, _, current_node_id, path_edges = heapq.heappop(pending)

            if current_cost > best_cost_by_node[current_node_id]:
                continue

            if current_node_id == target:
                return Route(
                    source_node_id=source,
                    target_node_id=target,
                    edges=path_edges,
                )

            for edge in edges_by_source.get(current_node_id, ()):
                link = self._topology.network.get_link(edge.link_id)
                edge_cost = require_non_negative_real(
                    link_cost(link),
                    field_name=f"link_cost for '{edge.link_id}'",
                )
                next_node_id = edge.target_node_id
                next_cost = current_cost + edge_cost

                if next_cost >= best_cost_by_node.get(next_node_id, float("inf")):
                    continue

                best_cost_by_node[next_node_id] = next_cost
                next_path_edges = path_edges + (edge,)
                heapq.heappush(
                    pending,
                    (
                        next_cost,
                        len(next_path_edges),
                        next(sequence),
                        next_node_id,
                        next_path_edges,
                    ),
                )

        return None

    def paths_with_max_hops(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
        max_hops: int,
    ) -> tuple[Route, ...]:
        """Return deterministic simple paths with at most ``max_hops`` edges.

        Simple path means no repeated node. This prevents cycles from producing
        unbounded route sets.

        Routes are returned in deterministic depth-first edge order, not sorted
        by hop count.
        Parallel topology edges are preserved, so two different link IDs
        between the same nodes can produce two different candidate routes.

        Parameters
        ----------
        source_node_id, target_node_id : str
            Endpoint node IDs.
        port_kind : PortKind
            Edge kind to traverse.
        max_hops : int
            Non-negative maximum number of edges in each returned route.

        Returns
        -------
        tuple[Route, ...]
            Simple routes in deterministic depth-first order.
        """

        source = self._require_node(source_node_id, field_name="source_node_id")
        target = self._require_node(target_node_id, field_name="target_node_id")
        resolved_port_kind = self._require_port_kind(port_kind)
        resolved_max_hops = self._require_max_hops(max_hops)

        edges_by_source = self._outgoing_edges_by_source(
            port_kind=resolved_port_kind,
        )

        routes: list[Route] = []

        def visit(
            current_node_id: str,
            path_edges: tuple[TopologyEdge, ...],
            seen_node_ids: set[str],
        ) -> None:
            if len(path_edges) > resolved_max_hops:
                return

            if current_node_id == target:
                routes.append(
                    Route(
                        source_node_id=source,
                        target_node_id=target,
                        edges=path_edges,
                    )
                )
                return

            if len(path_edges) == resolved_max_hops:
                return

            for edge in edges_by_source.get(current_node_id, ()):
                next_node_id = edge.target_node_id

                if next_node_id in seen_node_ids:
                    continue

                visit(
                    next_node_id,
                    path_edges + (edge,),
                    seen_node_ids | {next_node_id},
                )

        visit(source, (), {source})

        return tuple(routes)

    def _outgoing_edges_by_source(
        self,
        *,
        port_kind: PortKind,
    ) -> dict[str, tuple[TopologyEdge, ...]]:
        edges_by_source: dict[str, list[TopologyEdge]] = {}

        for edge in self._topology.edges:
            if edge.port_kind != port_kind:
                continue

            edges_by_source.setdefault(edge.source_node_id, []).append(edge)

        return {
            source_node_id: tuple(edges)
            for source_node_id, edges in edges_by_source.items()
        }

    def _require_node(self, node_id: str, *, field_name: str) -> str:
        resolved_node_id = ensure_nonempty_id(node_id, field_name=field_name)

        if resolved_node_id not in self._topology.network.nodes:
            raise KeyError(f"unknown node id '{resolved_node_id}'")

        return resolved_node_id

    @staticmethod
    def _require_port_kind(port_kind: PortKind) -> PortKind:
        if not isinstance(port_kind, PortKind):
            raise TypeError("port_kind must be PortKind")

        return port_kind

    @staticmethod
    def _require_max_hops(max_hops: int) -> int:
        if type(max_hops) is not int:
            raise TypeError("max_hops must be int")

        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")

        return max_hops


__all__ = [
    "Route",
    "RoutePlanner",
]
