"""Live node-level topology views over explicit ``Network`` links.

The topology layer projects physical ``NetworkLink`` records into directed
graph edges for routing and metrics. Runtime port wires are deliberately
ignored here, even when those wires deliver events during a simulation run.
"""

from __future__ import annotations

from dataclasses import dataclass

from simyuj.components.ports import PortKind
from simyuj.primitives.ids import ensure_nonempty_id

from .link import NetworkLink
from .network import Network


@dataclass(frozen=True, slots=True)
class TopologyEdge:
    """Directed node-level view of a physical ``NetworkLink``.

    The edge contains graph data only: link ID, source node, target node, and
    port kind. It does not include component ports, target actions, or runtime
    delivery state.
    """

    link_id: str
    source_node_id: str
    target_node_id: str
    port_kind: PortKind

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.link_id, field_name="link_id")
        ensure_nonempty_id(self.source_node_id, field_name="source_node_id")
        ensure_nonempty_id(self.target_node_id, field_name="target_node_id")

        if not isinstance(self.port_kind, PortKind):
            raise TypeError("port_kind must be PortKind")


class NetworkTopology:
    """Read-only live graph view over explicit ``Network`` topology links.

    ``NetworkTopology`` is for graph questions. Runtime delivery questions
    belong to ``PortConnection`` objects stored in ``Network.wires``.
    Queries rebuild deterministic views from the current network links, so
    later ``add_*_link`` calls are visible.
    """

    __slots__ = ("_network",)

    def __init__(self, network: Network) -> None:
        if not isinstance(network, Network):
            raise TypeError("network must be Network")
        self._network = network

    @property
    def network(self) -> Network:
        """Return the underlying live network."""

        return self._network

    def nodes(self) -> tuple[str, ...]:
        """Return known node IDs in deterministic order."""

        return tuple(sorted(self._network.nodes))

    @property
    def edges(self) -> tuple[TopologyEdge, ...]:
        """Return directed topology edges in deterministic link-ID order."""

        return self._edges()

    def neighbors(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[str, ...]:
        """Return unique outgoing neighbor node IDs in sorted order."""

        return tuple(
            sorted(
                {
                    edge.target_node_id
                    for edge in self.outgoing_edges(
                        node_id,
                        port_kind=port_kind,
                    )
                }
            )
        )

    def outgoing_edges(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[TopologyEdge, ...]:
        """Return explicit topology edges whose source is ``node_id``.

        Parallel links are preserved as separate edges.
        """

        resolved_node_id = self._require_node(node_id)
        resolved_port_kind = self._require_optional_port_kind(port_kind)

        return tuple(
            edge
            for edge in self._edges()
            if edge.source_node_id == resolved_node_id
            and self._matches_port_kind(edge, resolved_port_kind)
        )

    def incoming_edges(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[TopologyEdge, ...]:
        """Return explicit topology edges whose target is ``node_id``.

        Parallel links are preserved as separate edges.
        """

        resolved_node_id = self._require_node(node_id)
        resolved_port_kind = self._require_optional_port_kind(port_kind)

        return tuple(
            edge
            for edge in self._edges()
            if edge.target_node_id == resolved_node_id
            and self._matches_port_kind(edge, resolved_port_kind)
        )

    def has_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> bool:
        """Return whether an explicit directed topology edge exists."""

        resolved_source_node_id = self._require_node(source_node_id)
        resolved_target_node_id = self._require_node(target_node_id)
        resolved_port_kind = self._require_optional_port_kind(port_kind)

        return any(
            edge.source_node_id == resolved_source_node_id
            and edge.target_node_id == resolved_target_node_id
            and self._matches_port_kind(edge, resolved_port_kind)
            for edge in self._edges()
        )

    def _edges(self) -> tuple[TopologyEdge, ...]:
        return tuple(
            self._edge_from_link(self._network.links[link_id])
            for link_id in sorted(self._network.links)
        )

    def _require_node(self, node_id: str) -> str:
        resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")

        if resolved_node_id not in self._network.nodes:
            raise KeyError(f"unknown node id '{resolved_node_id}'")

        return resolved_node_id

    @staticmethod
    def _require_optional_port_kind(
        port_kind: PortKind | None,
    ) -> PortKind | None:
        if port_kind is not None and not isinstance(port_kind, PortKind):
            raise TypeError("port_kind must be PortKind or None")

        return port_kind

    @staticmethod
    def _matches_port_kind(
        edge: TopologyEdge,
        port_kind: PortKind | None,
    ) -> bool:
        return port_kind is None or edge.port_kind == port_kind

    @staticmethod
    def _edge_from_link(link: NetworkLink) -> TopologyEdge:
        if not isinstance(link, NetworkLink):
            raise TypeError("link must be NetworkLink")

        return TopologyEdge(
            link_id=link.link_id,
            source_node_id=link.source_node_id,
            target_node_id=link.target_node_id,
            port_kind=link.port_kind,
        )


__all__ = [
    "NetworkTopology",
    "TopologyEdge",
]
