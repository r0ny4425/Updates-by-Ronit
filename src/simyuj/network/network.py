"""Network registry for node topology and runtime port wires.

``Network`` keeps physical topology links separate from runtime
``PortConnection`` wiring. Topology queries and routing inspect explicit
``NetworkLink`` records only. Port wires are event-delivery plumbing and never
create graph reachability.

Use links to describe the node graph. Use wires to schedule delivery between
component-owned ports. A channel stored on a link affects a payload only when
the runtime wires place that channel in the event path.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from simyuj.components.connections import PortConnection
from simyuj.components.connections import connect_ports as create_port_connection
from simyuj.components.ports import Port, PortKind
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.runtime.binding import BindingContext, bind_if_supported

from .link import NetworkLink
from .node import Node

if TYPE_CHECKING:
    from .routing import Route, RoutePlanner
    from .topology import NetworkTopology, TopologyEdge


class Network:
    """Registry for nodes, physical links, and runtime port wires.

    Nodes store endpoint devices, agents, and optional local port aliases.
    Physical topology is declared with :meth:`add_link`,
    :meth:`add_quantum_link`, or :meth:`add_classical_link`. Runtime event
    delivery is declared separately with :meth:`wire_ports`.

    Topology/routing methods never inspect runtime wires. A quantum wire from
    one component to another is not a network edge unless an explicit quantum
    link also exists between the corresponding nodes.

    The registry does not execute events. ``wire_ports`` returns a
    ``PortConnection``; transmission happens later when that connection, or a
    component using it, schedules an event on a ``Timeline``.
    """

    __slots__ = ("_topology_id", "_nodes", "_links", "_wires")

    def __init__(self, topology_id: str = "topology") -> None:
        self._topology_id = ensure_nonempty_id(
            topology_id,
            field_name="topology_id",
        )
        self._nodes: dict[str, Node] = {}
        self._links: dict[str, NetworkLink] = {}
        self._wires: dict[str, PortConnection] = {}

    @property
    def topology_id(self) -> str:
        """Return this network's stable topology identifier."""

        return self._topology_id

    @property
    def nodes(self) -> Mapping[str, Node]:
        """Read-only live mapping of node ID to ``Node``."""

        return MappingProxyType(self._nodes)

    @property
    def links(self) -> Mapping[str, NetworkLink]:
        """Read-only live mapping of physical link ID to ``NetworkLink``."""

        return MappingProxyType(self._links)

    @property
    def wires(self) -> Mapping[str, PortConnection]:
        """Read-only live mapping of runtime wire ID to ``PortConnection``."""

        return MappingProxyType(self._wires)

    @property
    def quantum_links(self) -> Mapping[str, NetworkLink]:
        """Read-only mapping containing only quantum topology links."""

        return MappingProxyType(
            {
                link_id: link
                for link_id, link in self._links.items()
                if link.port_kind is PortKind.QUANTUM
            }
        )

    @property
    def classical_links(self) -> Mapping[str, NetworkLink]:
        """Read-only mapping containing only classical topology links."""

        return MappingProxyType(
            {
                link_id: link
                for link_id, link in self._links.items()
                if link.port_kind is PortKind.CLASSICAL
            }
        )

    @property
    def edges(self) -> tuple[TopologyEdge, ...]:
        """Return directed topology edges derived from explicit links."""

        return self._topology_view().edges

    def neighbors(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[str, ...]:
        """Return outgoing neighbor node IDs from explicit topology links."""

        return self._topology_view().neighbors(node_id, port_kind=port_kind)

    def outgoing_edges(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[TopologyEdge, ...]:
        """Return topology edges leaving ``node_id``."""

        return self._topology_view().outgoing_edges(
            node_id,
            port_kind=port_kind,
        )

    def incoming_edges(
        self,
        node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> tuple[TopologyEdge, ...]:
        """Return topology edges entering ``node_id``."""

        return self._topology_view().incoming_edges(
            node_id,
            port_kind=port_kind,
        )

    def has_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind | None = None,
    ) -> bool:
        """Return whether a directed topology edge exists."""

        return self._topology_view().has_edge(
            source_node_id,
            target_node_id,
            port_kind=port_kind,
        )

    def fewest_hops_path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
    ) -> Route | None:
        """Return one deterministic fewest-hop route, or ``None``."""

        return self._route_planner().fewest_hops_path(
            source_node_id,
            target_node_id,
            port_kind=port_kind,
        )

    def lowest_cost_path(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
        link_cost: Callable[[NetworkLink], float],
    ) -> Route | None:
        """Return one deterministic lowest-cost route, or ``None``."""

        return self._route_planner().lowest_cost_path(
            source_node_id,
            target_node_id,
            port_kind=port_kind,
            link_cost=link_cost,
        )

    def paths_with_max_hops(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
        max_hops: int,
    ) -> tuple[Route, ...]:
        """Return deterministic simple routes with at most ``max_hops`` edges."""

        return self._route_planner().paths_with_max_hops(
            source_node_id,
            target_node_id,
            port_kind=port_kind,
            max_hops=max_hops,
        )

    def add_node(self, node: Node) -> Node:
        """Register a node namespace with this network."""

        if not isinstance(node, Node):
            raise TypeError("node must be network.Node")

        if node.node_id in self._nodes:
            raise ValueError(f"node id '{node.node_id}' already exists")

        self._nodes[node.node_id] = node
        return node

    def get_node(self, node_id: str) -> Node:
        """Return a registered node by ID."""

        resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")
        return self._nodes[resolved_node_id]

    def add_link(
        self,
        link_id: str,
        source_node_id: str,
        target_node_id: str,
        *,
        port_kind: PortKind,
        transport: object | None = None,
    ) -> NetworkLink:
        """Register a directed physical topology link.

        The optional ``transport`` object is link-owned runtime state, such as
        a quantum or classical channel. It is bound by :meth:`bind_all`, but it
        does not become a device on either endpoint node.

        Registering a link changes topology and route search. It does not
        connect component ports or schedule transmissions. Use
        :meth:`wire_ports` for the runtime delivery path.
        """

        resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")
        resolved_source_node_id = ensure_nonempty_id(
            source_node_id,
            field_name="source_node_id",
        )
        resolved_target_node_id = ensure_nonempty_id(
            target_node_id,
            field_name="target_node_id",
        )

        if not isinstance(port_kind, PortKind):
            raise TypeError("port_kind must be PortKind")

        if resolved_link_id in self._links:
            raise ValueError(f"link id '{resolved_link_id}' already exists")

        self.get_node(resolved_source_node_id)
        self.get_node(resolved_target_node_id)

        link = NetworkLink(
            link_id=resolved_link_id,
            source_node_id=resolved_source_node_id,
            target_node_id=resolved_target_node_id,
            port_kind=port_kind,
            transport=transport,
        )
        self._links[resolved_link_id] = link
        return link

    def add_quantum_link(
        self,
        link_id: str,
        source_node_id: str,
        target_node_id: str,
        *,
        channel: object | None = None,
    ) -> NetworkLink:
        """Register a directed quantum topology link."""

        return self.add_link(
            link_id,
            source_node_id,
            target_node_id,
            port_kind=PortKind.QUANTUM,
            transport=channel,
        )

    def add_classical_link(
        self,
        link_id: str,
        source_node_id: str,
        target_node_id: str,
        *,
        channel: object | None = None,
    ) -> NetworkLink:
        """Register a directed classical topology link."""

        return self.add_link(
            link_id,
            source_node_id,
            target_node_id,
            port_kind=PortKind.CLASSICAL,
            transport=channel,
        )

    def wire_ports(
        self,
        wire_id: str,
        source_port: Port,
        target_port: Port,
        *,
        target_action: str,
    ) -> PortConnection:
        """Connect component-owned ports for runtime event delivery only.

        The returned ``PortConnection`` is one-way from ``source_port`` to
        ``target_port``. Calling ``transmit`` on that connection schedules a
        timeline event targeting ``target_port.owner`` with ``target_action``
        as the default event action and a ``PortDelivery`` payload.

        The ports do not need to be registered as node aliases. Aliases are
        only human-facing names; this method uses the actual ``Port`` objects.

        This method does not create a ``NetworkLink`` and does not affect
        ``edges``, ``neighbors``, or route search. It also does not transmit
        anything immediately.
        """

        resolved_wire_id = ensure_nonempty_id(wire_id, field_name="wire_id")

        if resolved_wire_id in self._wires:
            raise ValueError(f"wire id '{resolved_wire_id}' already exists")

        if not isinstance(source_port, Port):
            raise TypeError("source_port must be components.Port")

        if not isinstance(target_port, Port):
            raise TypeError("target_port must be components.Port")

        connection = create_port_connection(
            source_port,
            target_port,
            connection_id=resolved_wire_id,
            target_action=target_action,
        )
        self._wires[resolved_wire_id] = connection
        return connection

    def get_link(self, link_id: str) -> NetworkLink:
        """Return physical topology link metadata by link ID."""

        resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")
        return self._links[resolved_link_id]

    def get_wire(self, wire_id: str) -> PortConnection:
        """Return runtime ``PortConnection`` by wire ID."""

        resolved_wire_id = ensure_nonempty_id(wire_id, field_name="wire_id")
        return self._wires[resolved_wire_id]

    def bind_all(self, context: BindingContext) -> tuple[object, ...]:
        """Bind all bind-capable node devices and link transports once.

        Agents are not bound here. Their lifecycle belongs to the runtime that
        starts and coordinates control-plane work.
        """

        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        bound_entities: list[object] = []
        seen: set[int] = set()

        for node_id in sorted(self._nodes):
            node = self._nodes[node_id]
            for device_id in sorted(node.devices):
                device = node.devices[device_id]
                if self._bind_once(
                    device,
                    context,
                    seen,
                    component_id=device_id,
                    meta=(
                        ("node_id", node_id),
                        ("device_id", device_id),
                    ),
                ):
                    bound_entities.append(device)

        for link_id in sorted(self._links):
            transport = self._links[link_id].transport
            if transport is None:
                continue
            if self._bind_once(
                transport,
                context,
                seen,
                component_id=link_id,
                meta=(("link_id", link_id),),
            ):
                bound_entities.append(transport)

        return tuple(bound_entities)

    @staticmethod
    def _bind_once(
        entity: object,
        context: BindingContext,
        seen: set[int],
        *,
        component_id: str,
        meta: tuple[tuple[str, object], ...],
    ) -> bool:
        marker = id(entity)
        if marker in seen:
            return False
        seen.add(marker)

        return bind_if_supported(
            entity,
            context.timeline,
            context.logger,
            entity_id=context.entity_id,
            component_id=component_id,
            meta=context.meta + meta,
        )

    def _topology_view(self) -> NetworkTopology:
        from .topology import NetworkTopology

        return NetworkTopology(self)

    def _route_planner(self) -> RoutePlanner:
        from .routing import RoutePlanner

        return RoutePlanner(self._topology_view())


__all__ = ["Network"]
