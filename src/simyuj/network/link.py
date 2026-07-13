"""Physical topology links between network nodes.

``NetworkLink`` records graph-level reachability. It may reference a transport
component such as a quantum or classical channel, but runtime event delivery is
owned by separate ``PortConnection`` wires. A channel stored on a link models
transport only when the runtime wires deliver payloads through that channel.
"""

from __future__ import annotations

from dataclasses import dataclass

from simyuj.components.ports import PortKind
from simyuj.primitives.ids import ensure_nonempty_id


@dataclass(frozen=True, slots=True)
class NetworkLink:
    """Directed physical topology link between two nodes.

    ``NetworkLink`` is graph metadata. It may point at link-owned transport
    state, such as a quantum or classical channel, but it does not imply that
    endpoint device ports are directly connected. Use ``Network.wire_ports`` to
    create runtime ``PortConnection`` delivery paths.

    ``transport`` is intentionally generic. The link records where link-owned
    state lives, but it does not validate channel behavior or put the transport
    in the event path.

    Keep this record about reachability, not scheduling. Ports, target actions,
    delivery events, and protocol behavior belong outside the link record.
    """

    link_id: str
    source_node_id: str
    target_node_id: str
    port_kind: PortKind
    transport: object | None = None

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.link_id, field_name="link_id")
        ensure_nonempty_id(self.source_node_id, field_name="source_node_id")
        ensure_nonempty_id(self.target_node_id, field_name="target_node_id")

        if not isinstance(self.port_kind, PortKind):
            raise TypeError("port_kind must be PortKind")


__all__ = ["NetworkLink"]
