"""Structural port endpoints for component-to-component routing.

Ports identify where payloads enter and leave components. They are not event
targets and they do not schedule timeline work; runtime delivery is handled by
``PortConnection`` in the connections module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.primitives.ids import ensure_nonempty_id

if TYPE_CHECKING:
    from .connections import PortConnection


class PortKind(Enum):
    """Logical payload plane accepted by a port."""

    QUANTUM = "quantum"
    CLASSICAL = "classical"


class PortDirection(Enum):
    """One-way routing role for a port endpoint.

    ``INGRESS`` marks an input endpoint and ``EGRESS`` marks an output
    endpoint.
    """

    INGRESS = "ingress"
    EGRESS = "egress"


@dataclass(slots=True, eq=False)
class Port:
    """Structural runtime endpoint owned by one component.

    Parameters
    ----------
    name : str
        Non-empty local port name, such as ``"in"`` or ``"out"``.
    owner : Component
        Component that owns this port and receives events routed to input
        ports.
    owner_id : str
        Non-empty stable identifier for the owning component.
    port_kind : PortKind
        Logical payload plane accepted by this port.
    direction : PortDirection
        Whether this port is an input or output endpoint.

    Attributes
    ----------
    connection : PortConnection or None
        Runtime connection attached to this port. It is set by
        ``PortConnection`` construction and starts as ``None``.

    Notes
    -----
    Ports are structural routing records. They do not implement
    ``handle_event()``, do not inspect payloads, and do not own timeline
    scheduling. ``PortConnection.transmit()`` targets the owner of the target
    port, never the port object itself.

    ``eq=False`` preserves object identity semantics. Existing connection code
    relies on the exact port object to check whether a delivery arrived at the
    expected local endpoint.

    ``connection`` is intentionally runtime-mutable. ``PortConnection``
    installs itself on both endpoint ports during wiring so components can
    cheaply check whether a port is connected before transmitting.
    """

    name: str
    owner: Component = field(repr=False, compare=False)
    owner_id: str
    port_kind: PortKind
    direction: PortDirection
    connection: PortConnection | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.name, field_name="name")
        ensure_nonempty_id(self.owner_id, field_name="owner_id")

        if not isinstance(self.owner, Component):
            raise TypeError("owner must be an engine Component")

        if not isinstance(self.port_kind, PortKind):
            raise TypeError("port_kind must be PortKind")

        if not isinstance(self.direction, PortDirection):
            raise TypeError("direction must be PortDirection")

    @property
    def is_connected(self) -> bool:
        """Whether this port currently has an attached runtime connection."""
        return self.connection is not None


__all__ = [
    "Port",
    "PortDirection",
    "PortKind",
]
