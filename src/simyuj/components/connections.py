"""Runtime connections between component ports.

Connections convert port-level transmission into timeline events targeted at
components. They preserve the local source and target port identities in a
``PortDelivery`` payload so receiving components can validate which boundary
was crossed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id

from .ports import Port, PortDirection


@dataclass(frozen=True, slots=True)
class PortDelivery:
    """Immutable payload wrapper for one port-routed delivery.

    Parameters
    ----------
    payload : object
        Application payload being delivered, such as a ``Signal`` or
        ``ClassicalMessage``.
    source_port : Port
        Output port that transmitted the payload.
    target_port : Port
        Input port on the receiving component.
    connection_id : str
        Non-empty identifier of the connection that scheduled the delivery.

    Notes
    -----
    ``Event.meta`` is debug and tracing metadata. Event-facing components
    should use ``PortDelivery`` to determine what arrived and which local port
    received it.
    """

    payload: object
    source_port: Port
    target_port: Port
    connection_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_port, Port):
            raise TypeError("source_port must be Port")
        if not isinstance(self.target_port, Port):
            raise TypeError("target_port must be Port")
        ensure_nonempty_id(self.connection_id, field_name="connection_id")


@dataclass(frozen=True, slots=True)
class PortConnection:
    """One-way runtime connection from an output port to an input port.

    Parameters
    ----------
    connection_id : str
        Non-empty runtime connection identifier.
    source_port : Port
        Output port that owns transmissions for this connection.
    target_port : Port
        Input port whose owner receives scheduled delivery events.
    target_action : str
        Default event action delivered to ``target_port.owner``.

    Notes
    -----
    Construction validates direction, kind compatibility, distinct owners, and
    single-connection ownership on both ports. On success, the connection stores
    itself on both endpoint ports.

    The connection routes payloads to ``target_port.owner``. Ports are never
    event targets.

    Connections are one-to-one: one source port, one target port, and each port
    may have only one installed runtime connection.
    """

    connection_id: str
    source_port: Port
    target_port: Port
    target_action: str

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.connection_id, field_name="connection_id")
        ensure_nonempty_id(self.target_action, field_name="target_action")

        if not isinstance(self.source_port, Port):
            raise TypeError("source_port must be Port")
        if not isinstance(self.target_port, Port):
            raise TypeError("target_port must be Port")

        if self.source_port is self.target_port:
            raise ValueError("source_port and target_port must differ")

        if self.source_port.owner is self.target_port.owner:
            raise ValueError(
                "source and target ports must belong to different components"
            )

        if self.source_port.direction is not PortDirection.EGRESS:
            raise ValueError("source_port must be an EGRESS port")

        if self.target_port.direction is not PortDirection.INGRESS:
            raise ValueError("target_port must be an INGRESS port")

        if self.source_port.port_kind is not self.target_port.port_kind:
            raise ValueError("source_port and target_port must have the same port_kind")

        if self.source_port.connection is not None:
            raise ValueError(
                f"source port '{self.source_port.owner_id}.{self.source_port.name}' "
                "is already connected"
            )

        if self.target_port.connection is not None:
            raise ValueError(
                f"target port '{self.target_port.owner_id}.{self.target_port.name}' "
                "is already connected"
            )

        self.source_port.connection = self
        self.target_port.connection = self

    def transmit(
        self,
        payload: object,
        timeline: Timeline,
        *,
        action: str | None = None,
        time: int | None = None,
        priority: int = 0,
        source: object | None = None,
        subsystem_id: str = "components",
        meta: dict[str, Any] | None = None,
    ) -> Event:
        """Schedule a port delivery event on the timeline.

        Parameters
        ----------
        payload : object
            Payload to wrap in ``PortDelivery``.
        timeline : Timeline
            Timeline that will own and execute the scheduled event.
        action : str, optional
            Event action override. When omitted, ``target_action`` is used.
        time : int, optional
            Absolute simulation tick for delivery, not a delay. When omitted,
            the current timeline time is used.
        priority : int, default=0
            Event priority passed to the timeline.
        source : object, optional
            Event source override. When omitted, the source port owner is used.
        subsystem_id : str, default="components"
            Event subsystem identifier.
        meta : dict, optional
            Additional event metadata merged after connection metadata.

        Returns
        -------
        Event
            The event scheduled on the timeline.

        Raises
        ------
        TypeError
            If ``timeline``, ``priority``, ``time``, or ``meta`` has the wrong
            type.
        ValueError
            If ``time`` is earlier than the current timeline time or the
            resolved event action is empty.

        Notes
        -----
        The target is always the target port's owning component, never the port
        itself. The scheduled event payload is a fresh ``PortDelivery`` that
        preserves the original source and target port objects.

        Caller-supplied ``meta`` is merged after connection metadata, so it may
        override connection metadata keys. Components should use
        ``PortDelivery`` for correctness and treat ``Event.meta`` as tracing
        context.
        """

        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")

        if type(priority) is not int:
            raise TypeError("priority must be int")

        event_time = timeline.current_time if time is None else time
        if type(event_time) is not int:
            raise TypeError("time must be int or None")
        if event_time < timeline.current_time:
            raise ValueError("cannot transmit into the past")

        event_action = self.target_action if action is None else action
        ensure_nonempty_id(event_action, field_name="action")

        delivery = PortDelivery(
            payload=payload,
            source_port=self.source_port,
            target_port=self.target_port,
            connection_id=self.connection_id,
        )

        event_meta = {
            "connection_id": self.connection_id,
            "source_owner_id": self.source_port.owner_id,
            "source_port": self.source_port.name,
            "target_owner_id": self.target_port.owner_id,
            "target_port": self.target_port.name,
            "target_action": event_action,
        }
        if meta is not None:
            if not isinstance(meta, dict):
                raise TypeError("meta must be dict or None")
            event_meta.update(meta)

        target = self.target_port.owner
        if not isinstance(target, Component):
            raise TypeError("target_port.owner must be Component")

        return timeline.schedule(
            Event(
                time=event_time,
                priority=priority,
                target_ref=target,
                action=event_action,
                payload_ref=delivery,
                source=source if source is not None else self.source_port.owner,
                subsystem_id=subsystem_id,
                meta=event_meta,
            )
        )


def default_connection_id(source_port: Port, target_port: Port) -> str:
    """Return the default ``owner.port->owner.port`` connection identifier.

    Parameters
    ----------
    source_port, target_port : Port
        Endpoint ports used to derive the identifier.
    """
    if not isinstance(source_port, Port):
        raise TypeError("source_port must be Port")
    if not isinstance(target_port, Port):
        raise TypeError("target_port must be Port")
    return (
        f"{source_port.owner_id}.{source_port.name}"
        f"->{target_port.owner_id}.{target_port.name}"
    )


def connect_ports(
    source_port: Port,
    target_port: Port,
    *,
    target_action: str,
    connection_id: str | None = None,
) -> PortConnection:
    """Validate and connect one source port to one target port.

    Parameters
    ----------
    source_port : Port
        Egress endpoint.
    target_port : Port
        Ingress endpoint.
    target_action : str
        Default ``Event.action`` delivered to ``target_port.owner``.
    connection_id : str, optional
        Explicit connection identifier. When omitted, a default identifier is
        derived from the endpoint owner IDs and port names.

    Returns
    -------
    PortConnection
        Runtime connection installed on both endpoint ports.

    Notes
    -----
    Successful construction mutates both ports by setting their
    ``connection`` attribute to the returned ``PortConnection``.
    """

    if not isinstance(source_port, Port):
        raise TypeError("source_port must be Port")
    if not isinstance(target_port, Port):
        raise TypeError("target_port must be Port")

    resolved_connection_id = (
        default_connection_id(source_port, target_port)
        if connection_id is None
        else connection_id
    )

    return PortConnection(
        connection_id=resolved_connection_id,
        source_port=source_port,
        target_port=target_port,
        target_action=target_action,
    )


def require_connection(port: Port) -> PortConnection:
    """Return a port's connection or raise if the port is unconnected.

    Components call this at transmission time when a missing connection should
    be a runtime configuration error.
    """
    if not isinstance(port, Port):
        raise TypeError("port must be Port")
    if port.connection is None:
        raise RuntimeError(f"port '{port.owner_id}.{port.name}' is not connected")
    return port.connection


__all__ = [
    "PortConnection",
    "PortDelivery",
    "connect_ports",
    "default_connection_id",
    "require_connection",
]
