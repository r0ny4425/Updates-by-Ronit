"""Classical message ingress and egress for control-plane agents.

``ClassicalEndpoint`` owns stable classical ports for an agent and sends
``ClassicalMessage`` instances through ordinary port connections. The endpoint
is an adapter, not a scheduled component; delivered messages target the owning
agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simyuj.components.connections import PortDelivery, require_connection
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.messages.transport import ClassicalMessage

from .payloads import AgentMessage

if TYPE_CHECKING:
    from .agent import Agent


class RoutingError(RuntimeError):
    """Raised when an endpoint cannot resolve an output port for a message."""

    pass


@dataclass(slots=True)
class ClassicalEndpoint:
    """Non-scheduled classical adapter owned by an agent component.

    Parameters
    ----------
    owner_agent : Agent
        Agent component that owns the generated ports.
    address : str or None, default=None
        Classical address exposed by the endpoint. ``None`` uses
        ``owner_agent.agent_id``.

    Notes
    -----
    Input and output ports are created lazily and are stable per local name.
    The endpoint validates incoming ``PortDelivery`` targets when an agent has
    enabled classical support.
    """

    owner_agent: Agent
    address: str | None = None

    _in_ports: dict[str, Port] = field(init=False, default_factory=dict)
    _out_ports: dict[str, Port] = field(init=False, default_factory=dict)
    _routes: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.owner_agent, Component):
            raise TypeError("owner_agent must be Component")

        agent_id = getattr(self.owner_agent, "agent_id", None)
        if not isinstance(agent_id, str) or agent_id == "":
            raise ValueError("owner_agent must expose non-empty agent_id")

        if self.address is None:
            self.address = agent_id
        else:
            ensure_nonempty_id(self.address, field_name="address")

    def in_port(self, name: str = "classical_in") -> Port:
        """Return a stable named classical ingress port owned by the agent.

        Parameters
        ----------
        name : str, default="classical_in"
            Local input port name.

        Returns
        -------
        Port
            Classical ingress port owned by ``owner_agent``.
        """
        ensure_nonempty_id(name, field_name="classical input port name")

        existing = self._in_ports.get(name)
        if existing is not None:
            return existing

        port = Port(
            name=name,
            owner=self.owner_agent,
            owner_id=self.owner_agent.agent_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )
        self._in_ports[name] = port
        return port

    def out_port(self, name: str = "classical_out") -> Port:
        """Return a stable named classical egress port owned by the agent.

        Parameters
        ----------
        name : str, default="classical_out"
            Local output port name.

        Returns
        -------
        Port
            Classical egress port owned by ``owner_agent``.
        """
        ensure_nonempty_id(name, field_name="classical output port name")

        existing = self._out_ports.get(name)
        if existing is not None:
            return existing

        port = Port(
            name=name,
            owner=self.owner_agent,
            owner_id=self.owner_agent.agent_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )
        self._out_ports[name] = port
        return port

    def owns_input(self, port: Port) -> bool:
        """Return whether ``port`` is one of this endpoint's input ports."""
        return any(port is known for known in self._in_ports.values())

    def extract(self, payload: object, time: int) -> AgentMessage:
        """Unwrap a strict ``AGENT_MESSAGE`` payload into an ``AgentMessage``.

        Parameters
        ----------
        payload : object
            ``ClassicalMessage`` or ``PortDelivery`` containing a
            ``ClassicalMessage``.
        time : int
            Timeline tick at which the message arrived.

        Returns
        -------
        AgentMessage
            Hydrated message delivered to ``Agent.on_message``.

        Raises
        ------
        ValueError
            If a ``PortDelivery`` targets a port not owned by this endpoint.
        TypeError
            If the delivered payload is not a ``ClassicalMessage``.
        """
        if isinstance(payload, PortDelivery):
            if not self.owns_input(payload.target_port):
                raise ValueError(
                    "PortDelivery target_port is not an agent classical input port"
                )
            if not isinstance(payload.payload, ClassicalMessage):
                raise TypeError(
                    "AGENT_MESSAGE PortDelivery payload must be ClassicalMessage"
                )
            return AgentMessage(
                message=payload.payload,
                receive_time=time,
                source_port_name=payload.source_port.name,
                target_port_name=payload.target_port.name,
                connection_id=payload.connection_id,
            )

        if not isinstance(payload, ClassicalMessage):
            raise TypeError(
                "AGENT_MESSAGE payload_ref must be ClassicalMessage or PortDelivery"
            )

        return AgentMessage(
            message=payload,
            receive_time=time,
        )

    def add_route(self, receiver_id: str, port_name: str) -> None:
        """Add a routing rule mapping a receiver address to a local output port.

        Parameters
        ----------
        receiver_id : str
            Destination address (typically the receiver's agent_id).
        port_name : str
            Local classical output port name to use for this destination.
        """
        ensure_nonempty_id(receiver_id, field_name="receiver_id")
        ensure_nonempty_id(port_name, field_name="port_name")
        self._routes[receiver_id] = port_name

    def send(
        self,
        message: ClassicalMessage,
        timeline: Timeline,
        *,
        port_name: str | None = None,
        time: int | None = None,
        priority: int = 0,
    ) -> Event:
        """Schedule a message through a connected classical output port.

        Parameters
        ----------
        message : ClassicalMessage
            Message to transmit through the selected output port.
        timeline : Timeline
            Timeline used to schedule the transmission event.
        port_name : str or None, default=None
            Explicit local output port name. If ``None``, the endpoint attempts
            to resolve the port using its internal routing table and
            ``message.receiver_id``. The port must already be connected.
        time : int or None, default=None
            Absolute timeline tick for delivery. ``None`` uses the current
            timeline time.
        priority : int, default=0
            Event priority passed to the connection.

        Returns
        -------
        Event
            Scheduled delivery event returned by the connection.

        Raises
        ------
        TypeError
            If ``message`` or ``timeline`` has the wrong type.
        RoutingError
            If ``port_name`` is ``None`` and no route matches ``message.receiver_id``.
        RuntimeError
            If the selected output port has no connection.

        Notes
        -----
        The endpoint does not require ``message.sender_id`` to match its
        ``address``; callers should enforce protocol-level identity rules before
        sending.

        Examples
        --------
        >>> from simyuj.components import connect_ports
        >>> from simyuj.control import AGENT_MESSAGE, Agent
        >>> from simyuj.primitives.messages import ClassicalMessage
        >>> from simyuj.engine import Timeline
        >>> alice = Agent(agent_id="alice")
        >>> bob = Agent(agent_id="bob")
        >>> _ = alice.enable_classical()
        >>> _ = bob.enable_classical()
        >>> _ = connect_ports(
        ...     alice.classical.out_port("to_bob"),
        ...     bob.classical.in_port("from_alice"),
        ...     target_action=AGENT_MESSAGE,
        ... )
        >>> message = ClassicalMessage(
        ...     sender_id="alice",
        ...     receiver_id="bob",
        ...     body="ping",
        ... )
        >>> alice.classical.add_route("bob", "to_bob")
        >>> event = alice.classical.send(
        ...     message,
        ...     Timeline(master_seed=1),
        ... )
        >>> event.action
        'agent_message'
        """
        if not isinstance(message, ClassicalMessage):
            raise TypeError("message must be ClassicalMessage")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")

        resolved_port_name = port_name
        if resolved_port_name is None:
            resolved_port_name = self._routes.get(message.receiver_id)
            if resolved_port_name is None:
                raise RoutingError(
                    f"No route configured for receiver '{message.receiver_id}'"
                )

        connection = require_connection(self.out_port(resolved_port_name))
        return connection.transmit(
            message,
            timeline,
            time=timeline.current_time if time is None else time,
            priority=priority,
            source=self.owner_agent,
            subsystem_id="control",
            meta={
                "agent_id": self.owner_agent.agent_id,
                "message_id": message.message_id,
                "message_type": message.message_type,
            },
        )


__all__ = ["ClassicalEndpoint"]
