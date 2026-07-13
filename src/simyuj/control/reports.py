"""Report ingress adapter for control-plane agents.

The endpoint owns stable classical ingress ports on behalf of an agent and
normalizes supported ``AGENT_REPORT`` payload forms before dispatch. It is not a
``Component`` and is never scheduled directly on the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.components.connections import PortDelivery
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.primitives.ids import ensure_nonempty_id

from .payloads import AgentReport


@dataclass(slots=True)
class AgentReportEndpoint:
    """Non-scheduled report adapter owned by an agent component.

    ``AGENT_REPORT`` payloads may be:

    - ``PortDelivery(report)`` from a component report output port,
    - ``AgentReport(report)`` for explicit control-plane wrapping, or
    - a raw report object for direct-scheduled convenience events.

    The endpoint is not a ``Component`` and must not be used as an event target.

    Parameters
    ----------
    owner_agent : Component
        Agent component that owns the endpoint ports.
    default_port_name : str, default="report"
        Port name used by ``input_port``.
    """

    owner_agent: Component
    default_port_name: str = "report"

    _ports: dict[str, Port] = field(init=False, default_factory=dict)
    _owner_agent_id: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.owner_agent, Component):
            raise TypeError("owner_agent must be Component")

        agent_id = getattr(self.owner_agent, "agent_id", None)
        if not isinstance(agent_id, str) or agent_id == "":
            raise ValueError("owner_agent must expose non-empty agent_id")
        self._owner_agent_id = agent_id

        ensure_nonempty_id(self.default_port_name, field_name="default_port_name")

    def port(self, name: str | None = None) -> Port:
        """Return a stable named report ingress port owned by the agent.

        Parameters
        ----------
        name : str or None, default=None
            Local report port name. ``None`` uses ``default_port_name``.

        Returns
        -------
        Port
            Classical ingress port owned by ``owner_agent``. Repeated calls
            with the same name return the same port object.
        """
        port_name = self.default_port_name if name is None else name
        ensure_nonempty_id(port_name, field_name="report_port_name")

        existing = self._ports.get(port_name)
        if existing is not None:
            return existing

        port = Port(
            name=port_name,
            owner=self.owner_agent,
            owner_id=self._owner_agent_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )
        self._ports[port_name] = port
        return port

    @property
    def input_port(self) -> Port:
        """Default single-source convenience report port."""
        return self.port(self.default_port_name)

    @property
    def ports(self) -> tuple[Port, ...]:
        """Known report ports sorted by local port name."""
        return tuple(self._ports[name] for name in sorted(self._ports))

    def owns(self, port: Port) -> bool:
        """Return whether ``port`` is one of this endpoint's report ports."""
        return any(port is known for known in self._ports.values())

    def extract(self, payload: object) -> object:
        """Unwrap a supported ``AGENT_REPORT`` payload.

        Parameters
        ----------
        payload : object
            ``PortDelivery``, ``AgentReport``, or raw report object carried by
            an ``AGENT_REPORT`` event.

        Returns
        -------
        object
            Report object passed to ``Agent.on_report``.

        Raises
        ------
        ValueError
            If a ``PortDelivery`` targets a port not owned by this endpoint.

        Notes
        -----
        Raw payload support is intentionally a direct-scheduled convenience for
        tests and simple control events. Port-delivered reports are checked
        against endpoint-owned ports so reports cannot be accidentally delivered
        through another agent's ingress port.
        """
        if isinstance(payload, PortDelivery):
            if not self.owns(payload.target_port):
                raise ValueError("PortDelivery target_port is not an agent report port")
            return payload.payload

        if isinstance(payload, AgentReport):
            return payload.report

        return payload


__all__ = ["AgentReportEndpoint"]
