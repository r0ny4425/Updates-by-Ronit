"""Generic event-driven control-plane agent components.

Agents are simulator components that receive scheduled events and dispatch them
to typed hook methods. They are intentionally protocol-neutral: subclasses own
policy, while ``SessionRuntime`` supplies event-local context and schedules the
initial start events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simyuj.components.connections import PortDelivery
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.messages.transport import ClassicalMessage
from simyuj.runtime.binding import BindingContext

from .actions import AGENT_EVENT, AGENT_MESSAGE, AGENT_REPORT, AGENT_START, AGENT_TIMER
from .context import AgentContext
from .payloads import AgentMessage, AgentStart, TimerFired
from .reports import AgentReportEndpoint

if TYPE_CHECKING:
    from simyuj.components.ports import Port

    from .classical import ClassicalEndpoint


ContextProvider = Callable[[Event, Timeline], AgentContext]


@dataclass(slots=True)
class Agent(Component):
    """Base component for user and control-plane orchestration policy.

    Parameters
    ----------
    agent_id : str
        Non-empty identifier used for event metadata, report ports, and
        optional classical endpoints.

    Notes
    -----
    ``Agent.handle_event`` recognizes the generic control actions and dispatches
    them to ``on_start``, ``on_timer``, ``on_message``, ``on_report``, or
    ``on_event``. The default typed hooks are no-ops except ``on_event``, which
    raises for unsupported actions. Subclasses usually override only the hooks
    they need.
    """

    agent_id: str

    _report_endpoint: AgentReportEndpoint = field(init=False, repr=False)
    _classical_endpoint: ClassicalEndpoint | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _context_provider: ContextProvider | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.agent_id, field_name="agent_id")
        self._report_endpoint = AgentReportEndpoint(owner_agent=self)

    @property
    def report_port(self) -> Port:
        """Default report ingress port for this agent."""
        return self._report_endpoint.input_port

    @property
    def reports(self) -> AgentReportEndpoint:
        """Report endpoint that owns this agent's report ingress ports."""
        return self._report_endpoint

    def enable_classical(self, address: str | None = None) -> ClassicalEndpoint:
        """Create or return this agent's classical endpoint.

        Parameters
        ----------
        address : str or None, default=None
            Optional endpoint address. ``None`` uses ``agent_id``.

        Returns
        -------
        ClassicalEndpoint
            Stable endpoint instance owned by this agent.

        Notes
        -----
        ``address`` is applied only when the endpoint is first created. Later
        calls return the existing endpoint without reconfiguring it.
        """
        if self._classical_endpoint is None:
            from .classical import ClassicalEndpoint

            self._classical_endpoint = ClassicalEndpoint(self, address=address)
        return self._classical_endpoint

    @property
    def classical(self) -> ClassicalEndpoint:
        """Classical endpoint, created on first access."""
        return self.enable_classical()

    def bind(self, context: BindingContext) -> None:
        """Default binding hook for runtime compatibility."""
        del context
        return None

    def attach_context_provider(self, provider: ContextProvider) -> None:
        """Attach the runtime callback used to build event-local context."""
        if not callable(provider):
            raise TypeError("provider must be callable")
        self._context_provider = provider

    def _context_for(self, event: Event, timeline: Timeline) -> AgentContext:
        if self._context_provider is None:
            raise RuntimeError("Agent is not attached to a SessionRuntime")
        return self._context_provider(event, timeline)

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """Dispatch one scheduled event to the matching agent hook.

        Parameters
        ----------
        event : Event
            Event targeted to this agent.
        timeline : Timeline
            Timeline currently dispatching the event.

        Raises
        ------
        TypeError
            If ``event`` or ``timeline`` has the wrong type, or if a known
            action carries an incompatible payload.
        ValueError
            If ``event.target_ref`` is not this agent. The default
            ``on_event`` also raises ``ValueError`` for unsupported actions.
        RuntimeError
            If the agent has not been attached to a ``SessionRuntime`` or other
            context provider.

        Notes
        -----
        Cross-component behavior should still be modeled by scheduling events
        on the timeline. Hook methods receive ``AgentContext`` so they can use
        services such as timers, memory, reports, and classical messaging.
        Agents with an enabled classical endpoint validate ``PortDelivery``
        targets against endpoint-owned ingress ports; raw direct-scheduled
        ``ClassicalMessage`` events remain supported.
        """
        if not isinstance(event, Event):
            raise TypeError("event must be Event")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        if event.target_ref is not self:
            raise ValueError("event target_ref must be this Agent")

        ctx = self._context_for(event, timeline)

        if event.action == AGENT_START:
            if not isinstance(event.payload_ref, AgentStart):
                raise TypeError("AGENT_START payload_ref must be AgentStart")
            self.on_start(event.payload_ref, ctx)
            return

        if event.action == AGENT_TIMER:
            if not isinstance(event.payload_ref, TimerFired):
                raise TypeError("AGENT_TIMER payload_ref must be TimerFired")
            self.on_timer(event.payload_ref, ctx)
            return

        if event.action == AGENT_MESSAGE:
            self.on_message(self._extract_message(event.payload_ref, event.time), ctx)
            return

        if event.action == AGENT_REPORT:
            self.on_report(self._report_endpoint.extract(event.payload_ref), ctx)
            return

        if event.action == AGENT_EVENT:
            self.on_event(event, ctx)
            return

        self.on_event(event, ctx)

    def _extract_message(self, payload: object, time: int) -> AgentMessage:
        if self._classical_endpoint is not None:
            return self._classical_endpoint.extract(payload, time)

        if isinstance(payload, PortDelivery):
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

    def on_start(self, start: AgentStart, ctx: AgentContext) -> None:
        """Handle an ``AGENT_START`` payload.

        Subclasses override this hook to initialize control-plane state or
        schedule first actions. The base implementation does nothing.
        """
        pass

    def on_timer(self, timer: TimerFired, ctx: AgentContext) -> None:
        """Handle an ``AGENT_TIMER`` callback.

        The base implementation does nothing.
        """
        pass

    def on_message(self, message: AgentMessage, ctx: AgentContext) -> None:
        """Handle an ``AGENT_MESSAGE`` classical message.

        The base implementation does nothing.
        """
        pass

    def on_report(self, report: object, ctx: AgentContext) -> None:
        """Handle an ``AGENT_REPORT`` payload after endpoint extraction.

        The base implementation does nothing.
        """
        pass

    def on_event(self, event: Event, ctx: AgentContext) -> None:
        """Handle custom or unsupported event actions.

        The base implementation raises ``ValueError`` so subclasses must opt in
        to custom event actions explicitly.
        """
        del ctx
        raise ValueError(f"unsupported event action for Agent: {event.action!r}")


@dataclass(slots=True)
class NodeAgent(Agent):
    """Agent that declares an expected network-node binding.

    Parameters
    ----------
    agent_id : str
        Non-empty agent identifier.
    node_id : str
        Non-empty network node id. ``Node.add_agent`` and ``SessionRuntime``
        validate that the agent is registered on the matching node.
    """

    node_id: str

    def __post_init__(self) -> None:
        Agent.__post_init__(self)
        ensure_nonempty_id(self.node_id, field_name="node_id")


__all__ = [
    "Agent",
    "ContextProvider",
    "NodeAgent",
]
