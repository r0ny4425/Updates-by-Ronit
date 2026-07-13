"""Event-local runtime context passed to control-plane agents.

``AgentContext`` groups the current event, timeline, network views, and optional
control services made available to an agent hook. The record is immutable, but
the objects it references, such as ``Timeline`` and service instances, remain
live runtime objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.network import Network, Node
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology

if TYPE_CHECKING:
    from .classical import ClassicalEndpoint
    from .devices import DeviceResolver
    from .memory import MemoryService
    from .pairs import PairService
    from .resources import ResourceService
    from .timers import TimerService


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentContext:
    """Event-local runtime services available to an agent hook.

    Parameters
    ----------
    agent_id : str
        Identifier of the agent receiving the event.
    node_id : str or None
        Node id where this agent was registered, otherwise ``None``.
    session_id : str
        Runtime session identifier.
    timeline : Timeline
        Timeline currently dispatching the event.
    event : Event
        Event being handled.
    network : Network
        Runtime network.
    topology : NetworkTopology
        Topology view for route planning.
    route_planner : RoutePlanner
        Route planner built for the runtime topology.
    node : Node or None, default=None
        Bound node for node-local agents.
    timers, devices, memory, resources, pairs, classical : service or None
        Optional service handles attached by ``SessionRuntime`` according to
        the agent type and configured runtime subsystems.

    Notes
    -----
    ``timers`` is always present for agents created through ``SessionRuntime``.
    ``devices`` and ``memory`` are present for agents attached to a node.
    ``resources`` and ``pairs`` require the corresponding manager or registry
    to be supplied to the runtime. ``classical`` is present only after the agent
    enables a classical endpoint.
    """

    agent_id: str
    node_id: str | None
    session_id: str

    timeline: Timeline
    event: Event

    network: Network
    topology: NetworkTopology
    route_planner: RoutePlanner

    node: Node | None = None

    timers: TimerService | None = None
    devices: DeviceResolver | None = None
    memory: MemoryService | None = None
    resources: ResourceService | None = None
    pairs: PairService | None = None
    classical: ClassicalEndpoint | None = None


__all__ = ["AgentContext"]
