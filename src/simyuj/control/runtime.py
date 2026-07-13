"""Explicit lifecycle runtime for node-attached control-plane agents.

``SessionRuntime`` binds a network, discovers agents registered on its nodes,
attaches event-local contexts to those agents, and schedules one deterministic
start event per agent. It does not auto-start devices; component behavior still
begins through explicit events or agent hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.entanglement.registry import EntangledPairRegistry
from simyuj.network import Network, Node
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.resources.manager import ResourceManager
from simyuj.runtime.binding import BindingContext, bind_if_supported

from .actions import AGENT_START
from .agent import Agent, ContextProvider, NodeAgent
from .context import AgentContext
from .devices import DeviceResolver
from .memory import MemoryService
from .pairs import PairService
from .payloads import AgentStart
from .resources import ResourceService
from .timers import TimerService


@dataclass(slots=True)
class SessionRuntime:
    """Bind a network and schedule node-attached agent starts explicitly.

    Parameters
    ----------
    timeline : Timeline
        Timeline that owns scheduling, event ordering, and execution.
    network : Network
        Network containing devices, ports, and nodes used by the session.
    resource_manager : ResourceManager or None, default=None
        Optional shared memory-resource manager exposed through
        ``AgentContext.resources``.
    pair_registry : EntangledPairRegistry or None, default=None
        Optional shared entangled-pair registry exposed through
        ``AgentContext.pairs``.
    topology : NetworkTopology or None, default=None
        Optional topology view. When omitted, one is built from ``network``.
    route_planner : RoutePlanner or None, default=None
        Optional route planner. When omitted, one is built from ``topology``.
    session_id : str, default="session"
        Non-empty session id stored in start payloads, contexts, and event
        metadata.
    start_time : int, default=0
        Non-negative timeline tick for scheduled ``AGENT_START`` events.
    start_priority : int, default=0
        Event priority for scheduled ``AGENT_START`` events.

    Notes
    -----
    Agents must be registered on nodes with ``Node.add_agent(...)`` before the
    runtime is constructed. ``bind_all`` binds network devices before agents.
    ``schedule_agent_starts`` then schedules starts in sorted ``agent_id`` order
    for deterministic replay.

    Examples
    --------
    >>> from simyuj.control import NodeAgent, SessionRuntime
    >>> from simyuj.engine import Timeline
    >>> from simyuj.network import Network, Node
    >>> class HelloAgent(NodeAgent):
    ...     def on_start(self, start, ctx):
    ...         self.started = (start.agent_id, ctx.session_id)
    >>> timeline = Timeline(master_seed=1)
    >>> agent = HelloAgent(agent_id="alice-agent", node_id="alice")
    >>> node = Node("alice")
    >>> _ = node.add_agent(agent)
    >>> network = Network()
    >>> _ = network.add_node(node)
    >>> runtime = SessionRuntime(
    ...     timeline=timeline,
    ...     network=network,
    ...     session_id="demo",
    ... )
    >>> _ = runtime.run()
    >>> agent.started
    ('alice-agent', 'demo')
    """

    timeline: Timeline
    network: Network

    resource_manager: ResourceManager | None = None
    pair_registry: EntangledPairRegistry | None = None
    topology: NetworkTopology | None = None
    route_planner: RoutePlanner | None = None

    session_id: str = "session"
    start_time: int = 0
    start_priority: int = 0

    _agents: tuple[Agent, ...] = field(init=False, default=())
    _agent_node_ids: dict[str, str] = field(init=False, default_factory=dict)
    _timer_services: dict[str, TimerService] = field(
        init=False,
        default_factory=dict,
    )
    _bound: bool = field(init=False, default=False)
    _starts_scheduled: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if not isinstance(self.timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        if not isinstance(self.network, Network):
            raise TypeError("network must be Network")
        ensure_nonempty_id(self.session_id, field_name="session_id")
        if type(self.start_time) is not int:
            raise TypeError("start_time must be int")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if type(self.start_priority) is not int:
            raise TypeError("start_priority must be int")

        if self.topology is None:
            self.topology = NetworkTopology(self.network)
        elif self.topology.network is not self.network:
            raise ValueError("topology must reference this runtime network")

        if self.route_planner is None:
            self.route_planner = RoutePlanner(self.topology)
        elif self.route_planner.topology is not self.topology:
            raise ValueError("route_planner must reference this runtime topology")

        self._collect_agents()
        self._validate_agents()
        self._attach_context_providers()

    @property
    def agents(self) -> tuple[Agent, ...]:
        """Return agents discovered from network nodes in runtime order."""

        return self._agents

    def _collect_agents(self) -> None:
        agents: list[Agent] = []
        agent_node_ids: dict[str, str] = {}
        seen: set[str] = set()

        for node_id in sorted(self.network.nodes):
            node = self.network.nodes[node_id]
            for agent_id in sorted(node.agents):
                agent = node.agents[agent_id]
                if not isinstance(agent, Agent):
                    raise TypeError("node agents must contain only Agent instances")
                if agent.agent_id != agent_id:
                    raise ValueError("node agent key must match agent_id")
                if agent_id in seen:
                    raise ValueError(f"duplicate agent id '{agent_id}'")
                seen.add(agent_id)

                if isinstance(agent, NodeAgent) and agent.node_id != node_id:
                    raise ValueError(
                        f"agent node_id '{agent.node_id}' does not match node "
                        f"'{node_id}'"
                    )

                agents.append(agent)
                agent_node_ids[agent_id] = node_id

        self._agents = tuple(sorted(agents, key=lambda item: item.agent_id))
        self._agent_node_ids = agent_node_ids

    def _validate_agents(self) -> None:
        seen: set[str] = set()
        for agent in self._agents:
            if not isinstance(agent, Agent):
                raise TypeError("agents must contain only Agent instances")
            if agent.agent_id in seen:
                raise ValueError(f"duplicate agent id '{agent.agent_id}'")
            seen.add(agent.agent_id)
            node_id = self._agent_node_ids.get(agent.agent_id)
            if node_id not in self.network.nodes:
                raise ValueError(f"unknown node_id '{node_id}' for agent")

    def _attach_context_providers(self) -> None:
        def make_provider(agent: Agent) -> ContextProvider:
            def provider(event: Event, timeline: Timeline) -> AgentContext:
                return self._context_for_agent(agent, event, timeline)

            return provider

        for agent in self._agents:
            agent.attach_context_provider(make_provider(agent))

    def _context_for_agent(
        self,
        agent: Agent,
        event: Event,
        timeline: Timeline,
    ) -> AgentContext:
        node_id = self._agent_node_ids.get(agent.agent_id)
        node: Node | None = (
            self.network.get_node(node_id) if node_id is not None else None
        )
        timers = self._timer_service_for(agent, timeline)
        devices = DeviceResolver(node=node) if node is not None else None
        memory = (
            MemoryService(
                node=node,
                timeline=timeline,
                session_id=self.session_id,
                source_agent=agent,
            )
            if node is not None
            else None
        )
        resources = (
            ResourceService(self.resource_manager, owner_agent_id=agent.agent_id)
            if self.resource_manager is not None
            else None
        )
        pairs = (
            PairService(self.pair_registry, timeline=timeline)
            if self.pair_registry is not None
            else None
        )
        topology = self.topology
        route_planner = self.route_planner
        if topology is None or route_planner is None:
            raise RuntimeError("runtime topology and route planner are not initialized")

        return AgentContext(
            agent_id=agent.agent_id,
            node_id=node_id,
            session_id=self.session_id,
            timeline=timeline,
            event=event,
            network=self.network,
            topology=topology,
            route_planner=route_planner,
            node=node,
            timers=timers,
            devices=devices,
            memory=memory,
            resources=resources,
            pairs=pairs,
            classical=agent._classical_endpoint,
        )

    def _timer_service_for(self, agent: Agent, timeline: Timeline) -> TimerService:
        service = self._timer_services.get(agent.agent_id)
        if service is None:
            service = TimerService(
                owner_agent=agent,
                timeline=timeline,
                session_id=self.session_id,
            )
            self._timer_services[agent.agent_id] = service
        return service

    def bind_all(self) -> None:
        """Bind network devices, then bind agents, at most once.

        Notes
        -----
        Repeated calls are no-ops. Binding order is deterministic: network
        binding runs first, then agents are bound in sorted ``agent_id`` order.
        """
        if self._bound:
            return

        context = BindingContext(
            timeline=self.timeline,
            logger=self.timeline.logger,
            entity_id=self.session_id,
            component_id="session_runtime",
            meta=(("session_id", self.session_id),),
        )

        self.network.bind_all(context)

        for agent in self._agents:
            bind_if_supported(
                agent,
                self.timeline,
                self.timeline.logger,
                entity_id=self.session_id,
                component_id=agent.agent_id,
                meta=(("agent_id", agent.agent_id),),
            )

        self._bound = True

    def schedule_agent_starts(self) -> tuple[Event, ...]:
        """Schedule one ``AGENT_START`` event per agent in stable order.

        Returns
        -------
        tuple[Event, ...]
            Scheduled start events sorted by ``agent_id``.

        Raises
        ------
        RuntimeError
            If ``bind_all`` has not run, or if starts were already scheduled.
        """
        if not self._bound:
            raise RuntimeError(
                "bind_all() must be called before scheduling agent starts"
            )
        if self._starts_scheduled:
            raise RuntimeError("agent starts already scheduled")

        events: list[Event] = []
        for agent in self._agents:
            node_id = self._agent_node_ids.get(agent.agent_id)
            event = self.timeline.schedule(
                Event(
                    time=self.start_time,
                    priority=self.start_priority,
                    target_ref=agent,
                    action=AGENT_START,
                    payload_ref=AgentStart(
                        agent_id=agent.agent_id,
                        node_id=node_id,
                        session_id=self.session_id,
                    ),
                    source=self,
                    subsystem_id="control",
                    meta={
                        "session_id": self.session_id,
                        "agent_id": agent.agent_id,
                        "node_id": node_id,
                    },
                )
            )
            events.append(event)

        self._starts_scheduled = True
        return tuple(events)

    def run_until_empty(self):
        """Drain events already scheduled on the timeline.

        This method does not call ``bind_all`` or schedule agent starts.
        """
        return self.timeline.run_until_empty()

    def run_until(self, t_end: int) -> None:
        """Advance the timeline until ``t_end`` using existing scheduled events.

        Unlike ``run``, this method does not call ``bind_all`` or schedule
        ``AGENT_START`` events.
        """
        self.timeline.run_until(t_end)

    def run(self):
        """Bind, schedule starts if needed, then drain the timeline."""
        if not self._bound:
            self.bind_all()
        if not self._starts_scheduled:
            self.schedule_agent_starts()
        return self.run_until_empty()


__all__ = ["SessionRuntime"]
