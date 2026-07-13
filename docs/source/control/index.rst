Control
=======

``simyuj.control`` provides event-driven agents and services for node-local
protocol decisions. A control agent is a simulator component: it receives
timeline events, reacts in typed hooks, and schedules follow-up work through the
same deterministic event engine as devices and channels.

Use it for protocol flow, retries, classical coordination, memory requests,
resource reservations, and entangled-pair bookkeeping. Concrete protocol policy
lives in agent subclasses or protocol packages.

A First Agent
-------------

A minimal agent subclasses ``NodeAgent`` and implements the hooks it needs.
The runtime discovers agents registered on nodes, binds the network, and then
schedules one ``agent_start`` event per agent.

.. code-block:: pycon

   >>> from dataclasses import dataclass, field

   >>> from simyuj.control import AgentContext, NodeAgent, SessionRuntime
   >>> from simyuj.engine import Timeline
   >>> from simyuj.network import Network, Node


   >>> @dataclass(slots=True)
   ... class HelloAgent(NodeAgent):
   ...     started_nodes: list[str | None] = field(default_factory=list)
   ...
   ...     def on_start(self, start, ctx: AgentContext) -> None:
   ...         del start
   ...         self.started_nodes.append(ctx.node_id)

   >>> timeline = Timeline(master_seed=1)

   >>> node = Node("alice")
   >>> agent = HelloAgent(agent_id="alice-control", node_id="alice")
   >>> _ = node.add_agent(agent)

   >>> network = Network()
   >>> _ = network.add_node(node)

   >>> runtime = SessionRuntime(timeline=timeline, network=network)
   >>> _ = runtime.run()

   >>> agent.started_nodes
   ['alice']

Agent Events
------------

Control agents do not run in a side loop. They receive normal timeline events,
and unsupported actions are routed to ``Agent.on_event``. The base
implementation raises ``ValueError`` so subclasses must explicitly opt in to
custom control actions.

.. list-table::
   :header-rows: 1

   * - Event action
     - Delivered to
     - Typical use
   * - ``agent_start``
     - ``on_start()``
     - Initialize agent state and schedule first work.
   * - ``agent_timer``
     - ``on_timer()``
     - Retry, deadline, timeout, or delayed control action.
   * - ``agent_message``
     - ``on_message()``
     - Classical message delivery.
   * - ``agent_report``
     - ``on_report()``
     - Device or component report delivery.
   * - ``agent_event``
     - ``on_event()``
     - Caller-defined control events.

Runtime Context
---------------

``SessionRuntime`` owns the control lifecycle. Agents are attached to nodes with
``Node.add_agent(...)`` before the runtime is constructed. The runtime validates
agent ids across all nodes, checks ``NodeAgent`` node bindings, builds topology
and route-planning views when not provided, binds network devices before
agents, and schedules start events in sorted ``agent_id`` order for
deterministic replay.

Every hook receives an ``AgentContext``. The context record is immutable, but
it points to live runtime objects such as the timeline, network, and optional
services.

.. list-table::
   :header-rows: 1

   * - Service
     - Present when
     - Purpose
   * - ``ctx.timers``
     - Always under ``SessionRuntime``
     - Schedule callbacks to the same agent.
   * - ``ctx.classical``
     - The agent enabled a classical endpoint
     - Send messages through connected classical ports.
   * - ``ctx.devices``
     - The agent is attached to a node
     - Resolve node-local devices and ports.
   * - ``ctx.memory``
     - The agent is attached to a node
     - Schedule quantum-memory request events.
   * - ``ctx.resources``
     - A ``ResourceManager`` was supplied
     - Reserve and update memory-slot bookkeeping.
   * - ``ctx.pairs``
     - An ``EntangledPairRegistry`` was supplied
     - Register, query, and update entangled-pair lifecycle state.

Services
--------

Classical and report endpoints create stable classical ports owned by the
agent component. The endpoint objects are not components and are never timeline
targets. Port-delivered payloads are checked against endpoint-owned ingress
ports before the agent hook receives the normalized object.

Timers are ordinary timeline events targeted back to the owning agent. The
runtime keeps one timer service per agent, so timer ids are agent-local.
``replace=True``, ``set_once=True``, and cancellation by timer id work across
separate hooks in the same session.

Memory, resource, and pair services delegate to existing component or registry
surfaces. ``MemoryService`` schedules quantum-memory request events at the
current timeline tick. ``ResourceService`` delegates memory-slot reservations
and lifecycle updates to ``ResourceManager`` while using the current agent id
as reservation owner. ``PairService`` delegates entangled-pair registration,
lookup, and lifecycle transitions to ``EntangledPairRegistry``.

Raw ``ResourceManager`` and ``EntangledPairRegistry`` objects are runtime-owned
and are not exposed on ``AgentContext``. Agent code uses ``ctx.resources`` and
``ctx.pairs``.

Runtime Notes
-------------

Schedule work through the timeline. Device behavior, memory operations,
reports, and classical messages stay visible as timeline events.

Keep protocol policy in agent subclasses or protocol packages. The control
package provides reusable machinery for agents, messages, timers, requests, and
resource views.

Use explicit identifiers for protocol-level requests. Some services can
generate deterministic request ids for convenience, but durable protocol ids
come from the caller.

Randomness, event ordering, time advancement, cancellation, and event
identifiers stay with the engine timeline.

``runtime.run_until(t)`` only advances events that are already scheduled. It
does not bind the runtime or schedule agent starts. Use ``runtime.run()`` for
the normal lifecycle, or call ``bind_all()`` and ``schedule_agent_starts()``
explicitly when you need manual control.

Classical endpoints can send through an explicit local output port, or route by
``message.receiver_id`` when ``port_name`` is omitted. Missing address-book
routes raise ``RoutingError``. Agent ``on_message`` hooks receive an
``AgentMessage`` wrapper containing the transport message, receive time, and
optional port/connection context.

``delay=0`` timers and same-tick memory requests still go through the timeline.
They are not immediate callbacks or direct method calls.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Actions <actions>
   Agent <agent>
   Agent Context <context>
   Session Runtime <runtime>
   Device Resolver <devices>
   Timer Service <timers>
   Classical Endpoint <classical>
   Memory Service <memory>
   Resource Service <resources>
   Pair Service <pairs>
   Payloads <payloads>
   Reports <reports>
