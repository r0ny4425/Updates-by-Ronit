Network Registry
================

``Network`` is the object most user code talks to. It holds the nodes, the
physical links, and the runtime wires.

It is useful to read those three collections separately:

* ``network.nodes`` is the named set of node namespaces.
* ``network.links`` is the physical graph used by topology, routing, and
  metrics.
* ``network.wires`` is runtime port plumbing used to schedule delivery events.

Keeping links and wires separate avoids a common mistake: a detector report
wire, an agent message wire, or a source-to-channel wire should not create a
new graph edge.

Creating Nodes
--------------

.. code-block:: python

   from simyuj.network import Network, Node

   network = Network("demo")
   network.add_node(Node("alice"))
   network.add_node(Node("bob"))

Node IDs must be unique. Nodes may contain devices, agents, and optional port
aliases. Those aliases are useful for humans and controllers, but topology does
not depend on them.

Adding Links
------------

Use explicit topology APIs when you want the graph to say that one node can
reach another:

.. code-block:: python

   from simyuj.components import ClassicalChannel, QuantumChannel
   from simyuj.network import Network, Node

   network = Network("link-demo")
   network.add_node(Node("alice"))
   network.add_node(Node("bob"))

   q_channel = QuantumChannel(channel_id="q_ab_channel")
   c_channel = ClassicalChannel(channel_id="c_ab_channel")

   network.add_quantum_link("q_ab", "alice", "bob", channel=q_channel)
   network.add_classical_link("c_ab", "alice", "bob", channel=c_channel)

Link IDs must be unique. Both calls create ``NetworkLink`` records in
``network.links``. Filtered views are available as ``network.quantum_links`` and
``network.classical_links``.

The optional ``channel`` is stored as ``link.transport``. That lets routing or
metrics inspect link-owned state, and lets ``Network.bind_all`` bind the
transport before execution. It does not automatically connect device ports.
Runtime delivery still needs wires.

Wiring Ports
------------

Use ``wire_ports`` when an event needs to travel from one component port to
another:

.. code-block:: python

   wire = network.wire_ports(
       "detector_report",
       detector.output_port,
       agent.reports.port("detector"),
       target_action=AGENT_REPORT,
   )

Wire IDs must be unique. The source and target are actual ``Port`` objects.
They do not need to be registered as node port aliases first.

Runtime wires are kept in ``network.wires``. They are deliberately ignored by
``edges``, ``neighbors``, and route search. This is what keeps report ports,
notice ports, and local control plumbing out of topology.

``wire_ports`` returns a ``PortConnection``. The connection is the object that
schedules the delivery event:

.. code-block:: python

   wire = network.wire_ports(
       "q_wire_ab",
       source.output_port,
       sink.input_port,
       target_action="receive_signal",
   )

   wire.transmit(signal, timeline)

The call to ``wire_ports`` only installs the connection. The call to
``wire.transmit(...)`` schedules an event on ``timeline``. When the timeline
executes it, the target is ``sink`` because ``sink`` owns the target port. The
event action is ``"receive_signal"`` unless the caller overrides it for that
specific transmission.

If a channel should model the transport, wire through the channel:

.. code-block:: python

   network.add_quantum_link("q_ab", "alice", "bob", channel=q_channel)

   network.wire_ports(
       "source_to_channel",
       source.output_port,
       q_channel.input_port,
       target_action=ACTION_TRANSMIT_QUANTUM,
   )

   network.wire_ports(
       "channel_to_sink",
       q_channel.output_port,
       sink.input_port,
       target_action="receive_signal",
   )

Now the source delivers to the channel first. The channel can apply delay,
loss, noise, and metadata before it forwards to the sink.

Binding
-------

``bind_all`` binds node devices first, then link transports, in deterministic
ID order. If the same object appears more than once, it is bound once.

Registered agents are not bound here. Agent lifecycle belongs to the runtime
that starts and coordinates control-plane work.

For Developers
--------------

The network layer is intentionally thin. It validates IDs, stores records, and
creates ``PortConnection`` objects. It should not call component
``handle_event`` methods, run protocol logic, inspect qstate payloads, or infer
topology from arbitrary wires.

When adding new network behavior, decide which collection owns the fact:

* Node-local names belong on ``Node``.
* Graph reachability belongs in ``NetworkLink``.
* Runtime delivery belongs in ``PortConnection`` through ``wire_ports``.
* Route scoring belongs in routing, planning, metrics, or caller policy.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/network.py``

.. automodule:: simyuj.network.network
   :members:
   :show-inheritance:
