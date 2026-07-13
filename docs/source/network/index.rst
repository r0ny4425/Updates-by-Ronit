Network
=======

The network package describes the shape of a simulated quantum network: which
named nodes exist, which physical links connect them, and which component ports
are wired for event delivery.

It does not run protocols. It does not reserve memories. It does not create
entanglement. Those jobs belong to the timeline, resource, entanglement,
control, and protocol layers.

The main idea is: **links describe topology; wires deliver events**.

A link can make Alice and Bob neighbors. A wire can deliver a detector report,
a memory notice, or a signal from one component port to another. Those are
intentionally different things.

A First Network
---------------

Start by naming the places where local devices and agents live:

.. code-block:: python

   from simyuj.components import PortKind, QuantumChannel
   from simyuj.network import Network, Node

   network = Network("two-node-demo")
   network.add_node(Node("alice"))
   network.add_node(Node("bob"))
   quantum_channel = QuantumChannel(channel_id="q_alice_bob_channel")

Then add an explicit topology link:

.. code-block:: python

   from simyuj.components import QuantumChannel
   from simyuj.network import Network, Node

   network = Network("link-demo")
   network.add_node(Node("alice"))
   network.add_node(Node("bob"))
   quantum_channel = QuantumChannel(channel_id="q_alice_bob_channel")

   network.add_quantum_link(
       "q_alice_bob",
       "alice",
       "bob",
       channel=quantum_channel,
   )

This says Alice can reach Bob through a quantum link. It does not send a photon
yet, and it does not connect any component ports.

Runtime delivery is installed with a wire:

.. code-block:: python

   wire = network.wire_ports(
       "source_to_channel",
       source.output_port,
       quantum_channel.input_port,
       target_action=ACTION_TRANSMIT_QUANTUM,
   )

   wire.transmit(signal, timeline)

``wire_ports`` creates a ``PortConnection``. ``wire.transmit(...)`` schedules a
timeline event for the owner of the target port. No network API should call a
component's ``handle_event`` method directly.

Mental Model
------------

Each kind of network fact has one home:

.. list-table::
   :header-rows: 1

   * - Concept
     - Use it for
     - Where to read more
   * - ``Node``
     - Naming local devices, agents, and port aliases.
     - :doc:`node`
   * - ``NetworkLink``
     - Recording directed physical reachability between nodes.
     - :doc:`link`
   * - ``PortConnection``
     - Delivering payloads from one component-owned port to another.
     - :doc:`network`
   * - ``Route``
     - Describing a selected path through topology.
     - :doc:`routing`

When To Use What
----------------

Use ``Node`` when you need a named local namespace for devices, agents, or
human-friendly port aliases.

Use ``add_quantum_link`` or ``add_classical_link`` when routing, metrics, or
topology queries should see a physical connection between nodes.

Use ``wire_ports`` when a payload should move between component ports during
timeline execution.

Use routing helpers when protocol or control code needs a path through the
topology:

.. code-block:: pycon

   >>> from simyuj.components import PortKind, QuantumChannel
   >>> from simyuj.network import Network, Node
   >>> network = Network("route-demo")
   >>> _ = network.add_node(Node("alice"))
   >>> _ = network.add_node(Node("bob"))
   >>> channel = QuantumChannel(channel_id="q_alice_bob_channel")
   >>> _ = network.add_quantum_link(
   ...     "q_alice_bob",
   ...     "alice",
   ...     "bob",
   ...     channel=channel,
   ... )

   >>> route = network.fewest_hops_path(
   ...     "alice",
   ...     "bob",
   ...     port_kind=PortKind.QUANTUM,
   ... )
   >>> route.link_ids
   ('q_alice_bob',)

A route is metadata only. It does not reserve memory, transmit a photon,
schedule events, or create entanglement.

Common Shape
------------

A channel often appears in both places:

.. code-block:: python

   network.add_quantum_link(
       "q_alice_bob",
       "alice",
       "bob",
       channel=quantum_channel,
   )

   network.wire_ports(
       "source_to_channel",
       source.output_port,
       quantum_channel.input_port,
       target_action=ACTION_TRANSMIT_QUANTUM,
   )

   network.wire_ports(
       "channel_to_receiver",
       quantum_channel.output_port,
       receiver.input_port,
       target_action="receive_signal",
   )

The link makes Alice and Bob neighbors in the quantum topology. The wires make
the event path. The channel handles the transmission because it is on that path.

Developer Boundary
------------------

Think of the network package as the simulator's map. It knows the nodes, the
physical links between them, and the wires that carry events between ports. The
actual decisions still happen elsewhere.

When reading or writing network code, use this rule of thumb:

* ``add_*_link`` changes topology.
* ``wire_ports`` changes runtime delivery.
* routes describe paths, but do not execute them.
* report, notice, local memory, detector, and agent-message connections should
  be wires, not topology links.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Node <node>
   Network Link <link>
   Network Registry <network>
   Topology <topology>
   Routing <routing>
   Planning <planning>
