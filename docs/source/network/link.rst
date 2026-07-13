Network Link
============

``NetworkLink`` is the record for one directed physical edge in the network
graph. It has a source node, a target node, a ``PortKind``, and optional
transport state.

It answers a graph question: "can this kind of link carry traffic from this
node to that node?"

Links Are Directed
------------------

A link from Alice to Bob does not automatically create a link from Bob to Alice:

.. code-block:: python

   from simyuj.components import QuantumChannel
   from simyuj.network import Network, Node

   network = Network("link-demo")
   network.add_node(Node("alice"))
   network.add_node(Node("bob"))
   quantum_channel = QuantumChannel(channel_id="alice_to_bob")

   network.add_quantum_link(
       "q_alice_bob",
       "alice",
       "bob",
       channel=quantum_channel,
   )

If the model needs both directions, add two links with two link IDs.

Transport State
---------------

A common case is a quantum or classical channel stored as link transport:

.. code-block:: pycon

   >>> from simyuj.components import QuantumChannel
   >>> from simyuj.network import Network, Node

   >>> network = Network("transport-demo")
   >>> _ = network.add_node(Node("alice"))
   >>> _ = network.add_node(Node("bob"))
   >>> quantum_channel = QuantumChannel(channel_id="alice_to_bob")

   >>> link = network.add_quantum_link(
   ...     "q_alice_bob",
   ...     "alice",
   ...     "bob",
   ...     channel=quantum_channel,
   ... )

   >>> link.transport is quantum_channel
   True

The channel belongs to the link. It is not registered as a device on Alice or
Bob just so routing can find it. The link gives metrics, route planners, and
binding code a stable place to find link-owned state:

.. code-block:: pycon

   >>> from simyuj.components import QuantumChannel
   >>> from simyuj.network import Network, Node
   >>> network = Network("lookup-demo")
   >>> _ = network.add_node(Node("alice"))
   >>> _ = network.add_node(Node("bob"))
   >>> quantum_channel = QuantumChannel(channel_id="alice_to_bob")
   >>> _ = network.add_quantum_link(
   ...     "q_alice_bob",
   ...     "alice",
   ...     "bob",
   ...     channel=quantum_channel,
   ... )

   >>> network.get_link("q_alice_bob").transport is quantum_channel
   True

That still does not mean the channel will see a payload. It will see payloads
only if runtime wires deliver to its input port.

Runtime Delivery Is Separate
----------------------------

Wire component ports when a payload should actually move through the channel:

.. code-block:: python

   network.wire_ports(
       "source_to_channel",
       source.output_port,
       quantum_channel.input_port,
       target_action=ACTION_TRANSMIT_QUANTUM,
   )

``NetworkLink`` does not wrap a ``PortConnection``. The link answers graph
questions. The wire answers delivery questions.

Developer Notes
---------------

Keep ``NetworkLink`` small. It should stay immutable graph metadata plus
optional transport state. It should not grow source or target ports, target
actions, protocol state, or scheduling behavior. Those belong to components,
``PortConnection``, control agents, or higher-level protocol code.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/link.py``

.. automodule:: simyuj.network.link
   :members:
   :show-inheritance:
