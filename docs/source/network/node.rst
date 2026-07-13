Node
====

``Node`` is a named local namespace inside a network. It gives stable names to
the devices, agents, and component-owned ports that live at one simulated
location.

A node does not run anything by itself. Components still own their behavior,
ports still belong to components, and control agents are started by the runtime,
not by ``Node``.

What Belongs On A Node
----------------------

Use a node for local names:

.. code-block:: python

   alice = Node("alice")

   alice.add_device("source", source)
   alice.register_port("source_out", source.output_port)

Devices are stored by node-local name. Port aliases are also node-local names;
they point to ports that already belong to a component.

Agents
------

Nodes can also register control agents:

.. code-block:: python

   alice.add_agent(alice_agent)

If the agent is a ``NodeAgent``, its ``node_id`` must match the node. Registering
the agent only records that it belongs to this node. It does not bind, start, or
schedule the agent.

Port Aliases
------------

Port aliases are convenience names for user code, controllers, and examples:

.. code-block:: python

   alice.register_port("source_out", source.output_port)

They do not make topology. Topology is created with
``Network.add_link(...)``, ``Network.add_quantum_link(...)``, or
``Network.add_classical_link(...)``.

A channel between Alice and Bob should usually be attached to the link between
Alice and Bob as transport, not registered as a device on either node just so
routing can find it.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/node.py``

.. automodule:: simyuj.network.node
   :members:
   :show-inheritance:
