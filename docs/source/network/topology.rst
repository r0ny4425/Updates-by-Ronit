Topology
========

Topology is the node graph built from explicit ``NetworkLink`` records. Runtime
wires are ignored.

This means topology answers questions such as:

* Which node IDs are known?
* Which directed links leave Alice?
* Is Bob reachable from Alice over quantum links?
* Which link IDs make up a candidate route?

It does not answer runtime delivery questions such as:

* Which component owns this port?
* What event action will be delivered?
* Did a source emit a signal?
* Did a channel apply delay, loss, or noise?

Those runtime questions belong to ports, connections, components, and the
timeline.

Most user code should ask ``Network`` directly:

.. code-block:: python

   network.edges
   network.neighbors("alice", port_kind=PortKind.QUANTUM)
   network.outgoing_edges("alice")
   network.incoming_edges("bob")
   network.has_edge("alice", "bob", port_kind=PortKind.QUANTUM)

``NetworkTopology`` is the lower-level object used by ``Network`` and
``RoutePlanner``. It rebuilds ``TopologyEdge`` records from the current links
each time it is queried, so it reflects later ``add_*_link`` calls.

Ordering And Parallel Links
---------------------------

Topology views are deterministic:

* ``nodes()`` returns node IDs in sorted order.
* ``edges`` returns topology edges in link-ID order.
* ``neighbors(...)`` returns unique neighbor node IDs in sorted order.

Parallel links are preserved as separate edges. If Alice has two quantum links
to Bob, ``neighbors("alice")`` still returns Bob once, but ``outgoing_edges`` and
``edges`` include both link IDs.

TopologyEdge
------------

``TopologyEdge`` is intentionally small. It contains graph data only:

* ``link_id``
* ``source_node_id``
* ``target_node_id``
* ``port_kind``

It does not contain source ports, target ports, target actions, or notice/report
wires. If you need the channel object, look up the link with
``network.get_link(edge.link_id)`` and inspect ``link.transport``.

Developer Notes
---------------

``NetworkTopology`` is a live view over ``Network.links``. It should stay a
read-only projection. Avoid teaching it about component ports or
``PortConnection`` records; that would make route planning depend on runtime
plumbing and would blur the boundary between graph reachability and event
delivery.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/topology.py``

.. automodule:: simyuj.network.topology
   :members:
   :show-inheritance:
