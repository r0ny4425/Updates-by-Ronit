Routing
=======

Routing searches topology edges. It does not look at runtime wires, device
ports, memory state, entanglement state, or qstate payloads.

Use routing when you need graph metadata: a path through known nodes and links.
The returned route does not reserve resources, transmit signals, or create
entanglement.

Fewest-Hop Routes
-----------------

Use ``fewest_hops_path`` when any shortest path is good enough:

.. code-block:: python

   route = network.fewest_hops_path(
       "alice",
       "bob",
       port_kind=PortKind.QUANTUM,
   )

This returns one deterministic fewest-hop route, or ``None``. The search is
directed, and it only follows edges with the requested ``PortKind``.

If several fewest-hop routes exist, the route discovered first from
deterministic link-ID edge order is returned.

Lowest-Cost Routes
------------------

Use ``lowest_cost_path`` when each traversed link has an additive cost. The
route planner still walks explicit topology links filtered by ``PortKind``;
the callable decides what one link costs.

.. code-block:: python

   route = network.lowest_cost_path(
       "alice",
       "bob",
       port_kind=PortKind.QUANTUM,
       link_cost=lambda link: link.transport.length_m,
   )

The callable receives the ``NetworkLink``, so it can inspect ``link_id``,
endpoints, ``port_kind``, and optional ``transport`` state. Costs must be
finite and non-negative. Length, delay, attenuation in dB, and custom penalty
scores fit this model.

For success probabilities, convert multiplicative probabilities into an
additive cost, for example ``-log(probability)``.

Candidate Routes
----------------

Use ``paths_with_max_hops`` when protocol or control code wants options:

.. code-block:: python

   candidates = network.paths_with_max_hops(
       "alice",
       "bob",
       port_kind=PortKind.QUANTUM,
       max_hops=4,
   )

Candidates are simple directed paths, meaning no repeated node. They are
returned in deterministic depth-first link-ID order, not sorted by hop count.

Parallel links are preserved as separate candidates. If Alice has two links to
Bob, each link can produce its own one-hop route.

Candidate ranking is still useful for route policies that are not simple
additive link costs. A route that needs the most free memories, avoids busy
repeaters, or reserves protocol-specific workspace should usually be generated,
filtered, and ranked above the routing layer.

Route Records
-------------

``Route`` is immutable graph metadata. A zero-hop route is valid only when the
source and target node are the same.

A route exposes the pieces most callers need:

* ``hops``
* ``link_ids``
* ``node_ids``
* ``port_kinds``

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/routing.py``

.. automodule:: simyuj.network.routing
   :members:
   :show-inheritance:
