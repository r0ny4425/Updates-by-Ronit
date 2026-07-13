Metrics
=======

``simyuj.metrics`` provides deterministic scoring helpers for links and routes.
Callers provide candidate routes and per-link values; the helpers validate
values and aggregate them.

Use it to compare existing ``Route`` objects by hop count, additive cost, delay,
or independent success probability.

A Small Route Score
-------------------

Most route helpers read values from mappings keyed by link ID.

.. code-block:: pycon

   >>> from simyuj.components import PortKind
   >>> from simyuj.metrics import best_route, route_success_probability, total_link_cost
   >>> from simyuj.network import Network, Node


   >>> network = Network("metric-demo")
   >>> for node_id in ("alice", "relay", "bob"):
   ...     _ = network.add_node(Node(node_id))

   >>> _ = network.add_quantum_link("q_ab", "alice", "bob")
   >>> _ = network.add_quantum_link("q_ar", "alice", "relay")
   >>> _ = network.add_quantum_link("q_rb", "relay", "bob")

   >>> routes = network.paths_with_max_hops(
   ...     "alice",
   ...     "bob",
   ...     port_kind=PortKind.QUANTUM,
   ...     max_hops=2,
   ... )
   >>> [route.link_ids for route in routes]
   [('q_ab',), ('q_ar', 'q_rb')]

   >>> cost_by_link = {
   ...     "q_ab": 2.0,
   ...     "q_ar": 1.0,
   ...     "q_rb": 1.0,
   ... }

   >>> success_by_link = {
   ...     "q_ab": 0.80,
   ...     "q_ar": 0.95,
   ...     "q_rb": 0.90,
   ... }

   >>> cheapest = best_route(
   ...     routes,
   ...     lambda route: total_link_cost(route, cost_by_link),
   ... )
   >>> cheapest.link_ids
   ('q_ab',)

   >>> most_reliable = best_route(
   ...     routes,
   ...     lambda route: 1.0 - route_success_probability(route, success_by_link),
   ... )
   >>> most_reliable.link_ids
   ('q_ar', 'q_rb')

The helpers do not decide what a cost means. The same additive machinery can
represent distance, delay, price, loss budget, or any other finite
non-negative quantity.

Link Values
-----------

``link_metric()`` and ``edge_metric()`` read finite non-negative values.
Missing values raise ``KeyError`` unless a default is supplied.

``link_success_probability()`` and ``edge_success_probability()`` use the same
lookup pattern, but validate values as probabilities in ``[0, 1]``.

Defaults are per-lookup fallbacks. They do not fill the mapping, and route
helpers apply them independently to every missing route link.

Route Values
------------

.. list-table::
   :header-rows: 1

   * - Helper
     - Use when
   * - ``hop_count()``
     - You want the fewest directed links.
   * - ``total_link_cost()``
     - You have additive costs keyed by link ID.
   * - ``total_link_delay()``
     - You have additive delays keyed by link ID.
   * - ``total_link_metric()``
     - You want the generic additive form with your own field name.
   * - ``route_success_probability()``
     - You want independent per-link success probabilities multiplied.
   * - ``route_score()``
     - You want to compute a non-negative score directly from each edge.
   * - ``best_route()``
     - You have candidate routes and a metric where smaller is better.

Zero-hop routes return ``0.0`` for additive totals and ``1.0`` for success
probability.

Choosing Routes
---------------

``best_route()`` minimizes the metric. This is natural for hop count, cost,
delay, and penalties. If you have a reward or success probability, transform it
first.

For example, do this:

.. code-block:: python

   best_route(routes, lambda route: 1.0 - route_success_probability(route, probs))

not this:

.. code-block:: python

   best_route(routes, lambda route: route_success_probability(route, probs))

The second form picks the lowest success probability because lower metrics win.
Ties preserve input order, so callers can provide routes in deterministic
preference order.

Route Assumptions
-----------------

``route_success_probability()`` multiplies independent per-link probabilities.
Correlated failures, shared hardware, purification, swapping, retries, and
scheduling contention need protocol-level scoring.

Protocol-specific scoring belongs in controller or protocol code, usually as a
callable passed to ``best_route()``.

``route_score()`` expects non-negative costs or penalties.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Link Metrics <link>
   Path Metrics <path>
