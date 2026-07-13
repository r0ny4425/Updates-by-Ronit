Planning
========

Planning helpers choose among candidate routes. They are where user code can
say what "best" means.

Use ``network.lowest_cost_path`` first when the route score is simply the sum
of non-negative link costs. Use this page when you already have candidates, or
when the score depends on the whole route rather than one link at a time.

Planning helpers only pick routes. They do not reserve resources, mutate the
network, or start protocol work.

Custom Metrics
--------------

Use ``best_planned_route`` when you already have candidate routes:

.. code-block:: python

   def total_length(route):
       return sum(
           network.get_link(link_id).transport.length_m
           for link_id in route.link_ids
       )

   best = best_planned_route(candidates, total_length)

The metric must return a finite non-negative score. Lower is better. If two
routes receive the same score, their original candidate order is preserved.

Metrics should be read-only. If a route policy needs current memory state,
inspect that state while scoring, then let resource code perform any reservation
after a route is chosen.

Generate And Rank
-----------------

Use ``best_candidate_route`` when you want to generate candidates and score
them in one call:

.. code-block:: python

   best = best_candidate_route(
       RoutePlanner(NetworkTopology(network)),
       "alice",
       "bob",
       port_kind=PortKind.QUANTUM,
       max_hops=4,
       metric=total_length,
   )

This uses the planner's deterministic candidate order before ranking. Ties keep
that order.

Additive Link Costs
-------------------

For delay, length, or any other additive per-link value on existing candidates,
build a mapping from link ID to cost:

.. code-block:: python

   length_by_link = {
       link_id: link.transport.length_m
       for link_id, link in network.quantum_links.items()
   }

   best = best_route_by_link_cost(candidates, length_by_link)

Use the optional ``default`` argument only when a missing link ID should have a
known fallback cost. Otherwise, provide a cost for every link that may appear in
the candidate routes.

Stateful Route Policies
-----------------------

Candidate ranking is the right fit for policies such as "has enough free
memories", "leaves the largest memory margin", or "avoids a busy repeater".
Those scores depend on the whole route and on current resource state, so they
are better handled by generating candidates, filtering infeasible routes, and
ranking what remains.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/network/planning.py``

.. automodule:: simyuj.network.planning
   :members:
   :show-inheritance:
