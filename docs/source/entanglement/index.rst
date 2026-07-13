Entanglement
============

The ``simyuj.entanglement`` package stores optional bookkeeping records for
known memory-backed entangled pairs. Pair generation, qstate mutation,
swapping, purification, memory reservation, and memory operations are handled
by component, qstate, resource, and control code.

Use the registry after a component or controller has decided that two memory
positions should be exposed as a reusable pair. The registry tracks endpoints,
lifecycle state, fidelity, timing, metadata, and route selection. It is not a
global entanglement oracle and it does not track flying entanglement.

A Small Lifecycle
-----------------

A pair record is immutable. The registry stores the current version of the
record and replaces it when lifecycle operations happen.

.. code-block:: pycon

   >>> from simyuj.entanglement import EntangledPairRecord, EntangledPairRegistry
   >>> from simyuj.resources import MemoryRef


   >>> registry = EntangledPairRegistry()

   >>> pair = EntangledPairRecord(
   ...     pair_id="pair:0",
   ...     left=MemoryRef("alice", "qmem", 0),
   ...     right=MemoryRef("bob", "qmem", 0),
   ...     fidelity=0.94,
   ...     created_at=10,
   ...     expires_at=20,
   ... )

   >>> _ = registry.register(pair)

   >>> reserved = registry.reserve("pair:0")
   >>> consumed = registry.consume("pair:0")

   >>> reserved.pair_id
   'pair:0'
   >>> consumed.is_terminal
   True
   >>> registry.get("pair:0") is consumed
   True

The original ``pair`` object is still the old immutable record. Use
``registry.get(pair_id)`` when you need the current lifecycle state.

Records And Registry
--------------------

.. list-table::
   :header-rows: 1

   * - Piece
     - Role
   * - ``EntangledPairRecord``
     - Immutable description of one pair: ID, endpoint memories, state,
       fidelity, timing, generation link, and metadata.
   * - ``EntangledPairRegistry``
     - Owns the current record for each pair ID and enforces lifecycle rules.
   * - ``simyuj.entanglement.queries``
     - Builds read-only candidate views for nodes, routes, and memory refs.

Pair endpoints are stored as ``left`` and ``right`` labels, but connection
checks are undirected. A pair from Alice to Bob also satisfies a Bob-to-Alice
query.

``EntangledPairRecord`` stores resource-layer references, not qstate objects,
subsystem identifiers, Pauli frames, detector reports, or protocol correction
state.

Optional occupancy tokens (``left_occupancy_token``, ``right_occupancy_token``)
distinguish the specific memory occupancy a pair refers to. Without tokens, a
historical pair and an active pair sharing the same ``MemoryRef`` look
identical. Protocol-specific details such as Bell-state outcomes, source
report IDs, or correction conventions should stay in workflow code or be stored
as plain metadata when useful for traceability.

Lifecycle States
----------------

.. list-table::
   :header-rows: 1

   * - State
     - Meaning
   * - ``AVAILABLE``
     - The pair can be selected by query helpers and reserved.
   * - ``RESERVED``
     - The pair is active but already claimed by a controller flow.
   * - ``CONSUMED``
     - The pair was used and is no longer active.
   * - ``EXPIRED``
     - The pair passed its usable lifetime.
   * - ``FAILED``
     - The pair is no longer usable after a failed operation or link condition.

Active records are ``AVAILABLE`` or ``RESERVED``. Terminal records are kept for
history but do not block the same memory positions from being reused by later
active pairs.

Lifecycle operations enforce valid transitions:

.. list-table::
   :header-rows: 1

   * - Operation
     - Allowed source state
     - Result state
   * - ``reserve``
     - ``AVAILABLE``
     - ``RESERVED``
   * - ``release``
     - ``RESERVED``
     - ``AVAILABLE``
   * - ``consume``
     - ``AVAILABLE`` or ``RESERVED``
     - ``CONSUMED``
   * - ``expire``
     - ``AVAILABLE`` or ``RESERVED``
     - ``EXPIRED``
   * - ``fail``
     - ``AVAILABLE`` or ``RESERVED``
     - ``FAILED``

``expire_before(now)`` expires active pairs whose ``expires_at`` tick is less
than or equal to ``now``.

Queries
-------

Query helpers do not mutate the registry. They return candidates.

``available_pairs_for_route_hops()`` returns available pairs per route hop. It
does not reserve anything, choose one pair per hop, or guarantee a globally
consistent assignment. Controllers should reserve the chosen pairs explicitly.

``pairs_by_node_pair()`` can include historical terminal records. Pass a
``state`` filter when you only want a specific lifecycle state.

Fidelity filters are strict in one important way: if ``min_fidelity`` is given,
pairs without a fidelity estimate are excluded.

Registry Notes
--------------

``record.reserved()`` only returns a replacement record. Use
``registry.reserve(pair_id)`` when you want lifecycle rules enforced.

``registry.pairs`` is a read-only live mapping. Use ``all_pairs()`` or a query
helper when deterministic tuple output matters.

Query helpers expose candidates. Controller code still owns selection,
reservation, and rollback if later work fails.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Pair Records <pair>
   Registry <registry>
   Queries <queries>
