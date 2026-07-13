Resources
=========

``simyuj.resources`` is the bookkeeping layer for memory ownership. It answers
questions like:

* which memory positions exist on each node?
* which slots are free?
* which protocol or workflow is currently holding them?
* what resource-layer state should control code see?

Resources track memory ownership and availability. Components and control
agents still perform physical memory operations through timeline events.
Availability queries are time-aware: callers pass the current tick so the
manager can exclude free slots that are still recovering.

A Small Reservation
-------------------

Start with a manager, register memory slots, reserve one, then mirror the
physical lifecycle when your protocol actually uses the memory:

.. code-block:: pycon

   >>> from simyuj.resources import MemorySlotState, ResourceManager

   >>> manager = ResourceManager()
   >>> refs = manager.register_memory("alice", "qmem", num_positions=2)

   >>> reservation = manager.reserve_memories(
   ...     10,
   ...     {"alice": 1},
   ...     owner="session:alice-bob",
   ... )

   >>> reservation.memory_refs == (refs[0],)
   True
   >>> manager.get_slot(refs[0]).state is MemorySlotState.RESERVED
   True

   >>> _ = manager.commit_reservation(reservation.reservation_id, owner="session:alice-bob")

   >>> _ = manager.mark_occupied(refs[0])

   >>> _ = manager.release_reservation(reservation.reservation_id, owner="session:alice-bob")
   >>> manager.get_slot(refs[0]).state is MemorySlotState.OCCUPIED
   True

The important split is this: reservations describe ownership, while slot state
describes the resource manager's view of memory availability.

The Pieces
----------

.. list-table::
   :header-rows: 1

   * - Object
     - Role
   * - ``MemoryRef``
     - Stable address: ``node_id + device_id + position``.
   * - ``MemorySlotView``
     - Read current resource-layer state.
   * - ``Reservation``
     - Record ownership of memory refs or link IDs.
   * - ``ResourceManager``
     - Register slots, reserve slots, track holders and lifecycle state.
   * - Route helpers
     - Convert a route into caller-supplied per-node memory demand.

Memory Addresses
----------------

``MemoryRef`` is the resource-layer address for one memory position:
``node_id + device_id + position``.

``device_id`` is the name used for the memory device inside the node. When a
manager is built from a ``Network``, this is the node-local device name, not
necessarily the underlying ``QuantumMemory.memory_id``. The physical
``memory_id`` is kept as metadata so callers can still trace where a slot came
from.

Slot State
----------

``MemorySlotState`` is bookkeeping state, separate from the physical
``MemoryPositionStatus`` used by ``QuantumMemory``.

.. list-table::
   :header-rows: 1

   * - Slot State
     - Meaning
   * - ``FREE``
     - Slot is available for reservation.
   * - ``RESERVED``
     - Slot is held by an active or committed reservation.
   * - ``OCCUPIED``
     - Slot is physically in use according to the resource manager view.
   * - ``CONSUMED``
     - Occupied contents were consumed.
   * - ``EXPIRED``
     - Reserved or occupied contents expired.
   * - ``FAILED``
     - Slot failed from any prior state.

Reservation State
-----------------

``Reservation`` records caller ownership of exact memory refs, optional link
IDs, timing metadata, and trace metadata. Its state says what happened to the
ownership record, not what happened inside the physical memory.

.. list-table::
   :header-rows: 1

   * - Reservation State
     - Meaning
   * - ``ACTIVE``
     - Reservation can still be committed or closed.
   * - ``COMMITTED``
     - Reservation was handed off to runtime or protocol code.
   * - ``RELEASED``
     - Reservation was closed normally.
   * - ``EXPIRED``
     - Reservation was closed because it expired.
   * - ``CANCELLED``
     - Reservation was closed by cancellation.

Committing a reservation does not mark memory as occupied. Release, cancel, and
expire remove reservation holders. Slots that are still ``RESERVED`` become
``FREE``; slots already marked ``OCCUPIED``, ``CONSUMED``, ``EXPIRED``, or
``FAILED`` keep that resource-layer state. Commit, release, and cancel require
the caller owner to match the reservation owner, otherwise ``UnauthorizedError``
is raised.

Route Requirements
------------------

Route helpers convert a generic ``Route`` into per-node memory requirements
using a caller-supplied ``node_requirements`` function. The function can return
an integer count for any memory at a node, or a mapping of ``device_id`` to
counts for targeted reservations. The caller decides whether the memory is for
endpoints, repeaters, purification, buffering, or another workflow.

.. code-block:: python

   from simyuj.resources.route_requirements import reserve_route_memories

   reservation = reserve_route_memories(
       10,
       manager,
       route,
       node_requirements=lambda node, idx, length: (
           {"qmem_a": 2, "qmem_b": 1} if node == "relay" else 1
       ),
       owner="swap-session",
   )

The result is a normal ``ResourceManager`` reservation. Link reservation,
protocol choice, memory operations, and entanglement generation stay with the
caller. Resource link filtering is explicit metadata filtering: register slots
with metadata such as ``("link_id", "link:a-b")`` and pass that same
``link_id`` to ``available_memories``.

Usage Rules
-----------

* ``device_id`` is the node-local device name used in resource refs. It may
  differ from ``QuantumMemory.memory_id``.
* ``available_memories(now, ...)`` returns free, unheld slots whose
  ``ready_at`` is ``None`` or not later than ``now``.
* ``commit_reservation`` marks ownership handoff. Call ``mark_occupied`` when
  memory is actually in use.
* ``release_reservation`` removes the holder. Occupied memory keeps its
  resource-layer state until the caller reports a later lifecycle change.
* Route helpers only turn a route into memory counts. Protocol policy stays
  with the caller.
* Resource records store ownership and lifecycle metadata, not qstate objects,
  photons, or component-owned mutable state.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Memory Records <memory>
   Reservations <reservation>
   Route Requirements <route_requirements>
   Resource Manager <manager>
