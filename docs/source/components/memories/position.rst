Positions
=========

``position.py`` defines the classical state snapshots used by
``QuantumMemory`` to track physical memory slots. A position record is not a
quantum state object. It records which slot is empty, busy, occupied, or
recovering, and which qstate subsystem label belongs to the slot while occupied.

Position records are snapshots exposed through memory state, reports, and
tests. Protocol code schedules memory requests rather than constructing
position records directly.

Position Lifecycle
------------------

Each physical position has a ``MemoryPositionStatus``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Status
     - Meaning
   * - ``EMPTY``
     - The slot holds no quantum carrier. It can absorb only when the current
       tick is at or after ``ready_at``.
   * - ``ABSORBING``
     - A delayed absorb is pending. The incoming signal is stored temporarily,
       but no memory subsystem is owned yet.
   * - ``OCCUPIED``
     - The slot owns a stable memory subsystem and can be emitted, measured,
       operated on, discarded, or expired.
   * - ``EMITTING``
     - A delayed emit is pending. The slot still owns the memory subsystem until
       completion succeeds or fails.
   * - ``MEASURING``
     - A delayed measurement is pending. The slot remains reserved until
       measurement completion decides whether it is cleared.
   * - ``APPLYING_OPERATOR``
     - A delayed operator application is pending. The slot remains reserved
       until completion returns it to ``OCCUPIED``.

``EMPTY`` is the only available state. Busy states reserve the position so other
operations cannot reuse the same physical slot while delayed work is pending.

Status-Owned Fields
-------------------

The record validates which fields belong to each lifecycle state:

* ``EMPTY`` owns no ``memory_subsystem``, ``stored_signal``, storage timing, or
  expiry time.
* ``ABSORBING`` keeps the pending ``stored_signal`` but owns no memory subsystem
  yet.
* ``OCCUPIED`` and busy stored states require ``memory_subsystem``,
  ``stored_time``, and ``last_noise_update_time``.

The record validates field ownership and basic non-negative timing values.
Timeline operation handlers own transition timing.

Subsystem Labels And Tokens
---------------------------

The physical position index is stable. While a position is occupied, it owns a
stable memory subsystem label:

.. code-block:: text

   memory:<memory_id>:position:<position>

If the same physical position is reused later, it reuses the same memory
subsystem label. ``occupancy_token`` distinguishes the current occupancy from
stale delayed completions or stale expiry events.

When emission succeeds, the memory subsystem is relabelled to a unique emitted
photon subsystem:

.. code-block:: text

   photon:<memory_id>:position:<position>:emit:<emission_counter>

This relabeling lets reports and emitted signals identify where a photon came
from without keeping the memory position occupied.

Recovery Timing
---------------

``ready_at`` controls automatic recovery after a quantum carrier is removed. An
empty position is available only when both conditions hold:

.. code-block:: python

   record.status is MemoryPositionStatus.EMPTY
   current_time >= record.ready_at

If a protocol requests a specific position before ``ready_at``,
``QuantumMemory`` raises an error. If the protocol lets the memory choose a
position automatically, recovering positions are skipped until ready.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/memories/position.py``

.. automodule:: simyuj.components.memories.position
   :members: MemoryPositionStatus, MemoryPositionRecord, memory_subsystem_id, emitted_photon_subsystem_id
   :show-inheritance:
