.. _memory-reports:

Memory Reports
==============

Memory reports are immutable outcome payloads produced by ``QuantumMemory``
after memory operations have been handled. They describe results after qstate
and position-record changes have already happened.

Reports are observation and protocol-bookkeeping payloads.

Where Reports Appear
--------------------

Produced reports are appended to ``memory.reports`` and logged. If the
classical ``notice`` port is connected, the same report object is transmitted
through that port at the current timeline tick.

``MemoryOperatorReport`` is the main exception: operator reports are produced
only when the memory ``notice`` port is connected. Without a notice connection,
a successful operator application can update qstate without appending an
operator report to ``memory.reports``.

``success`` is the main boolean branch for protocol logic. ``status`` is a
compact operation-specific string for debugging and finer-grained detail.

Report Types
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Report type
     - Produced for
   * - ``MemoryAbsorbReport``
     - Absorbing an incoming photon signal into one memory position.
   * - ``MemoryEmitReport``
     - Emitting one occupied memory position as an outgoing photon signal.
   * - ``MemoryOperatorReport``
     - Applying an operator to ordered occupied positions, when ``notice`` is
       connected.
   * - ``MemoryMeasurementReport``
     - Measuring ordered occupied positions through the detector readout
       primitive.
   * - ``MemoryDiscardReport``
     - Explicitly discarding one occupied memory position.
   * - ``MemoryExpireReport``
     - Expiring one occupied position when its occupancy token still matches.
   * - ``MemoryMetaUpdateReport``
     - Updating classical metadata for one occupied position.

Operation Notes
---------------

Absorb reports describe either a stored photon target or a failed absorption.
On successful absorb, ``memory_subsystem`` is the stable memory-position qstate
label. On failed absorb, the incoming photon target has already been discarded.

Emit, discard, and expire reports describe operations after the memory position
has already been cleared into recovery. For emit, the previous memory subsystem
has already been relabelled to the emitted-photon subsystem and the outgoing
``Signal`` has already been transmitted.

Operator reports preserve the requested position order in both ``positions`` and
``memory_subsystems``.

Measurement reports wrap the detector readout primitive's ``DetectionReport``.
For destructive measurement, use ``cleared_positions`` rather than ``success``
to determine which memory positions were removed.

Metadata update reports can describe success or failure. ``occupancy_token`` is
the current position token at report time, not necessarily the caller's expected
token.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/memories/reports.py``

.. automodule:: simyuj.components.memories.reports
   :members: MemoryAbsorbReport, MemoryEmitReport, MemoryOperatorReport, MemoryMeasurementReport, MemoryDiscardReport, MemoryExpireReport, MemoryMetaUpdateReport, MemoryReport
   :show-inheritance:
