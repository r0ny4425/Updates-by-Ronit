Memories
========

Memory components store qstate-backed carriers in physical positions and
expose explicit event actions for absorb, emit, operator, measurement,
discard, expiry, and metadata updates. ``QuantumMemory`` owns classical
position lifecycle records, ports, notices, and reports while qstate math stays
inside the timeline qstate manager.

Each occupied position uses a stable memory subsystem label. Absorb relabels
an incoming signal target into that memory subsystem; emit relabels it back
into an outgoing photon subsystem. Storage noise is applied lazily before
operations that touch occupied positions.

Use the detailed pages by responsibility:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Need
     - Page
   * - Event actions and lifecycle
     - :doc:`memories/quantum_memory`
   * - Position state
     - :doc:`memories/position`
   * - Request payloads
     - :doc:`memories/requests`
   * - Report payloads
     - :doc:`memories/reports`
   * - Storage noise configuration
     - :doc:`memories/noise`
   * - Timeline log records
     - :doc:`memories/reporting`

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Quantum Memory <memories/quantum_memory>
   Positions <memories/position>
   Requests <memories/requests>
   Reports <memories/reports>
   Noise <memories/noise>
   Reporting <memories/reporting>
