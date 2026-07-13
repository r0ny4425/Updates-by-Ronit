Reports
=======

Detector report primitives are immutable payload records for completed detector
work.

There are two levels:

``RawClick``
   One low-level detector-channel firing after window evaluation, jitter, and
   dead-time filtering.

``DetectionReport``
   The resolved user-facing detector outcome produced from raw clicks,
   measurement metadata, or explicit readout.

Raw Clicks
----------

``RawClick`` records which detector fired, when it fired, why it fired, and
which logical outcome label the readout layer associated with that channel.

Click resolvers may sort, select, or aggregate raw clicks. The click record
itself is descriptive.

Detection Reports
-----------------

``DetectionReport`` is the detector result consumed by components, protocol
logic, tests, and tracing. Components store reports locally and may emit the
same report through a classical output port.

``success`` means the report has a usable logical outcome. It does not simply
mean that a physical detector clicked. For example, a failed double-click report
can carry raw clicks, while a no-click report carries none.

Common Flags
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Meaning
   * - ``FLAG_SIGNAL_CLICK``
     - Raw click came from signal-detection efficiency.
   * - ``FLAG_DARK_COUNT``
     - Raw click came from the dark-count model.
   * - ``FLAG_AFTERPULSE``
     - Raw click came from the afterpulse model.
   * - ``FLAG_NO_CLICK``
     - No raw click was available for report resolution.
   * - ``FLAG_DOUBLE_CLICK``
     - More than one raw click affected threshold-style resolution.
   * - ``FLAG_OUTSIDE_GATE``
     - Input arrived while the detector gate was closed.
   * - ``FLAG_TIMEOUT``
     - A detector component timed out while waiting for a matching input.
   * - ``FLAG_NO_OUTCOME``
     - Measurement or readout completed without a logical outcome.

Metadata
--------

Reports may include qstate measurement metadata, random measurement selection
metadata, signal ids, and component-specific metadata.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/reports.py``

.. automodule:: simyuj.components.detectors.primitives.reports
   :members: RawClick, DetectionReport, FLAG_OUTSIDE_GATE, FLAG_DEAD_TIME_BLOCKED, FLAG_NO_CLICK, FLAG_NO_OUTCOME, FLAG_DOUBLE_CLICK, FLAG_DARK_COUNT, FLAG_SIGNAL_CLICK, FLAG_AFTERPULSE, FLAG_TIMEOUT, FLAG_INVALID_PAYLOAD
   :show-inheritance:
