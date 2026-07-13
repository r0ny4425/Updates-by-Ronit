Detectors
=========

Detectors convert qstate-backed signals or memory/register subsystems into
classical ``DetectionReport`` objects.

Component classes own event handling, ports, binding, RNG streams, and report
history. Primitive modules define local measurement, timing, click, and readout
models without owning timeline flow.

Choosing A Detector
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 44 28

   * - Component
     - Use when
     - Output
   * - ``DetectorArray``
     - A quantum signal arrives through a detector ingress port.
     - Threshold-style ``DetectionReport`` with raw click details.
   * - ``BellStateAnalyzer``
     - Two quantum inputs must be buffered and resolved as a Bell measurement.
     - Bell-analysis ``DetectionReport`` with match or timeout status.
   * - ``QubitReadoutDevice``
     - A memory/register subsystem should be measured directly by request.
     - ``DetectionReport`` carrying the requested readout outcome.

Components And Primitives
-------------------------

Use detector components when building a simulation topology. They are
timeline-facing objects and own ports, event actions, binding, scheduling, and
report storage.

Use detector primitives when you need the reusable local pieces: detector
parameters, gates, timing windows, readout layouts, click models, dark counts,
measurement specs, result labels, and report records.

Basic Flow
----------

For signal-driven detectors, the component receives a ``PortDelivery`` event,
resolves the qstate target, applies the configured measurement/readout model,
and stores or emits a ``DetectionReport``.

For ``QubitReadoutDevice``, the request names the qstate subsystem directly
instead of arriving through a quantum ingress port.

Timing And Replay
-----------------

Detector timing and random choices are event-driven. Components declare their
RNG streams during binding and use timeline-owned streams for measurement
choices, qstate sampling, click resolution, dark counts, jitter, and related
stochastic effects.

With the same seed, configuration, and event sequence, detector reports should
replay deterministically.

Use the module pages when you need constructor arguments, event action names,
report fields, primitive helper behavior, or exact assumptions for one detector
class.

Module Pages
------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Single Photon Detector <detectors/single_photon>
   Detector Array <detectors/detector_array>
   Bell State Analyzer <detectors/bell_analyzer>
   Qubit Readout <detectors/qubit_readout>
   Detector Primitives <detectors/primitives/index>
