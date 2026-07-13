.. _detector-array:

Detector Array
==============

``DetectorArray`` is the signal-facing detector component for qstate-backed
quantum signals. It receives an incoming ``Signal`` through a quantum input
port, chooses and executes a measurement, maps the qstate result to detector
channel exposures, evaluates the configured detector channels, and stores or
emits a ``DetectionReport``.

Use it when a simulated optical or quantum signal reaches a receiver and
should become a classical detection result. The array evaluates an ordered
tuple of ``SinglePhotonDetector`` channels.

At a high level, one detection operation follows this path:

.. code-block:: text

   Signal
      -> resolve qstate targets
      -> choose and execute measurement
      -> map qstate result to detector exposures
      -> evaluate detector channels
      -> resolve raw clicks into DetectionReport
      -> store locally and optionally emit on classical output

Receiver Examples
-----------------

Typical receiver shapes include:

* a one-channel presence detector;
* a two-detector threshold receiver;
* a basis-selecting polarization receiver;
* a three-channel or four-channel detector bank;
* a receiver with channel imperfections such as efficiency loss, dark counts,
  jitter, dead time, or afterpulsing;
* a receiver that stores reports locally and optionally sends them through a
  classical output port.

Use ``SinglePhotonDetector`` directly only when you need the lower-level
channel model without ports, qstate, or scheduled events.

Concepts
--------

``DetectorArray`` coordinates four separate concerns:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Concept
     - Where it happens
     - Meaning
   * - Measurement
     - qstate layer
     - Chooses the quantum operation and returns a qstate result.
   * - Readout mapping
     - detector readout layer
     - Maps the qstate result label to detector-channel exposure records.
   * - Detector physics
     - each ``SinglePhotonDetector``
     - Samples signal clicks, dark counts, jitter, dead time, and afterpulses.
   * - Click resolution
     - click resolver
     - Converts low-level ``RawClick`` records into one logical report.

This means a qstate measurement result is not the same thing as a detector
click. A measurement can return ``"0"`` while the corresponding detector fails
to click because of efficiency, gate timing, or dead time. An unexposed
channel can also dark-count and produce a competing click.

Ports And Event Flow
--------------------

``DetectorArray`` is a timeline component. It accepts only
``ACTION_DETECT_SIGNAL`` events. The event payload must be a ``PortDelivery``
addressed to the array's quantum input port, and the delivery payload must be a
``Signal``.

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Surface
     - Name
     - Meaning
   * - Quantum input port
     - ``in``
     - Receives ``PortDelivery`` objects carrying ``Signal`` payloads.
   * - Classical output port
     - ``out``
     - Emits ``DetectionReport`` objects when connected.
   * - Stored reports
     - ``reports``
     - Local report history, kept whether or not ``out`` is connected.

The detector array does not decide higher-level protocol behavior. Protocol
code should schedule incoming signals, inspect reports, and decide whether a
result should be accepted, discarded, or used by the protocol.

Minimal Two-Detector Receiver
-----------------------------

This is the common threshold receiver: measure in the Z basis, expose ``d0``
for result ``"0"`` and ``d1`` for result ``"1"``.

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.click import ThresholdClickResolver
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams(
       efficiency=0.9,
       dark_count_rate_hz=100.0,
       dead_time_ticks=50,
       jitter_stddev_ticks=2.0,
   )

   receiver = DetectorArray(
       device_id="rx",
       detectors=(
           SinglePhotonDetector("d0", params=params),
           SinglePhotonDetector("d1", params=params),
       ),
       measurement="z",
       readout={
           "0": "d0",
           "1": "d1",
       },
       detection_window_ticks=10,
       click_resolver=ThresholdClickResolver(double_click_policy="fail"),
       consume_signal=True,
   )

Detector order is part of the readout contract, and detector ids must be
unique.

Measurement
-----------

``measurement`` is converted through ``Measure.from_spec``. A basis name such
as ``"z"`` is the simplest form:

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams()

   receiver = DetectorArray(
       device_id="rx_z",
       detectors=(
           SinglePhotonDetector("d0", params=params),
           SinglePhotonDetector("d1", params=params),
       ),
       measurement="z",
       readout={"0": "d0", "1": "d1"},
   )

For more control, pass a ``Measure`` object. Common specs include
``Measure.basis(...)``, ``Measure.povm(...)``, ``Measure.random(...)``, and
``Measure.by_meta(...)``. See :doc:`primitives/measurement` for target specs,
metadata-selected measurements, and random selection details.

Basis-Aware Example
~~~~~~~~~~~~~~~~~~~

When the selected measurement changes the meaning of result labels, use a
nested readout map keyed by measurement label:

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.click import ThresholdClickResolver
   from simyuj.components.detectors.primitives.measurement import Measure
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams()

   receiver = DetectorArray(
       device_id="basis_rx",
       detectors=(
           SinglePhotonDetector("z0", params=params),
           SinglePhotonDetector("z1", params=params),
           SinglePhotonDetector("x_plus", params=params),
           SinglePhotonDetector("x_minus", params=params),
       ),
       measurement=Measure.by_meta(
           "basis",
           mapping={
               "Z": Measure.basis("z"),
               "X": Measure.basis("x"),
           },
       ),
       readout={
           "z": {
               "0": "z0",
               "1": "z1",
           },
           "x": {
               "+": "x_plus",
               "-": "x_minus",
           },
       },
       detection_window_ticks=10,
       click_resolver=ThresholdClickResolver(double_click_policy="fail"),
   )

The outer readout keys are measurement labels such as ``"z"`` and ``"x"``.
They are not detector ids.

Readout Mapping
---------------

``readout`` maps qstate result labels to detector-channel exposures. For most
arrays, a dictionary is enough.

Flat Outcome Map
~~~~~~~~~~~~~~~~

.. code-block:: python

   readout = {
       "0": "d0",
       "1": "d1",
   }

If the qstate result label is ``"0"``, detector ``d0`` is exposed to the
signal. Detector ``d1`` is still evaluated as an unexposed channel, so it can
still dark-count.

Nested Measurement Map
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   readout = {
       "z": {
           "0": "z0",
           "1": "z1",
       },
       "x": {
           "+": "x_plus",
           "-": "x_minus",
       },
   }

The outer key is ``MeasurementCall.label``. The inner key is the qstate result
label.

Single-Detector Readout
~~~~~~~~~~~~~~~~~~~~~~~

A one-channel array may omit ``readout``:

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams()

   receiver = DetectorArray(
       device_id="presence_rx",
       detectors=(SinglePhotonDetector("d0", params=params),),
       measurement="z",
       readout=None,
   )

Arrays with more than one detector must provide a readout mapping or custom
readout layout. For custom callables, time offsets, and N-channel recipes, see
:doc:`primitives/readout`.

Custom Detector Banks
---------------------

``DetectorArray`` can represent custom detector banks by changing:

* the number of ``SinglePhotonDetector`` channels;
* detector ids and order;
* the measurement selection policy;
* the readout layout;
* the gate model;
* the click resolver;
* detector-channel physical parameters.

It cannot replace ``SinglePhotonDetector`` with arbitrary detector objects.
When building larger banks, the key contract is simple:

.. code-block:: text

   measurement result labels == readout keys
   readout values             == detector ids

For example, if a custom measurement returns ``"r2"``, the readout map should
map ``"r2"`` to the detector id that should be exposed to the signal.

Programmatic N-Channel Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detector banks with many channels, generate detector ids and the readout
map together so labels and detector ids stay aligned:

.. code-block:: python

   n_outputs = 4
   detector_ids = tuple(f"d{i}" for i in range(n_outputs))

   detectors = tuple(
       SinglePhotonDetector(detector_id, params=params)
       for detector_id in detector_ids
   )

   readout = {
       f"r{i}": detector_id
       for i, detector_id in enumerate(detector_ids)
   }

   receiver = DetectorArray(
       device_id="n_channel_rx",
       detectors=detectors,
       measurement=custom_measurement,
       readout=readout,
       detection_window_ticks=10,
   )

If ``custom_measurement`` returns ``"r2"``, detector ``d2`` is exposed to the
signal while the other channels are evaluated as unexposed channels.

Detector Channels
-----------------

Each channel is a ``SinglePhotonDetector`` with its own local state. Reusing
the same channel instance across detections is part of the physical model:
dead time and afterpulsing depend on previous accepted clicks.

The array-level behavior to remember is:

* unexposed channels are still evaluated, so they can dark-count;
* channel state persists across detector-array events;
* accepted raw clicks are passed to the configured click resolver.

For the channel physics model, including efficiency, dark counts, jitter, dead
time, afterpulsing, and photon-number-resolving mode, see
:doc:`single_photon`.

Gates And Detection Windows
---------------------------

The detector array evaluates each channel only inside active gate time.
``detection_window_ticks`` is the requested observation length. The actual
window can be shorter when a gate closes before the requested window ends.

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.gate import PeriodicGate
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams()

   gated_receiver = DetectorArray(
       device_id="gated_rx",
       detectors=(
           SinglePhotonDetector("d0", params=params),
           SinglePhotonDetector("d1", params=params),
       ),
       measurement="z",
       readout={"0": "d0", "1": "d1"},
       gate_model=PeriodicGate(
           period_ticks=100,
           open_duration_ticks=10,
           first_open_tick=0,
       ),
       detection_window_ticks=20,
   )

In this example, a signal arriving at tick ``5`` receives only the active part
of the requested 20-tick detection window, because the gate closes at tick
``10``. See :doc:`primitives/gate` for the gate model details.

If a signal arrives while the gate is closed:

* no qstate measurement is run;
* a failed report is stored with ``FLAG_OUTSIDE_GATE``;
* if ``consume_signal=True``, the signal qstate targets are discarded.

Click Resolution
----------------

Detector-channel evaluation produces zero or more ``RawClick`` records. The
click resolver converts those raw clicks into a single ``DetectionReport``.

``ThresholdClickResolver`` is the usual resolver for detector arrays. It uses
``RawClick.outcome_label`` as the logical report outcome:

.. code-block:: python

   from simyuj.components.detectors.detector_array import DetectorArray
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.click import ThresholdClickResolver
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams()

   receiver = DetectorArray(
       device_id="rx",
       detectors=(
           SinglePhotonDetector("d0", params=params),
           SinglePhotonDetector("d1", params=params),
       ),
       measurement="z",
       readout={"0": "d0", "1": "d1"},
       click_resolver=ThresholdClickResolver(double_click_policy="fail"),
   )

The double-click policy decides what happens when more than one raw click is
available for the same report:

.. list-table::
   :header-rows: 1
   :widths: 24 48 28

   * - Policy
     - Behavior
     - Typical use
   * - ``"fail"``
     - Produce an unsuccessful report with ``FLAG_DOUBLE_CLICK``.
     - Conservative protocol filtering.
   * - ``"first"``
     - Choose the earliest sorted click.
     - Earliest detector firing wins.
   * - ``"random"``
     - Choose uniformly from the sorted raw clicks.
     - Random double-click assignment.

See :doc:`primitives/click` for ``POVMLabelClickResolver`` and exact click
pattern helpers.

Reports And Output
------------------

Every handled detection appends one ``DetectionReport`` to
``receiver.reports``. If the classical output port is connected, the same
report object is also transmitted through ``receiver.output_port``.

.. code-block:: python

   report = receiver.reports[-1]

   if report.success:
       print("outcome:", report.outcome)
   else:
       print("failed detection:", report.flags)

A report can be unsuccessful and still contain useful information. A failed
double-click report can carry the raw clicks that caused the failure. A
no-click report can still carry a qstate measurement result.

``output_latency_ticks`` adds delay before a report is emitted on the
classical output port. If a report has raw clicks, the ready time is based on
the latest raw-click time. If there are no raw clicks, the ready time falls
back to the active detection-window completion time.

See :doc:`primitives/reports` for report fields and flag meanings.

Output Port Wiring
~~~~~~~~~~~~~~~~~~

Connect ``receiver.output_port`` when another component should receive reports
as events. The target action belongs to the receiving component:

.. code-block:: python

   connect_ports(
       receiver.output_port,
       report_sink.input_port,
       target_action="receive_report",
   )

If the output port is not connected, reports are still stored in
``receiver.reports``.

Signal Consumption
------------------

``consume_signal`` controls qstate lifetime after detection.

``consume_signal=True``
   The signal qstate targets are discarded after measurement. If the gate is
   closed, the targets are also discarded when this option is true.

``consume_signal=False``
   The signal qstate targets remain available after detector handling.

This is separate from measurement collapse. ``collapse`` controls whether the
qstate measurement collapses the measured subsystem. ``consume_signal``
controls whether the measured targets are discarded from the qstate manager
afterward.

Deterministic Binding
---------------------

Call ``bind(context)`` before timeline event execution. Binding declares
timeline-owned RNG streams for measurement selection, qstate measurement, click
resolution, and per-channel detector sampling. With the same timeline seed,
detector configuration, and event sequence, detector-array behavior is
reproducible.

Binding is idempotent for the same timeline and rejects rebinding to a
different timeline.

Setup Requirements
------------------

* each detector id is unique;
* detector order matches the readout contract;
* arrays with more than one detector provide ``readout`` entries for expected
  measurement result labels;
* nested readout maps use ``MeasurementCall.label`` as their outer keys;
* ``detection_window_ticks`` is positive;
* ``output_latency_ticks`` is non-negative;
* detector channels are ``SinglePhotonDetector`` instances;
* dead-time recovery is checked at the detection-window level;
* protocol code decides whether a report is accepted and whether
  ``consume_signal`` should be enabled.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/detector_array.py``

.. automodule:: simyuj.components.detectors.detector_array
   :members: DetectorArray
   :show-inheritance:
