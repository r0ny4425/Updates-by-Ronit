.. _qubit-readout:

Qubit Readout
=============

``QubitReadoutDevice`` is an event-driven readout component for quantum state
subsystems that are already known to the simulation. A controller schedules an
explicit ``QubitReadoutJob`` that names the qstate subsystems to read out.

Use it when the simulated device has direct access to memory qubits and you
want a detector-style ``DetectionReport`` from a qstate measurement. Common
jobs include single-qubit Z or X readout, Bell-basis readout of stored
subsystems, and classical confusion-map readout errors.

Use ``DetectorArray`` for readout triggered by an incoming signal and evaluated
through detector channels. Use ``BellStateAnalyzer`` for coincidence-style Bell
analysis of two arriving quantum signals.

Readout Flow
------------

One readout job does four things:

1. receive an ``ACTION_RUN_QUBIT_READOUT`` event;
2. choose the device default measurement or a per-job measurement override;
3. execute the qstate measurement on the job's explicit targets;
4. apply optional classical readout distortion and store or emit a report.

.. code-block:: text

   QubitReadoutJob
      -> qstate measurement
      -> true qstate result label
      -> optional readout model
      -> reported outcome
      -> DetectionReport

The qstate measurement can collapse the measured subsystem. The readout model
runs after that measurement and can only change the reported classical outcome.
It does not modify the qstate result that was returned by the qstate manager.

Ports And Event Flow
--------------------

``QubitReadoutDevice`` is a timeline component, but it is not a signal receiver.
It has no quantum input port. The protocol or controller schedules jobs directly
against the component.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Surface
     - Name
     - Meaning
   * - Event action
     - ``ACTION_RUN_QUBIT_READOUT``
     - Runs one explicit qstate readout job.
   * - Event payload
     - ``QubitReadoutJob``
     - Names the qstate targets and optional per-job overrides.
   * - Classical output port
     - ``out``
     - Emits ``DetectionReport`` objects when connected.
   * - Stored reports
     - ``reports``
     - Local report history, kept whether or not the output port is connected.

A readout event must target the readout device and carry a ``QubitReadoutJob``
as its payload. The job carries explicit qstate subsystem IDs instead of a
``Signal`` or ``PortDelivery``.

Minimal Z-Basis Readout
-----------------------

This is the usual direct readout setup: the device measures the job target in
the Z basis and reports the true qstate result label.

.. code-block:: python

   from simyuj.components.detectors.qubit_readout import (
       QubitReadoutDevice,
       QubitReadoutJob,
   )
   from simyuj.components.detectors.primitives.actions import (
       ACTION_RUN_QUBIT_READOUT,
   )
   from simyuj.components.detectors.primitives.measurement import Measure
   from simyuj.engine.event import Event
   from simyuj.engine.timeline import Timeline
   from simyuj.qstate import SubsystemId
   from simyuj.runtime.binding import bind_if_supported

   timeline = Timeline(master_seed=123)
   q0 = SubsystemId("q0")
   timeline.qstate.prepare("|0>", subsystems=(q0,))

   readout = QubitReadoutDevice(
       device_id="memory_readout",
       measurement=Measure.basis("z"),
       readout_model=None,
       output_latency_ticks=0,
   )

   bind_if_supported(readout, timeline)

   timeline.schedule(
       Event(
           time=0,
           target_ref=readout,
           action=ACTION_RUN_QUBIT_READOUT,
           payload_ref=QubitReadoutJob(
               job_id="read-q0",
               targets=(q0,),
           ),
           source=None,
           subsystem_id="components",
       )
   )

   timeline.run_until_empty()

After execution, the latest report is available locally:

.. code-block:: python

   report = readout.reports[-1]

   if report.success:
       print(report.outcome)
   else:
       print(report.flags)

``report.qstate_result`` stores the true qstate measurement result. ``report.outcome``
stores the reported classical readout outcome after the readout model has been
applied.

In protocol code, the controller usually owns the scheduling step. The important
contract is the same: schedule ``ACTION_RUN_QUBIT_READOUT`` on the readout
device, and pass a ``QubitReadoutJob`` as ``payload_ref``.

Readout Jobs
------------

A ``QubitReadoutJob`` is the payload for one explicit readout event.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``job_id``
     - Caller-chosen identifier copied into report metadata.
   * - ``targets``
     - Non-empty, unique ``SubsystemId`` objects to measure.
   * - ``measurement``
     - Optional per-job measurement override. ``None`` uses the device default.
   * - ``collapse``
     - Optional per-job collapse override. ``None`` keeps the selected
       measurement's own collapse setting.
   * - ``output_latency_ticks``
     - Optional per-job output-latency override for emitted reports.
   * - ``meta``
     - Extra metadata appended to the resulting report.

``targets`` must contain qstate ``SubsystemId`` objects, not plain string labels.
Use the same subsystem handles that were used when preparing or storing the
qstate data.

Per-Job Measurement Overrides
-----------------------------

Set a default measurement on the device when most jobs should use the same
basis. Override the measurement on individual jobs when a protocol sometimes
needs a different readout.

.. code-block:: python

   from simyuj.qstate import SubsystemId
   from simyuj.components.detectors.qubit_readout import (
       QubitReadoutDevice,
       QubitReadoutJob,
   )
   from simyuj.components.detectors.primitives.measurement import Measure

   q0 = SubsystemId("q0")

   readout = QubitReadoutDevice(
       device_id="adaptive_readout",
       measurement=Measure.basis("z"),
   )

   z_job = QubitReadoutJob(
       job_id="read-q0-z",
       targets=(q0,),
   )

   x_job = QubitReadoutJob(
       job_id="read-q0-x",
       targets=(q0,),
       measurement=Measure.basis("x"),
       collapse=True,
   )

Here ``z_job`` uses the device default. ``x_job`` overrides the measurement for
that one scheduled readout.

No-Outcome Readout
------------------

Use ``Measure.none()`` when a protocol needs to produce a readout report without
asking the qstate manager for a logical measurement outcome. This is mostly
useful for explicit discard paths or controller bookkeeping.

.. code-block:: python

   from simyuj.qstate import SubsystemId
   from simyuj.components.detectors.qubit_readout import QubitReadoutJob
   from simyuj.components.detectors.primitives.measurement import Measure

   q0 = SubsystemId("q0")

   skipped_job = QubitReadoutJob(
       job_id="skip-q0",
       targets=(q0,),
       measurement=Measure.none(),
   )

The resulting report has ``success=False``, ``outcome=None``, empty
``raw_clicks``, and the ``"no_outcome"`` flag. This is different from an
optical detector miss; qubit readout does not model raw detector clicks.

Classical Readout Distortion
----------------------------

A readout model changes the reported classical label after the qstate
measurement result is known. This is useful for modeling classical readout
errors or SPAM-like label confusion.

The default model is identity readout:

.. math::

   P(\hat{y} = y \mid y) = 1

where :math:`y` is the true qstate-result label and :math:`\hat{y}` is the
reported label.

A confusion map defines conditional probabilities:

.. math::

   P(\hat{y} = r \mid y = t) = C_{t,r},
   \qquad \sum_r C_{t,r} = 1

Each row is keyed by the true outcome. Each row maps possible reported outcomes
to probabilities. Rows are not normalized automatically; every configured row
must already sum to one.

Example: Single-Qubit SPAM-Like Readout Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simyuj.components.detectors.qubit_readout import QubitReadoutDevice
   from simyuj.components.detectors.primitives.measurement import Measure

   confusion = {
       "0": {"0": 0.98, "1": 0.02},
       "1": {"0": 0.05, "1": 0.95},
   }

   readout = QubitReadoutDevice(
       device_id="noisy_memory_readout",
       measurement=Measure.basis("z"),
       readout_model=confusion,
       output_latency_ticks=5,
   )

If the true qstate label is ``"0"``, the report outcome is ``"0"`` with
probability ``0.98`` and ``"1"`` with probability ``0.02``. The stored
``qstate_result`` still records the true qstate measurement result.

Callable Readout Models
~~~~~~~~~~~~~~~~~~~~~~~

For small custom classical relabeling, ``readout_model`` may also be a callable.
It receives the true outcome, qstate result, selected measurement call,
measurement context, and readout RNG, then returns the reported outcome.

.. code-block:: python

   from simyuj.components.detectors.qubit_readout import QubitReadoutDevice
   from simyuj.components.detectors.primitives.measurement import Measure

   def append_report_suffix(
       true_outcome,
       qstate_result,
       measurement_call,
       context,
       rng,
   ):
       del qstate_result, measurement_call, context, rng
       if true_outcome is None:
           return None
       return f"{true_outcome}:reported"

   readout = QubitReadoutDevice(
       device_id="custom_memory_readout",
       measurement=Measure.basis("z"),
       readout_model=append_report_suffix,
   )

Bell-State Readout
------------------

``QubitReadoutDevice`` can run a Bell-basis qstate measurement when the job
targets name the stored qstate subsystems to measure. Use ``Measure.bell()``,
not the string ``"bell"``. String measurement specs are interpreted through the
ordinary basis-measurement path.

.. code-block:: python

   from simyuj.qstate import SubsystemId
   from simyuj.components.detectors.qubit_readout import (
       QubitReadoutDevice,
       QubitReadoutJob,
   )
   from simyuj.components.detectors.primitives.measurement import Measure

   left_qubit = SubsystemId("left")
   right_qubit = SubsystemId("right")

   bell_readout = QubitReadoutDevice(
       device_id="bell_memory_readout",
       measurement=Measure.bell(collapse=True),
   )

   bell_job = QubitReadoutJob(
       job_id="bell-readout-0",
       targets=(left_qubit, right_qubit),
   )

The default ``Measure.bell()`` target spec measures all job targets. In a Bell
readout job, pass the two qstate subsystem IDs expected by the qstate manager's
Bell measurement implementation.

The resulting report uses the same report fields as other qubit readouts:

.. code-block:: python

   report = bell_readout.reports[-1]

   print(report.measurement_method)  # "bell"
   print(report.measurement_label)   # "bell" by default
   print(report.qstate_result)       # true Bell-measurement result object
   print(report.outcome)             # reported label after readout model

This is a direct qstate Bell measurement on explicit memory targets. It does
not model two optical inputs, detector coincidences, channel timing, dark
counts, or Bell-analyzer input buffering. Use the signal-facing Bell analyzer
for those models.

Bell Readout With Classical Label Confusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Bell readout can also use a readout model. This models classical label
confusion after the qstate Bell measurement, not an optical Bell-measurement
failure mode.

.. code-block:: python

   from simyuj.components.detectors.qubit_readout import QubitReadoutDevice
   from simyuj.components.detectors.primitives.measurement import Measure

   bell_confusion = {
       "phi_plus": {
           "phi_plus": 0.97,
           "phi_minus": 0.01,
           "psi_plus": 0.01,
           "psi_minus": 0.01,
       },
       "phi_minus": {
           "phi_plus": 0.01,
           "phi_minus": 0.97,
           "psi_plus": 0.01,
           "psi_minus": 0.01,
       },
       "psi_plus": {
           "phi_plus": 0.01,
           "phi_minus": 0.01,
           "psi_plus": 0.97,
           "psi_minus": 0.01,
       },
       "psi_minus": {
           "phi_plus": 0.01,
           "phi_minus": 0.01,
           "psi_plus": 0.01,
           "psi_minus": 0.97,
       },
   }

   bell_readout = QubitReadoutDevice(
       device_id="noisy_bell_memory_readout",
       measurement=Measure.bell(collapse=True),
       readout_model=bell_confusion,
   )

Use labels that match the labels returned by your qstate manager. If a true
label is absent from the confusion map, it passes through unchanged.

Output Port And Latency
-----------------------

Every handled job appends one ``DetectionReport`` to ``reports``. If the
classical output port is connected, the same report is transmitted through
``output_port``.

.. code-block:: python

   connect_ports(
       readout.output_port,
       report_sink.input_port,
       target_action="receive_report",
   )

``output_latency_ticks`` controls when the report is emitted through the output
connection. A job can override the device default:

.. code-block:: python

   from simyuj.qstate import SubsystemId
   from simyuj.components.detectors.qubit_readout import QubitReadoutJob

   q0 = SubsystemId("q0")

   job = QubitReadoutJob(
       job_id="read-q0-with-latency",
       targets=(q0,),
       output_latency_ticks=20,
   )

The stored report time remains the event-handling time. The latency override
affects output emission only.

The emitted output event carries routing metadata for downstream components:
``device_id``, ``output_port``, ``report_id``, ``job_id``, and
``qubit_readout_device``.

Reports
-------

``QubitReadoutDevice`` reports use the same ``DetectionReport`` record as other
detector components, but job-style qubit readout does not produce raw detector
clicks.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Report field
     - Meaning for qubit readout
   * - ``success``
     - ``True`` when the reported outcome is not ``None``.
   * - ``outcome``
     - Classical reported label after the readout model.
   * - ``qstate_result``
     - True qstate measurement result object returned by the qstate manager.
   * - ``measurement_method``
     - Selected measurement method, such as ``"projective"`` or ``"bell"``.
   * - ``measurement_label``
     - Human-readable measurement label, such as ``"z"`` or ``"bell"``.
   * - ``raw_clicks``
     - Always empty for job-style qubit readout.
   * - ``flags``
     - Empty on success; contains ``"no_outcome"`` when no logical outcome is
       available.
   * - ``meta``
     - Includes target IDs, measurement metadata, true/reported labels, qstate
       probability fields when available, readout model type, device metadata,
       and job metadata.

Report metadata always starts with ``job_id`` and ``readout_job_id``. Device
``detector_meta`` entries come next, followed by per-job ``meta`` entries.
Duplicate metadata keys are preserved, so downstream code should choose a
consistent first-match or last-match lookup rule.

Reproducibility
---------------

Bind the device before event execution. Binding declares timeline-owned RNG
streams for:

* ``(device_id, "qubit_readout", "measurement_choice")``;
* ``(device_id, "qubit_readout", "qstate_measurement")``;
* ``(device_id, "qubit_readout", "readout_model")``.

With a fixed timeline seed, fixed configuration, and the same event sequence,
qubit-readout behavior is reproducible. Binding is idempotent for the same
timeline and rejects rebinding to a different timeline.

Important Behavior Notes
------------------------

* ``QubitReadoutDevice`` accepts only ``ACTION_RUN_QUBIT_READOUT``.
* The event payload must be a ``QubitReadoutJob``.
* Jobs use explicit qstate ``SubsystemId`` targets, not incoming signals.
* Jobs do not produce raw detector clicks.
* The component has a classical output port named ``out`` and no quantum input
  port.
* ``readout_model=None`` reports the true qstate label; mappings and callables
  can change only the reported classical outcome.
* Confusion-map rows must be non-empty and sum to one.
* Outcomes absent from a confusion map pass through unchanged.
* ``Measure.none()`` produces a ``"no_outcome"`` report rather than a detector
  miss.
* Per-job measurement and collapse overrides affect only that job.
* Per-job output latency overrides output emission only; the stored report time
  remains the handling time.
* Bell readout is supported through ``Measure.bell()`` when the job targets are
  valid Bell-measurement targets for the qstate manager.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/qubit_readout.py``

.. automodule:: simyuj.components.detectors.qubit_readout
   :members: QubitReadoutDevice, QubitReadoutJob, QubitReadoutModel, IdentityQubitReadout, ConfusionMapQubitReadout, qubit_readout_model_from_spec
   :show-inheritance:
