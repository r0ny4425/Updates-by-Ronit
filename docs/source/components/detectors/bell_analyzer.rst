.. _bell-state-analyzer:

Bell State Analyzer
===================

``BellStateAnalyzer`` is the signal-facing Bell-state measurement component. It
receives qstate-backed quantum ``Signal`` objects on two quantum ingress ports,
``left`` and ``right``. When one input from each side can be paired, the analyzer
runs a Bell-basis measurement on the two qstate targets, creates a
``DetectionReport``, stores it locally, and optionally emits it through the
classical ``out`` port.

Use it for receiver models where two independently delivered quantum signals
must be matched and measured jointly, such as entanglement swapping, repeater
node Bell-state measurements, or protocol-level heralding experiments.

Use ``QubitReadoutDevice`` instead when the protocol already knows the qstate
targets and wants to submit an explicit readout job without signal arrival
pairing. Use ``DetectorArray`` instead for one incoming signal measured by a
single receiver array.

Flow
----

A Bell-analysis event has two stages:

.. code-block:: text

   left Signal arrives                 right Signal arrives
          │                                    │
          └────────────── paired by buffer ────┘
                              │
                              ▼
                  qstate Bell measurement
                              │
                              ▼
                      BSM decision model
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
      ideal label report          linear-optical click-pattern layer
                                               │
                                               ▼
                                      physical detector windows
                                               │
                                               ▼
                                      resolved DetectionReport

The qstate Bell measurement and the final report outcome are related but not
identical. In ``ideal`` mode, the final report is the Bell decision after
heralding. In ``linear_optical`` mode, the final report is resolved from
observed detector-click patterns after detector-channel physics has been
evaluated.

Bell Basis
----------

The analyzer reports Bell labels using the lowercase strings ``"phi+"``,
``"phi-"``, ``"psi+"``, and ``"psi-"``.

The usual two-qubit Bell basis is:

.. math::

   |\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}},
   \qquad
   |\Phi^-\rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}

.. math::

   |\Psi^+\rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}},
   \qquad
   |\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}

The component requires exactly one qstate target from the left signal and
exactly one qstate target from the right signal. It performs one two-target Bell
measurement.

Ports And Events
----------------

``BellStateAnalyzer`` has two quantum input ports and one classical output port.

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Surface
     - Name
     - Meaning
   * - Quantum input port
     - ``left``
     - Receives a ``PortDelivery`` whose payload is a qstate-backed ``Signal``.
   * - Quantum input port
     - ``right``
     - Receives the partner-side ``PortDelivery``.
   * - Classical output port
     - ``out``
     - Emits ``DetectionReport`` objects when connected.
   * - Stored reports
     - ``reports``
     - Local history of Bell-analysis and timeout reports.

The analyzer accepts two event actions:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Action
     - Payload
   * - ``ACTION_RUN_BELL_ANALYSIS``
     - ``PortDelivery`` addressed to ``left_input_port`` or ``right_input_port``.
       The delivery payload must be a ``Signal``.
   * - ``ACTION_COINCIDENCE_TIMEOUT``
     - Internal ``CoincidenceTimeout`` payload scheduled by the analyzer when an
       input is buffered without an immediate partner.

Normally protocol or connection code schedules signal-arrival events. Manual
events must target the ``BellStateAnalyzer``, use
``ACTION_RUN_BELL_ANALYSIS``, and carry a ``PortDelivery`` whose target port is
either ``bsa.left_input_port`` or ``bsa.right_input_port``.

Minimal Ideal Analyzer
----------------------

The simplest analyzer uses the default Bell measurement and the default ideal
BSM model. It reports any of the four Bell labels when the qstate measurement
returns a supported Bell result.

.. code-block:: python

   timeline.schedule(
       Event(
           time=10,
           target_ref=bsa,
           action=ACTION_RUN_BELL_ANALYSIS,
           payload_ref=PortDelivery(
               payload=left_signal,
               source_port=left_source_port,
               target_port=bsa.left_input_port,
               connection_id="left_to_bsa0",
           ),
           subsystem_id="components",
       )
   )
   timeline.schedule(
       Event(
           time=12,
           target_ref=bsa,
           action=ACTION_RUN_BELL_ANALYSIS,
           payload_ref=PortDelivery(
               payload=right_signal,
               source_port=right_source_port,
               target_port=bsa.right_input_port,
               connection_id="right_to_bsa0",
           ),
           subsystem_id="components",
       )
   )

   timeline.run_until_empty()
   report = bsa.reports[-1]

This configuration has no physical detector channels. It performs the qstate
Bell measurement directly and stores a label-only ``DetectionReport``.

Ideal Analyzer With Heralding Efficiency
----------------------------------------

``heralding_efficiency`` is a classical Bernoulli survival probability applied
after the qstate Bell label is known and after the model has checked whether the
label is detectable.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer, BSMModel

   bsa = BellStateAnalyzer(
       device_id="heralded_bsa",
       bsm_model=BSMModel(kind="ideal", heralding_efficiency=0.92),
       coincidence_window_ticks=10,
       consume_matched_inputs=True,
       consume_unmatched_inputs=True,
   )

For ``ideal`` mode, the detectable label set is:

.. code-block:: text

   phi+, phi-, psi+, psi-

The report succeeds only when the measured Bell label is detectable and the
heralding-efficiency draw survives. With ``heralding_efficiency=1.0``, all
supported Bell labels are reported. With ``heralding_efficiency=0.0``, no
logical Bell outcome is reported.

BSM Modes
---------

``bsm_model`` controls the classical Bell-state decision model. It may be either
a ``BSMModel`` instance or the string ``"ideal"`` or ``"linear_optical"``. A
string uses default ``heralding_efficiency=1.0``.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Mode
     - Detectable labels
     - Final report path
   * - ``"ideal"``
     - ``phi+``, ``phi-``, ``psi+``, ``psi-``
     - Direct label report after heralding.
   * - ``"linear_optical"``
     - ``psi+``, ``psi-``
     - Click-pattern report after heralding and physical detector-window
       evaluation.

For either mode, the qstate Bell measurement may still return any Bell label.
The mode decides which labels can be reported by the measurement model.

Mathematically, the decision-level success probability is:

.. math::

   P(\mathrm{decision\ success})
      = P(\mathrm{label\ is\ detectable})\,\eta_h

where :math:`\eta_h` is ``heralding_efficiency``. If Bell labels are uniformly
distributed, then ``linear_optical`` has
:math:`P(\mathrm{label\ is\ detectable}) = 1/2` before detector-channel losses,
dark counts, gates, jitter, dead time, and click-pattern resolution.

Pairing And Coincidence Windows
-------------------------------

The analyzer buffers unmatched arrivals. A match is formed only between one
``left`` input and one ``right`` input.

Matching uses this order:

1. The analyzer resolves a pair key from the incoming signal if ``pairing_key``
   is not ``None``.
2. It scans the opposite-side buffer in stored order.
3. A candidate is eligible only when
   ``abs(incoming.arrival_time - candidate.arrival_time) <= coincidence_window_ticks``.
4. If either side has a non-``None`` pair key, the two inputs match only when
   the incoming key is non-``None`` and equal to the candidate key.
5. If neither side has a key, FIFO fallback may match the oldest eligible
   opposite-side input when ``allow_fifo_fallback=True``.

The default ``pairing_key`` is ``"bsa_pair_id"``. Pair-key lookup scans signal
metadata containers in this order: ``protocol_params``, ``meta``,
``correlation_meta``, then ``timing_meta``. The first matching string key is
used.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer

   bsa = BellStateAnalyzer(
       device_id="keyed_bsa",
       pairing_key="bsa_pair_id",
       allow_fifo_fallback=True,
       coincidence_window_ticks=20,
   )

Important pairing behavior:

* keyed inputs do not FIFO-match unkeyed inputs;
* two keyed inputs match only if the keys are equal and non-``None``;
* two unkeyed inputs match only through FIFO fallback;
* if ``pairing_key=None`` and ``allow_fifo_fallback=False``, ordinary unkeyed
  arrivals will not match and will eventually time out.

Timeouts
--------

When an arrival cannot be paired immediately, it is buffered and the analyzer
schedules ``ACTION_COINCIDENCE_TIMEOUT`` at:

.. math::

   t_\mathrm{timeout} = t_\mathrm{arrival}
      + \mathrm{coincidence\_window\_ticks} + 1

The extra tick allows a partner arriving exactly at the end of the coincidence
window to match before the timeout is processed.

If the timeout fires and the buffered input is still present, the analyzer
stores a failed timeout report with ``FLAG_TIMEOUT``. If
``consume_unmatched_inputs=True``, the timed-out qstate target is discarded. If
the input already matched, the stale timeout event is ignored.

Linear-Optical Analyzer
-----------------------

``linear_optical`` mode adds a physical click-pattern layer. This layer is
careful but abstract: it does not simulate beamsplitter interference directly.
Instead, the analyzer first performs a qstate Bell measurement, applies the BSM
decision model, then maps successful ``psi+`` or ``psi-`` decisions to configured
physical detector-pair patterns.

A successful decision selects one configured ``ClickPattern`` for the reported
Bell label. Detectors in that pattern are exposed to signal clicks; all other
detectors are still evaluated as unexposed channels, so they may still produce
dark counts or afterpulse clicks depending on detector parameters.

The final report succeeds only when the observed raw-click detector tuple
matches one configured click pattern exactly. Observed clicks, not the intended
selected pattern, determine the final physical report outcome.

The built-in physical resolver expects two-click patterns for ``psi+`` and
``psi-`` outcomes. More than two raw clicks are treated as a too-many-clicks
failure.

General Four-Detector Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The detector names below are intentionally generic. They represent four output
channels of whatever optical layout your protocol is modeling. The click-pattern
mapping is part of your model configuration.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer, BSMModel
   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.click import ClickPattern
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams(
       efficiency=0.90,
       dark_count_rate_hz=50.0,
       dead_time_ticks=100,
       jitter_stddev_ticks=2.0,
   )

   detectors = (
       SinglePhotonDetector("d0", params=params),
       SinglePhotonDetector("d1", params=params),
       SinglePhotonDetector("d2", params=params),
       SinglePhotonDetector("d3", params=params),
   )

   click_patterns = (
       ClickPattern(outcome="psi+", detector_ids=("d0", "d3")),
       ClickPattern(outcome="psi+", detector_ids=("d1", "d2")),
       ClickPattern(outcome="psi-", detector_ids=("d0", "d1")),
       ClickPattern(outcome="psi-", detector_ids=("d2", "d3")),
   )

   bsa = BellStateAnalyzer(
       device_id="linear_bsa",
       bsm_model=BSMModel(kind="linear_optical", heralding_efficiency=0.95),
       detectors=detectors,
       click_patterns=click_patterns,
       coincidence_window_ticks=10,
       detection_window_ticks=5,
       allow_fifo_fallback=True,
       consume_matched_inputs=True,
       consume_unmatched_inputs=True,
   )

``linear_optical`` mode validates the physical pattern configuration:

* at least one detector channel is required;
* ``click_patterns`` must be non-empty;
* at least one ``psi+`` pattern and one ``psi-`` pattern are required;
* each pattern must contain exactly two distinct detector IDs;
* every pattern detector ID must exist in ``detectors``;
* pattern outcomes must be ``"psi+"`` or ``"psi-"``;
* the same detector pair cannot be mapped to different outcomes;
* duplicate outcome/detector-pair mappings are rejected.

This validation does not decide whether the click-pattern map is physically
appropriate for your optical layout. It only checks the configuration contract
implemented by the component.

Gated Linear-Optical Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gates affect the physical detector-window layer in ``linear_optical`` mode.
They do not affect ideal label-only reporting.

.. code-block:: python

   gated_bsa = BellStateAnalyzer(
       device_id="gated_linear_bsa",
       bsm_model=BSMModel(kind="linear_optical", heralding_efficiency=0.95),
       detectors=detectors,
       click_patterns=click_patterns,
       gate_model=PeriodicGate(
           period_ticks=100,
           open_duration_ticks=10,
           first_open_tick=0,
       ),
       detection_window_ticks=8,
       coincidence_window_ticks=10,
   )

The detector-window layer clips physical detector evaluation against the gate
model. If the physical layer produces no raw clicks, the final physical report
fails with no-click/no-outcome flags.

Custom Bell Measurement
-----------------------

By default, the analyzer uses a collapsing Bell measurement over named targets
``"left"`` and ``"right"``. A custom ``measurement`` may be provided, but after
selection it must resolve to a Bell ``MeasurementCall`` with ``collapse=True``.
Otherwise event handling raises ``ValueError``.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer
   from simyuj.components.detectors.primitives.measurement import Measure

   bsa = BellStateAnalyzer(
       device_id="custom_measurement_bsa",
       measurement=Measure.bell(
           targets=("left", "right"),
           collapse=True,
           label="bell",
       ),
       bsm_model="ideal",
       coincidence_window_ticks=5,
   )

The target names ``"left"`` and ``"right"`` are supplied by the analyzer when a
matched pair is measured. They bind to the single qstate target carried by each
buffered input.

Report Semantics
----------------

Every analysis or timeout report is appended to ``bsa.reports``. If the output
port is connected, the same report object is transmitted through ``out``.

.. code-block:: python

   connect_ports(
       bsa.output_port,
       report_sink.input_port,
       target_action="receive_report",
   )

For matched inputs, ``signal_id`` is a pair:

.. code-block:: python

   (left_signal_id, right_signal_id)

For timeout reports, ``signal_id`` is the single timed-out signal ID.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Report case
     - ``success``
     - ``outcome``
   * - Ideal Bell label survives
     - ``True``
     - The Bell label, such as ``"phi+"``.
   * - Ideal label is undetectable or heralding misses
     - ``False``
     - ``None``.
   * - Linear-optical clicks match a configured pattern
     - ``True``
     - The matched pattern outcome, ``"psi+"`` or ``"psi-"``.
   * - Linear-optical no-click, too-many-clicks, or unresolved pattern
     - ``False``
     - ``None``.
   * - Coincidence timeout
     - ``False``
     - ``None``.

The most common failure flags are:

* ``"no_outcome"`` for a Bell decision that cannot report a logical label;
* ``"no_click"`` plus ``"no_outcome"`` when the physical layer produces no
  raw clicks;
* ``"double_click"`` plus ``"no_outcome"`` when more than two raw clicks are
  observed in the built-in two-click pattern model;
* ``"timeout"`` when one buffered input expires without a partner.

Useful report metadata keys include:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metadata key
     - Meaning
   * - ``bsm_model``
     - ``"ideal"`` or ``"linear_optical"``.
   * - ``readout_model``
     - ``"label"`` for ideal reports or ``"linear_optical_click_patterns"``
       for physical click-pattern reports.
   * - ``true_bell_label``
     - Bell label returned by the qstate measurement.
   * - ``reported_bell_label``
     - Final reported label for label-only reports, or observed click-pattern
       outcome for physical reports.
   * - ``detectable_bell_label``
     - Whether the BSM model considers the true label detectable.
   * - ``survived_heralding_efficiency``
     - Whether the heralding-efficiency draw survived.
   * - ``bsm_failure_reason``
     - ``"undetectable_bell_label"``, ``"heralding_efficiency_miss"``, click
       failure reason, or ``None``.
   * - ``click_failure_reason``
     - ``"no_click"``, ``"too_many_clicks"``, ``"unresolved_click_pattern"``,
       or ``None`` for physical reports.
   * - ``pair_key``
     - Pairing key used for the matched pair, when present.
   * - ``arrival_delta_ticks``
     - ``right_arrival_time - left_arrival_time``.
   * - ``coincidence_window_ticks``
     - Configured coincidence window.

When ``out`` is connected, the emitted output event also carries routing
metadata: ``device_id``, ``output_port``, ``report_id``, ``signal_id``, and
``bell_state_analyzer``.

Physical Report Edge Cases
--------------------------

In ``linear_optical`` mode, final report success is based on the observed raw
clicks. This has two practical consequences:

* A decision-level success can still produce a failed final report if detector
  efficiency, dead time, gate timing, jitter, or pattern mismatch prevents the
  selected pattern from appearing.
* A decision-level failure can still be accompanied by dark-count clicks. If
  those raw clicks exactly match a configured pattern, the physical report may
  resolve to that observed pattern. Inspect ``true_bell_label``,
  ``detectable_bell_label``, ``survived_heralding_efficiency``, and
  ``bsm_failure_reason`` when distinguishing true heralded events from
  accidental click-pattern events.

This behavior follows from the implemented physical layer: detector clicks are
sampled after the BSM decision, and the final physical outcome is resolved from
``raw_clicks`` against ``click_patterns``.

Output Timing
-------------

``output_latency_ticks`` delays emitted reports but does not change the stored
``DetectionReport.time``.

For label-only ideal reports and timeout reports, emission uses the analysis or
timeout time as the fallback ready time. For physical linear-optical reports,
emission uses the latest raw-click time when raw clicks exist; no-click physical
reports use the detector-window completion fallback. The configured
``output_latency_ticks`` is added after that ready time.

Signal Consumption
------------------

``consume_matched_inputs`` controls qstate lifetime after successful pairing and
analysis:

``consume_matched_inputs=True``
   Discards both matched qstate targets after the Bell analysis report is
   produced.

``consume_matched_inputs=False``
   Leaves the matched targets in the qstate manager after analysis. The Bell
   measurement itself still requires ``collapse=True``.

``consume_unmatched_inputs`` controls timed-out buffered inputs:

``consume_unmatched_inputs=True``
   Discards the single timed-out qstate target when a timeout report is
   produced.

``consume_unmatched_inputs=False``
   Leaves timed-out targets in the qstate manager.

Reproducibility
---------------

Bind the analyzer before event execution. Binding declares deterministic,
timeline-owned RNG streams for:

* ``(device_id, "bell_state_analyzer", "measurement_choice")``;
* ``(device_id, "bell_state_analyzer", "qstate_measurement")``;
* ``(device_id, "bell_state_analyzer", "bsm_decision")``;
* ``(device_id, "bell_state_analyzer", "pattern_choice")``;
* ``(device_id, "bell_state_analyzer", detector_id, "efficiency")``;
* ``(device_id, "bell_state_analyzer", detector_id, "dark")``;
* ``(device_id, "bell_state_analyzer", detector_id, "jitter")``;
* ``(device_id, "bell_state_analyzer", detector_id, "afterpulse")``.

Binding is idempotent for the same timeline and rejects rebinding to a different
timeline. With a fixed timeline seed, fixed analyzer configuration, and the same
event sequence, the analyzer behavior is reproducible.

Common Patterns
---------------

Keyed entanglement-swapping node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a pair key when the protocol already knows which left and right signals
belong together.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer

   bsa = BellStateAnalyzer(
       device_id="swap_bsa",
       pairing_key="bsa_pair_id",
       allow_fifo_fallback=False,
       coincidence_window_ticks=25,
   )

In this configuration, signals must carry matching ``bsa_pair_id`` metadata.
Unkeyed signals will not match by FIFO and will time out.

Unkeyed laboratory-style coincidence receiver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use FIFO fallback when arrivals are naturally paired by time rather than by an
explicit metadata key.

.. code-block:: python

   from simyuj.components.detectors.bell_analyzer import BellStateAnalyzer

   bsa = BellStateAnalyzer(
       device_id="fifo_bsa",
       pairing_key=None,
       allow_fifo_fallback=True,
       coincidence_window_ticks=3,
   )

This configuration matches the oldest opposite-side unkeyed input within the
coincidence window.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/bell_analyzer.py``

.. automodule:: simyuj.components.detectors.bell_analyzer
   :members: BellStateAnalyzer, BSMModel, BSMDecision, BufferedBellInput, CoincidenceTimeout, bsm_model_from_spec
   :show-inheritance:
