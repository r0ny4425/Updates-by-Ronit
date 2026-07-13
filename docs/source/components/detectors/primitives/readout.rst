Readout
=======

Readout primitives map qstate measurement results to detector-channel
exposures.

Readout does not measure qstate and does not evaluate detector physics. It sits
between measurement and click generation.

Basic Two-Detector Readout
--------------------------

A common threshold receiver maps logical outcomes to detector ids:

.. code-block:: python

   from simyuj.components.detectors.primitives.readout import readout_from_spec

   readout = readout_from_spec(
       {
           "0": "d0",
           "1": "d1",
       },
       detector_ids=("d0", "d1"),
   )

If the qstate result label is ``"0"``, detector ``d0`` is exposed to the signal
and detector ``d1`` is returned as unexposed. Unexposed detectors are still
evaluated later, so they may still dark-count.

Basis-Aware Readout
-------------------

When the same detector array can measure in multiple bases, use a nested
mapping keyed by the selected measurement label:

.. code-block:: python

   from simyuj.components.detectors.primitives.readout import readout_from_spec

   readout = readout_from_spec(
       {
           "z": {
               "0": "d0",
               "1": "d1",
           },
           "x": {
               "+": "d0",
               "-": "d1",
           },
       },
       detector_ids=("d0", "d1"),
   )

The outer keys match ``MeasurementCall.label``. The inner keys match qstate
result labels.

Detector Exposure
-----------------

A ``DetectorExposure`` says how one detector channel should be evaluated:

- ``detector_id`` selects the channel;
- ``signal_present`` controls whether signal-click efficiency is sampled;
- ``outcome_label`` is copied to any raw click from that detector;
- ``time_offset_ticks`` shifts the exposure start time;
- ``meta`` is copied into raw-click metadata.

``signal_present=False`` does not disable the detector. It only means no signal
candidate is present. Dark counts and afterpulses can still happen.

Normalization
-------------

Detector components normalize readout output so every detector appears exactly
once and in component detector order. Missing detectors become unexposed
entries. This keeps dark-count behavior consistent even when a readout layout
only mentions the detector that should see the signal.

Custom Readout Layouts
----------------------

Pass a callable when a dictionary is not expressive enough. The callable
receives a ``ReadoutContext`` and must return a tuple of ``DetectorExposure``
records. Use callable layouts when one result exposes multiple channels,
exposures need time offsets, or the mapping depends on more than the result
label.

Qubit Readout Helper
--------------------

``run_qubit_readout`` is different from detector-array readout. It is used by
``QubitReadoutDevice`` for explicit qstate readout jobs. It measures explicit
subsystem ids, applies an optional classical readout model, and returns a
``DetectionReport`` without raw detector clicks.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/readout.py``

.. automodule:: simyuj.components.detectors.primitives.readout
   :members: DetectorExposure, ReadoutContext, ReadoutLayout, FixedReadout, OutcomeMapReadout, BasisOutcomeMapReadout, readout_from_spec, normalize_readout_exposures, run_qubit_readout
   :show-inheritance:
