Windows
=======

Window helpers connect readout exposures, gate schedules, detector RNG streams,
and ``SinglePhotonDetector`` channel models. They are used by detector
components after measurement and readout are complete.

Evaluation Flow
---------------

Detector-window evaluation usually follows this shape:

.. code-block:: text

   detector channels
      -> normalized detector order
      -> normalized readout exposures
      -> gate clipping
      -> per-channel RNG streams
      -> RawClick records

Detector order matters. Readout exposures must be normalized into the same
order as the detector tuple, so click metadata and output remain deterministic.

Gate Clipping
-------------

``active_detection_duration_at_arrival`` returns how much of a requested
detection window is active at an arrival tick.

If the arrival is outside the gate, the active duration is ``0``. If the gate
closes before the requested window ends, only the remaining active ticks are
evaluated.

Detector Evaluation
-------------------

``evaluate_detector_windows`` evaluates one exposure per detector channel. Each
exposure may shift the start time with ``time_offset_ticks``. Each shifted
window is clipped against the gate before the detector channel is asked to
evaluate it.

Unexposed detectors are still evaluated when their gate window is active. This
is intentional: an unexposed detector has no signal candidate, but it can still
produce dark counts or afterpulses.

RNG Binding
-----------

``bind_detector_rngs`` creates one ``DetectorRNGStreams`` bundle per detector
channel using timeline-owned streams. Components call this during ``bind`` so
detector randomness is declared before event execution starts.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/window.py``

.. automodule:: simyuj.components.detectors.primitives.window
   :members: normalize_detectors, validate_gate_model, bind_detector_rngs, require_detector_rngs, active_detection_duration_at_arrival, evaluate_detector_windows
   :show-inheritance:
