RNG Streams
===========

``DetectorRNGStreams`` is a small bundle of per-detector RNG streams used by
``SinglePhotonDetector``. It does not create randomness; detector components
bind these streams from ``Timeline.rng(...)`` before event execution.

The fields separate signal efficiency, dark counts, jitter, and afterpulse
sampling. The detector model decides which streams are consumed for a given
configuration.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/rng.py``

.. automodule:: simyuj.components.detectors.primitives.rng
   :members: DetectorRNGStreams
   :show-inheritance:
