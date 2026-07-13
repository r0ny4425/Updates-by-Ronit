Parameters
==========

``SinglePhotonDetectorParams`` is the immutable parameter record for one
single-photon detector channel.

Field Meaning
-------------

``efficiency``
   Probability that a present signal produces a signal-click candidate.

``dark_count_rate_hz``
   Poisson dark-count rate in hertz, sampled over the active window duration.

``dead_time_ticks``
   Recovery interval after an accepted click. The interval starts from the
   reported firing time after jitter.

``jitter_stddev_ticks``
   Standard deviation for non-negative detector latency in ticks. This models
   delayed firing, not symmetric timestamp noise.

``p_afterpulse``
   Integrated afterpulse probability over future time. It is not a per-window
   probability.

``afterpulse_decay_ticks``
   Exponential decay constant controlling how afterpulse probability mass falls
   after a previous accepted click.

``photon_number_resolving``
   Whether one evaluation window may return multiple accepted clicks.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/params.py``

.. automodule:: simyuj.components.detectors.primitives.params
   :members: SinglePhotonDetectorParams
   :show-inheritance:
