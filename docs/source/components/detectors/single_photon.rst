Single Photon Detector
======================

``SinglePhotonDetector`` models one physical detector channel, such as one
APD, SNSPD channel, or detector pixel. It is used inside event-driven detector
components.

It covers efficiency, dark counts, dead time, timing jitter, afterpulsing, and
optional photon-number-resolving behavior for one active detection window.

Use ``SinglePhotonDetector`` directly for:

* configuring detector channels for a receiver component;
* testing detector physics in isolation;
* building a custom detector component that already owns event flow;
* studying click timing, dead time, dark counts, or afterpulsing.

``DetectorArray`` and ``BellStateAnalyzer`` own ports, event handling, and
qstate measurement.

Basic Channel Configuration
---------------------------

.. code-block:: python

   from simyuj.components.detectors.single_photon import SinglePhotonDetector
   from simyuj.components.detectors.primitives.params import (
       SinglePhotonDetectorParams,
   )

   params = SinglePhotonDetectorParams(
       efficiency=0.85,
       dark_count_rate_hz=100.0,
       dead_time_ticks=50,
       jitter_stddev_ticks=2.0,
   )

   detector = SinglePhotonDetector("d0", params=params)

The same immutable parameter object can be reused across several detector
channels:

.. code-block:: python

   detectors = (
       SinglePhotonDetector("d0", params=params),
       SinglePhotonDetector("d1", params=params),
   )

Evaluation Model
----------------

A detector channel is evaluated over an active observation window:

.. code-block:: python

   clicks = detector.evaluate_window(
       time=100,
       signal_present=True,
       window_duration_ticks=10,
       rngs=rngs,
       outcome_label="0",
   )

The return value is a tuple of ``RawClick`` records. It may be empty.

During one window, the detector builds click candidates from:

* signal detection efficiency;
* dark-count sampling;
* afterpulse sampling from a previous accepted click;
* timing jitter applied to candidate click times.

Candidates outside the active window are discarded. Accepted candidates update
the detector state.

Physical Assumptions
--------------------

``SinglePhotonDetector`` uses a compact window-level detector model:

.. list-table::
   :header-rows: 1
   :widths: 24 50 26

   * - Effect
     - Model
     - Main parameter
   * - Signal efficiency
     - A signal-exposed detector samples a Bernoulli event,
       :math:`P(\mathrm{signal\ click}) = \eta`. Edge cases ``0.0`` and
       ``1.0`` do not consume the efficiency RNG.
     - ``efficiency``
   * - Dark counts
     - Dark counts are sampled as
       :math:`N_\mathrm{dark} \sim
       \operatorname{Poisson}(R_\mathrm{dark}\Delta t)` over the active window.
       The active duration is converted from ticks to seconds before sampling.
     - ``dark_count_rate_hz``
   * - Jitter
     - Jitter is non-negative detector latency. The sampled delay is added to
       the candidate time; a delayed click outside the active window is not
       reported.
     - ``jitter_stddev_ticks``
   * - Dead time
     - After an accepted click, the detector blocks later candidates until
       ``dead_until``. If a later window starts before that tick, the whole
       window is blocked.
     - ``dead_time_ticks``
   * - Afterpulsing
     - Afterpulsing is sampled only after a previous accepted click.
       ``p_afterpulse`` is integrated future probability mass, not a per-window
       probability.
     - ``p_afterpulse``, ``afterpulse_decay_ticks``

Statefulness
------------

``SinglePhotonDetector`` keeps two pieces of physical state:

``dead_until``
   The earliest tick at which another click can be accepted.

``last_click_time``
   The most recent accepted click time, used by the afterpulse model.

Because this state persists across calls, reusing the same detector instance is
part of the model. Create a new detector instance only when you want a fresh
physical channel.

Threshold Behavior
------------------

For the default threshold-detector mode, one accepted click ends evaluation for
that window. If ``photon_number_resolving=True``, the detector may return
multiple accepted clicks, subject to dead-time filtering.

Same-time candidates are resolved deterministically in this order:

1. signal;
2. dark count;
3. afterpulse.

Isolated Test Example
---------------------

Most simulations let detector components bind RNG streams automatically. For a
small unit test, you can provide simple deterministic RNG objects yourself:

.. code-block:: pycon

   >>> from simyuj.components.detectors.single_photon import SinglePhotonDetector
   >>> from simyuj.components.detectors.primitives.params import (
   ...     SinglePhotonDetectorParams,
   ... )
   >>> from simyuj.components.detectors.primitives.rng import DetectorRNGStreams

   >>> class ZeroRNG:
   ...     def random(self):
   ...         return 0.0
   ...
   ...     def poisson(self, lam):
   ...         return 0
   ...
   ...     def normal(self, *, loc, scale):
   ...         return 0.0

   >>> detector = SinglePhotonDetector(
   ...     "d0",
   ...     params=SinglePhotonDetectorParams(efficiency=1.0),
   ... )

   >>> rngs = DetectorRNGStreams(
   ...     efficiency=ZeroRNG(),
   ...     dark=ZeroRNG(),
   ...     jitter=ZeroRNG(),
   ... )

   >>> clicks = detector.evaluate_window(
   ...     time=10,
   ...     signal_present=True,
   ...     window_duration_ticks=1,
   ...     rngs=rngs,
   ... )

   >>> clicks[0].trigger
   'signal'
   >>> clicks[0].time
   10

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/single_photon.py``

.. automodule:: simyuj.components.detectors.single_photon
   :members: SinglePhotonDetector
   :show-inheritance:
