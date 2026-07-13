Dark Counts
===========

Dark-count primitives model detector clicks that occur without a signal photon.
They are used by ``SinglePhotonDetector`` while evaluating an active detection
window.

A detector channel has a dark-count rate, and each active window samples how
many dark events occurred inside that window.

Poisson Model
-------------

``DarkCountProcess`` treats dark counts as a Poisson process:

.. math::

   N \sim \mathrm{Poisson}(\lambda), \qquad \lambda = rT

where ``r`` is the dark-count rate in hertz and ``T`` is the active window
duration in seconds.

The probability of at least one dark count is:

.. math::

   P(N \ge 1) = 1 - e^{-rT}

The implementation uses a numerically stable form of this expression for small
windows.

Common Example
--------------

Use ``DarkCountProcess`` when you want to reason about dark-count probability
directly:

.. code-block:: python

   from simyuj.components.detectors.primitives.dark_counts import (
       DarkCountProcess,
   )

   process = DarkCountProcess(rate_hz=100.0)

   # Probability of at least one dark count in a 1 microsecond window.
   probability = process.p_at_least_one(1.0e-6)

Window Policy
-------------

``OnArrivalWindowDarkCounts`` decides where sampled dark counts appear inside a
detector window.

By default, the model is a coarse threshold-window model: if the Poisson count
is positive, it returns one dark click at the arrival tick.

.. code-block:: python

   from simyuj.components.detectors.primitives.dark_counts import (
       OnArrivalWindowDarkCounts,
   )

   policy = OnArrivalWindowDarkCounts(window_duration_ticks=50)

For time-resolved windows, dark-count offsets are sampled uniformly inside the
window after the Poisson count is known:

.. code-block:: python

   from simyuj.components.detectors.primitives.dark_counts import (
       OnArrivalWindowDarkCounts,
   )

   policy = OnArrivalWindowDarkCounts(
       window_duration_ticks=50,
       time_resolved=True,
   )

Threshold detectors usually keep only the earliest dark click. Use
``return_all_clicks=True`` only for time-resolved or photon-number-resolving
models.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/dark_counts.py``

.. automodule:: simyuj.components.detectors.primitives.dark_counts
   :members: DarkCountProcess, OnArrivalWindowDarkCounts
   :show-inheritance:
