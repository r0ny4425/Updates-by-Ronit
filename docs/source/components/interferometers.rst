Interferometers
===============

Components that recombine optical amplitudes. They are timeline event targets;
the arithmetic they call lives in ``simyuj.components.coherent_optics`` and takes
no timeline.

``DelayInterferometer`` is the only member today, which is why its event actions
are defined in its own module rather than in a shared constants module.

Delay Interferometer
--------------------

An ideal unbalanced Mach-Zehnder: each arriving pulse is split at a 50:50
beamsplitter, one arm is delayed by :math:`\tau`, and that arm is recombined with
the *next* pulse's undelayed arm. In DPS-QKD :math:`\tau` equals the pulse
period, so the recombination compares the optical phases of two adjacent slots
and the phase difference becomes an intensity difference between two output
ports.

Three things surprise most readers the first time:

- **An N-pulse train produces N+1 output slots.** The first pulse's short arm and
  the last pulse's long arm each meet vacuum, split evenly, and carry no bit.
  The energy ledger is :math:`\tfrac12\mu + (N-1)\mu + \tfrac12\mu = N\mu`.
- **Nearest-neighbour pairing only.** One long arm is held at a time, so
  :math:`\tau \approx T` is the supported regime. A :math:`\tau` of two slot
  periods is not a configuration, it is a gap.
- **Both output ports must be connected.** The destructive port carrying nearly
  nothing is a result, not an absence.

The device is ideal by specification and declares no RNG streams. It never checks
:math:`\tau` against the pulse period, because it does not know the period; a
mismatch shows up as the reported temporal overlap collapsing on every slot.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/interferometers/delay_interferometer.py``

.. automodule:: simyuj.components.interferometers.delay_interferometer
   :members:
   :show-inheritance:
