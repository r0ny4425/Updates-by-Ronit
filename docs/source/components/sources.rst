Sources
=======

Source components create quantum signals and place them into the component
graph. In a network model, they are the devices that prepare photons or
entangled pairs before channels, memories, detectors, or other receivers handle
them.

A source does not call the next component directly. It emits through an output
port, and the connection layer schedules delivery on the timeline. That keeps
source behavior replayable and keeps timing visible in the event history.

Two source models are provided:

- ``SinglePhotonSource`` prepares one photon signal for each successful
  emission attempt.
- ``EntangledPairSource`` prepares a shared two-qubit state and emits one
  signal for each member of the pair.

Basic Flow
----------

A source is usually used in this order:

.. code-block:: text

   schedule start -> attempt emission -> prepare qstate -> transmit signal

Each emission attempt may succeed or skip, depending on the configured emission
probability. Successful attempts create qstate-backed signals and send them
through connected quantum output ports.

Reports
-------

Sources also expose a classical ``report`` port. When connected, this port sends
a preparation report after a successful emission.

Reports are meant for control logic. They let an agent or protocol correlate
the emitted signal with sampler choices, qstate references, subsystem labels,
and timing metadata. They do not give the agent direct ownership of the qstate.

Timing And Replay
-----------------

Source timing is event-driven. Start time, duration, emission frequency, and
timing jitter are converted into simulation ticks before events are scheduled.

Random choices use timeline-owned RNG streams, so the same seed and source
configuration should replay the same emission decisions, sampled states, and
timing delays.

Use the detailed module pages when you need constructor arguments, report
fields, timing helper behavior, or the exact assumptions of each source class.

Module Pages
------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Single Photon Source <sources/single_photon_source>
   Entangled Pair Source <sources/entangled_pair_source>
   Source Reports <sources/reports>
   Shared Source Helpers <sources/common>
