SimYuj In 10 Minutes
====================

This guide walks through a complete SimYuj run: one photon leaves Alice, travels
through a quantum channel, and is measured by Bob.

The setup is deliberately small.

.. code-block:: text

   Alice source  -- quantum channel -->  Bob detector

Alice owns a photon source. Bob owns a detector. The channel between them adds
a five-tick delay. In this first run, nothing is noisy and nothing is lossy, so
we can focus on the part that matters first: how SimYuj moves a quantum signal
through simulation time.

By the end of the guide, you will have seen the main pieces of a SimYuj program:

* a ``Timeline`` that owns simulation time,
* components that react to events,
* ports that connect those components,
* a quantum signal backed by qstate,
* and a detector report that records what Bob saw.

The first run is simple, but it is not throwaway. It is the same shape larger
SimYuj experiments use: build the devices, wire their ports, schedule the first
event, run the timeline, and inspect the result.

For this example, keep one picture in mind:

.. code-block:: text

   tick 0    Alice emits a photon
   tick 0    the photon enters the channel
   tick 5    the photon reaches Bob
   tick 5    Bob measures it

That is SimYuj's basic rhythm. Components do not call each other directly. They
schedule events, and the timeline advances from one event to the next.

The Script
----------

Save this as ``one_photon.py`` and run it with SimYuj available.

.. code-block:: python

   from __future__ import annotations

   from simyuj.components import (
       ACTION_TRANSMIT_QUANTUM,
       QuantumChannel,
       SinglePhotonSource,
       connect_ports,
   )
   from simyuj.components.detectors import (
       ACTION_DETECT_SIGNAL,
       DetectorArray,
       SinglePhotonDetector,
       SinglePhotonDetectorParams,
   )
   from simyuj.engine import Timeline
   from simyuj.runtime.binding import bind_many


   # The timeline owns simulation time and deterministic random streams.
   timeline = Timeline(master_seed=1)

   # Alice emits one photon. With the default sampler, that photon is |0>.
   source = SinglePhotonSource(
       device_id="alice_source",
       frequency_hz=1e12,
       emission_probability=1.0,
       duration_s=1e-12,
   )

   # The channel waits five ticks and then forwards the photon.
   channel = QuantumChannel(
       channel_id="alice_to_bob",
       delay_ticks=5,
       fixed_insertion_loss_db=0.0,
   )

   # Bob measures in the z basis. Outcome 0 is wired to d0, outcome 1 to d1.
   detector = DetectorArray(
       device_id="bob_detector",
       detectors=(
           SinglePhotonDetector(
               detector_id="d0",
               params=SinglePhotonDetectorParams(
                   efficiency=1.0,
                   dark_count_rate_hz=0.0,
               ),
           ),
           SinglePhotonDetector(
               detector_id="d1",
               params=SinglePhotonDetectorParams(
                   efficiency=1.0,
                   dark_count_rate_hz=0.0,
               ),
           ),
       ),
       measurement="z",
       readout={"z": {"0": "d0", "1": "d1"}},
   )

   # Wire Alice to the channel, and the channel to Bob.
   connect_ports(
       source.output_port,
       channel.input_port,
       target_action=ACTION_TRANSMIT_QUANTUM,
   )
   connect_ports(
       channel.output_port,
       detector.input_port,
       target_action=ACTION_DETECT_SIGNAL,
   )

   # Bind components before the first event runs.
   bind_many([source, channel, detector], timeline)

   # Schedule Alice's first event, then run until no events remain.
   source.schedule_start(timeline)
   timeline.run_until_empty()

   print(f"timeline stopped at tick {timeline.current_time}")

   if not detector.reports:
       print("Bob recorded no detector report")
   else:
       report = detector.reports[0]
       clicked = ",".join(click.detector_id for click in report.raw_clicks) or "-"

       print(
           f"time={report.time} "
           f"signal={report.signal_id} "
           f"success={report.success} "
           f"outcome={report.outcome} "
           f"clicked={clicked}"
       )

You should see:

.. code-block:: text

   timeline stopped at tick 5
   time=5 signal=alice_source:photon:1 success=True outcome=0 clicked=d0

What Bob Saw
------------

The report is Bob's record of the measurement:

.. code-block:: text

   time=5 signal=alice_source:photon:1 success=True outcome=0 clicked=d0

``time=5``
   Bob measured the photon at simulation tick ``5``.

``signal=alice_source:photon:1``
   This was the first photon emitted by Alice's source.

``success=True``
   The detector produced a usable outcome.

``outcome=0``
   The measurement result was ``0``.

``clicked=d0``
   Detector channel ``d0`` clicked.

Why Did d0 Click?
-----------------

Alice's source used its default state sampler, so it emitted ``|0>``.

Bob measured in the ``z`` basis:

.. code-block:: python

   measurement="z"

The readout map says where each logical outcome goes:

.. code-block:: python

   readout={"z": {"0": "d0", "1": "d1"}}

So outcome ``0`` is reported through detector channel ``d0``.

Both detector channels are perfect in this first run. Their efficiency is
``1.0`` and their dark-count rate is ``0.0``. Nothing random or noisy is hiding
the basic flow.

Why Did It Happen At Tick 5?
----------------------------

The source emits at tick ``0``. The channel delay is five ticks:

.. code-block:: python

   delay_ticks=5

So Bob receives the photon at:

.. code-block:: text

   0 + 5 = 5

This is simulation time, not wall-clock time. SimYuj is not waiting five real
ticks. It is moving the timeline to the next scheduled event.

Try Changing The Experiment
---------------------------

First, make the channel longer:

.. code-block:: python

   channel = QuantumChannel(
       channel_id="alice_to_bob",
       delay_ticks=12,
       fixed_insertion_loss_db=0.0,
   )

Run the script again. Bob's report moves from tick ``5`` to tick ``12``.

Now make Bob's detector blind:

.. code-block:: python

   SinglePhotonDetectorParams(
       efficiency=0.0,
       dark_count_rate_hz=0.0,
   )

The photon still reaches Bob, but Bob cannot turn it into a signal click. The
report is no longer a successful detection.

Finally, try adding channel loss:

.. code-block:: python

   channel = QuantumChannel(
       channel_id="alice_to_bob",
       delay_ticks=5,
       fixed_insertion_loss_db=20.0,
   )

Now the photon may never reach Bob. When that happens, the script prints:

.. code-block:: text

   Bob recorded no detector report

That is different from a failed detector click. In this case Bob did not see
the photon at all.

What You Have Learned
---------------------

This first run used only three components, but it already showed the basic
SimYuj pattern:

.. code-block:: text

   build components -> connect ports -> bind -> schedule first event -> run

The important part is the middle:

.. code-block:: text

   connect ports -> schedule delivery events

Ports are not decorative handles. They are where one component's output becomes
another component's future input.

Where To Go Next
----------------

* :doc:`engine/index` explains event ordering and simulation time.
* :doc:`components/ports_connections` explains how component outputs become
  scheduled deliveries.
* :doc:`components/sources`, :doc:`components/channels`, and
  :doc:`components/detectors` cover the source, channel, and detector models
  used in this example.
* :doc:`qstate/index` explains how SimYuj stores quantum state and resolves
  measurements.
