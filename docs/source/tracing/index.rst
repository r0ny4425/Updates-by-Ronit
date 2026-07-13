Tracing
=======

Tracing records simulation activity without changing simulation behavior.

It is useful for debugging, tests, tutorials, and saved run logs. Timeline and
component code can emit structured records with simulation time, event IDs,
actions, endpoint names, and metadata.

Quick Start
-----------

For tests and notebooks, start with an in-memory sink:

.. code-block:: python

   from simyuj.engine import Timeline
   from simyuj.tracing import LogLevel, MemorySink, SimulationLogger

   sink = MemorySink()
   logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])

   timeline = Timeline(logger=logger)

   # Run your simulation here.

   for record in sink.records:
       print(record.sequence, record.category, record.message)

For command-line examples, use a text sink:

.. code-block:: python

   from simyuj.tracing import LogLevel, SimulationLogger, TextSink

   logger = SimulationLogger(
       level=LogLevel.INFO,
       sinks=[TextSink()],
       session_id="demo-run",
   )

For saved traces, use ``JsonlSink``. JSON Lines output is usually easiest to
archive, compare between runs, or load into another tool.

Choosing a Level
----------------

``LogLevel`` is ordered from least to most verbose:

.. list-table::
   :header-rows: 1

   * - Level
     - Meaning
   * - ``OFF``
     - Disable logging through ``SimulationLogger``.
   * - ``ERROR``
     - Error records only.
   * - ``WARNING``
     - Warnings and errors.
   * - ``INFO``
     - High-level lifecycle and user-facing records.
   * - ``DEBUG``
     - More detailed diagnostic records.
   * - ``TRACE``
     - Engine-internal and fine-grained trace records.

Use ``INFO`` when you want the shape of a run: lifecycle records, major
timeline activity, and user-facing progress.

Use ``DEBUG`` when something is confusing and you need more context.

Use ``TRACE`` when you are inspecting engine-level behavior such as scheduling,
batch execution, and individual event execution. It can produce many records,
so keep it for focused debugging.

What a Record Contains
----------------------

Each emitted ``SimulationLogRecord`` has a logger sequence number, level,
category, message, and optional simulation context such as:

* simulation time
* timeline event ID
* event action
* source and target names
* session, node, or link ID
* structured metadata

``SimulationLogger`` assigns sequence numbers only to records that pass level
filtering. Filtered-out records do not consume sequence numbers.

Metadata is stored as key/value pairs. ``freeze_meta`` validates that keys are
strings and freezes the outer container; values are kept as supplied by the
caller.

Sinks
-----

Sinks decide where records go:

.. list-table::
   :header-rows: 1

   * - Sink
     - Use it for
   * - ``MemorySink``
     - Tests, notebooks, and small interactive runs.
   * - ``TextSink``
     - Human-readable console or file output.
   * - ``JsonlSink``
     - Structured logs that should be saved or processed later.
   * - ``NullSink``
     - Explicitly discarding records.

``TextSink`` writes one readable line per record. ``JsonlSink`` writes one JSON
object per line and keeps metadata order by storing metadata as pairs.

Timeline Integration
--------------------

The engine timeline emits observational records for scheduling, batch
execution, event execution, run lifecycle, and queue exhaustion according to
the configured logger level.

Logging From Runtime Code
-------------------------

Code running during event handling can log through the ``Timeline`` passed to
``handle_event()``:

.. code-block:: python

   from simyuj.tracing import LogLevel

   class MyComponent:
       def handle_event(self, event, timeline):
           timeline.log(
               LogLevel.INFO,
               "component.my_component.received",
               "received event",
               event=event,
               meta={"payload_type": type(event.payload).__name__},
           )

Pass ``event=event`` when the record should inherit the event ID, action,
source name, and target name. Put small structured details in ``meta`` instead
of encoding them into the message.

Do not call sink objects directly from runtime code; the timeline owns the
logger for the run.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Levels <levels>
   Records <records>
   Sinks <sinks>
   Logger <logger>
