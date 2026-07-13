Sinks
=====

API Reference
-------------

.. rubric:: Source File

``src/simyuj/tracing/sinks.py``

Choosing a Sink
---------------

Use ``MemorySink`` when tests or notebooks need to inspect records after a
run. Use ``TextSink`` when a human should read the trace while the simulation
runs. Use ``JsonlSink`` when traces should be saved, compared, or processed by
another tool.

``JsonlSink`` owns an open file handle. Use it as a context manager, or call
``close()`` when the run is finished:

.. code-block:: python

   from simyuj.tracing import JsonlSink, LogLevel, SimulationLogger

   with JsonlSink(path="traces/demo.jsonl") as sink:
       logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])
       # Run the simulation with this logger.

.. automodule:: simyuj.tracing.sinks
   :members:
   :show-inheritance:
