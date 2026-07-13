Logger
======

API Reference
-------------

.. rubric:: Source File

``src/simyuj/tracing/logger.py``

Use ``SimulationLogger.log()`` for normal tracing. It checks the configured
level, assigns the next sequence number, freezes metadata, builds a
``SimulationLogRecord``, and forwards the record to each sink.

Use ``SimulationLogger.emit()`` only when you already have a complete
``SimulationLogRecord``. It forwards the record as-is: no level filtering is
applied and the logger sequence is not changed.

.. automodule:: simyuj.tracing.logger
   :members:
   :show-inheritance:
