Reporting
=========

``simyuj.components.memories.reporting`` converts memory reports into compact
timeline log records.

Report payload fields are documented in :doc:`reports`; this module derives the
timeline log entry for those reports.

``log_memory_report(...)`` accepts an already-created ``MemoryReport`` and emits
one ``LogLevel.DEBUG`` timeline record.

``QuantumMemory`` stores the report, logs it, and optionally sends it through
the notice port.

Log Fields
----------

``memory_report_log_fields(...)`` returns the category, message, and compact
metadata dictionary for a known memory report type.

The metadata is a trace summary, not a serialized report. Use the original
``MemoryReport`` when operation details matter.

``positions`` must be the current full ``memory.positions`` snapshot from the
same memory that produced the report. Successful absorb logs read ``expires_at``
from that snapshot after the position becomes occupied.

Operator reports are produced and logged only when the memory notice port is
connected. The qstate operator can still be applied without an operator report.

Some memory logs do not come from this module. Stale delayed completions, stale
expiry events, and binding logs are emitted directly by ``QuantumMemory``.

Metadata Lookup
---------------

``report_meta_value(report, key)`` returns the first matching metadata value or
``None``. Duplicate keys are not merged.

The helper handles only known memory report types. Unknown report types raise
``TypeError``.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/memories/reporting.py``

.. automodule:: simyuj.components.memories.reporting
   :members: log_memory_report, memory_report_log_fields, report_meta_value
   :show-inheritance:
