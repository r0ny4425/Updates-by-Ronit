Primitives
==========

The primitives package contains the shared low-level pieces used across
SimYuj: identifiers, metadata helpers, validation functions, unit conversions,
subsystem labels, and transport payload records.

These modules are protocol-neutral and side-effect free. They validate and
shape data at public boundaries, while higher layers handle scheduling,
routing, qstate changes, and protocol behavior.

Unit helpers provide the shared physical convention for time conversion:
one simulation tick maps to one picosecond, while the engine remains an
integer-time system.

Most primitive objects are small immutable records or plain helper functions.
They are useful when building components, channels, control messages, config
loaders, or tests that need the same validation rules as the rest of the
simulator.

Module Pages
------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Identifiers <ids>
   Metadata <meta>
   Subsystems <subsystems>
   Units <units>
   Validation <validation>
   Messages <messages/index>
