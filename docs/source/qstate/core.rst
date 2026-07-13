Core
====

The qstate core modules define the public ownership workflow. The manager is
the high-level facade, the store assigns monotonic state references, and the
record and location dataclasses bind representation payloads to layouts and
metadata.

State references are integer handles local to a ``QuantumStateStore``. Logical
subsystem IDs are tracked through the space layer and indexed by the store so
each live subsystem has exactly one owner.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Manager <manager>
   Store <store>
   Records <record>
   Identifiers <ids>
   Errors <errors>
   Checks <check>
   RNG <rng>
   Sampler <sampler>
