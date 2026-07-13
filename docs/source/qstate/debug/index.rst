Debug
=====

The debug package contains small helpers for inspecting qstate records and
checking store invariants during development.

These helpers are useful in tests, notebooks, and interactive debugging. They
are not part of the simulation event flow.

Dump Helpers
------------

Dump helpers return readable strings for layouts, payload summaries, records,
and stores. They do not print directly, so callers can log them, compare them
in tests, or display them in a notebook.

Invariant Helpers
-----------------

Invariant helpers check that records, layouts, payloads, and store ownership
indexes agree with each other. They raise errors when something is inconsistent
instead of silently repairing state.

Use them when adding a new state representation, changing store behavior, or
debugging a component that creates or consumes qstate references.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Dump <dump>
   Invariants <invariant>
