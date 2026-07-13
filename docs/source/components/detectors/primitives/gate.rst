Gate Windows
============

Gate primitives describe when a detector is active.

Detector components use gates to answer two questions:

- is the detector open at the arrival tick?
- how much of the requested detection window is actually active?

All gate intervals are half-open: ``[start, end)``. A tick equal to ``start``
is inside the gate; a tick equal to ``end`` is outside.

Common Gates
------------

``AlwaysOpenGate``
   Active at every non-negative tick. This is the default for detector
   components.

``PeriodicGate``
   Repeated gating, such as a detector that opens for a short interval every
   clock period.

``ScheduledGate``
   Explicit active windows.

Example: Periodic Gate
----------------------

A detector that opens for 3 ticks every 10 ticks, starting at tick 2:

.. code-block:: pycon

   >>> from simyuj.components.detectors.primitives.gate import PeriodicGate

   >>> gate = PeriodicGate(
   ...     period_ticks=10,
   ...     open_duration_ticks=3,
   ...     first_open_tick=2,
   ... )

   >>> gate.is_open(2)
   True
   >>> gate.is_open(4)
   True
   >>> gate.is_open(5)
   False

The open windows are:

.. code-block:: text

   [2, 5), [12, 15), [22, 25), ...

Example: Scheduled Gate
-----------------------

Use ``ScheduledGate`` when the active windows are explicit:

.. code-block:: pycon

   >>> from simyuj.components.detectors.primitives.gate import (
   ...     GateWindow,
   ...     ScheduledGate,
   ... )

   >>> gate = ScheduledGate(
   ...     windows=(
   ...         GateWindow(start=100, end=110),
   ...         GateWindow(start=150, end=160),
   ...     )
   ... )

   >>> gate.is_open(105)
   True
   >>> gate.is_open(110)
   False

Window Clipping
---------------

Detector components may request a detection window longer than the remaining
gate time. In that case, only the active portion is evaluated.

For example, if a signal arrives at tick ``108`` and the gate is ``[100, 110)``,
only two ticks remain active: ``108`` and ``109``.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/gate.py``

.. automodule:: simyuj.components.detectors.primitives.gate
   :members: GateWindow, GateModel, AlwaysOpenGate, PeriodicGate, ScheduledGate
   :show-inheritance:
