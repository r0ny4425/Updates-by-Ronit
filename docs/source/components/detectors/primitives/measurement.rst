Measurement
===========

Measurement primitives describe what qstate operation a detector should run.

The main idea is split in two steps:

1. choose a ``MeasurementCall`` from a ``Measure`` policy;
2. execute that call through ``execute_measurement_call``.

Only ``execute_measurement_call`` crosses into qstate execution. ``Measure`` and
``MeasurementCall`` describe the selected operation.

Basic Basis Measurement
-----------------------

Most detector arrays use a fixed projective basis:

.. code-block:: python

   from simyuj.components.detectors.primitives.measurement import Measure

   measurement = Measure.basis("z")

When a component handles a signal, it creates a ``MeasurementContext`` and asks
the policy to choose a call:

.. code-block:: python

   call = measurement.choose(context)

The call records the method, basis, target spec, collapse setting, and label.
It does not measure qstate by itself.

Measurement Methods
-------------------

``Measure.basis(...)``
   Projective measurement in a named basis such as ``"z"`` or ``"x"``.

``Measure.povm(...)``
   POVM measurement using a qstate ``POVM`` object.

``Measure.bell(...)``
   Bell-basis measurement, usually used by ``BellStateAnalyzer``.

``Measure.none(...)``
   No qstate measurement. This can optionally discard targets.

``Measure.random(...)``
   Randomly choose among measurement specs using probabilities that must sum to
   one.

``Measure.by_meta(...)``
   Choose a measurement from detector or signal metadata.

Example: Random Basis Choice
----------------------------

A common protocol pattern is to choose between two bases:

.. code-block:: python

   from simyuj.components.detectors.primitives.measurement import Measure

   measurement = Measure.random(
       {
           "z": 0.5,
           "x": 0.5,
       }
   )

When selected, the resulting ``MeasurementCall`` records the selected index,
probability, and measurement label. Detector reports copy this metadata so runs
can be audited later.

Example: Metadata-Selected Basis
--------------------------------

Use ``Measure.by_meta`` when the basis comes from detector metadata or signal
metadata:

.. code-block:: python

   from simyuj.components.detectors.primitives.measurement import Measure

   measurement = Measure.by_meta(
       "basis",
       {
           "z": "z",
           "x": "x",
       },
   )

Metadata lookup checks detector metadata first, then signal metadata, then
signal timing metadata.

Targets
-------

Measurement targets are explicit. The default target spec is ``"signal"``,
which means all qstate targets carried by the input signal.

Other supported target specs include:

- a ``SubsystemId``;
- an integer signal-target index;
- a named target such as ``"left"`` or ``"right"``;
- a tuple of target specs;
- a callable that returns one or more ``SubsystemId`` values.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/detectors/primitives/measurement.py``

.. automodule:: simyuj.components.detectors.primitives.measurement
   :members: MeasurementCall, MeasurementContext, resolve_measurement_targets, Measure, execute_measurement_call
   :show-inheritance:
