Space
=====

The space package is the bridge between names used by simulator code and axes
used by quantum-state math.

Components usually talk about logical things: a photon, a memory slot, or a
qubit owned by a node. State operations eventually need tensor axes. The space
records keep that mapping explicit.

Subsystem IDs
-------------

``SubsystemId`` is a stable name for one finite-dimensional quantum subsystem.
Two ``SubsystemId`` objects with the same name refer to the same logical
subsystem.

Use names that make ownership clear at the component boundary, for example
``"alice.memory.0"`` or ``"link.ab.photon"``. The qstate layer does not attach
protocol meaning to the name; it only uses it as an identity.

Layouts
-------

``StateLayout`` maps subsystem IDs to tensor axes and local Hilbert-space
dimensions.

.. code-block:: pycon

   >>> from simyuj.qstate.space import StateLayout, SubsystemId

   >>> q0 = SubsystemId("q0")
   >>> q1 = SubsystemId("q1")

   >>> layout = StateLayout((q0, q1), (2, 2))

   >>> layout.axis_of(q0)
   0
   >>> layout.axis_of(q1)
   1
   >>> layout.hilbert_dim
   4

The dimension tuple ``(2, 2)`` means two qubits. More generally, the total
Hilbert-space dimension is the product of the local dimensions:

.. math::

   \dim(\mathcal{H}) = d_0 d_1 \cdots d_{n-1}

For dense qubit states, axis 0 is the first subsystem and the most-significant
computational-basis axis. With layout ``(q0, q1)``, the basis order is
:math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`,
:math:`|11\rangle`.

Targets
-------

Manager workflows target logical subsystems:

.. code-block:: python

   qstate.apply(operation, targets=(q0, q1))

Target helpers resolve those subsystem IDs into layout axes before the lower
state handlers run. Lower-level helpers also accept subsystem names and integer
axes, but new component and protocol code should prefer ``SubsystemId`` values;
they keep ownership explicit and are harder to mix up.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Dimensions <dim>
   Layout <layout>
   Subsystems <subsystem>
   Targets <target>
