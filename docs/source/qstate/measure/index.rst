Measurement
===========

The measurement package contains the basis records, Bell-state helpers, POVM
records, sampling routines, and result objects used by qstate workflows.

Most code should measure through ``QuantumStateManager``:

.. code-block:: pycon

   >>> from random import Random
   >>> from simyuj.qstate import QuantumStateManager, SubsystemId

   >>> q0 = SubsystemId("q0")
   >>> qstate = QuantumStateManager()
   >>> qstate.prepare("|0>", subsystems=(q0,))
   0

   >>> result = qstate.measure(targets=(q0,), basis="z", rng=Random(1))
   >>> result.label
   '0'

The manager finds the stored state, delegates to the right representation
handler, and writes back the collapsed state when collapse is enabled.

Projective Measurement
----------------------

Projective measurement uses a basis such as ``"x"``, ``"y"``, or ``"z"`` and
returns a ``MeasurementResult``. If the probabilities are not deterministic,
the caller must supply an RNG.

In timeline-driven simulations, that RNG should come from ``Timeline.rng(...)``
so measurement choices are reproducible under replay.

Bell Measurement
----------------

Bell measurement acts on two qubit targets and returns a ``BellResult``. The
Bell basis is used heavily by entanglement swapping and teleportation-style
workflows.

For a two-qubit state, the Bell basis is:

* ``phi+``: :math:`(|00\rangle + |11\rangle) / \sqrt{2}`
* ``phi-``: :math:`(|00\rangle - |11\rangle) / \sqrt{2}`
* ``psi+``: :math:`(|01\rangle + |10\rangle) / \sqrt{2}`
* ``psi-``: :math:`(|01\rangle - |10\rangle) / \sqrt{2}`

POVMs
-----

POVM helpers support generalized measurements on density states. A POVM is a
set of positive operators whose sum is the identity. Use
``QuantumStateManager.measure_povm()`` unless you are working inside a
state-backend routine.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Basis <basis>
   Bell <bell>
   POVM <povm>
   Projective <projective>
   Results <result>
   Sampling <sample>
