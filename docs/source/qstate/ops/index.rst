Operations
==========

The operations package contains the objects and helpers used to change quantum
states: common gates, parameterized rotations, unitary records, reset helpers,
and Pauli-frame bookkeeping.

Most simulation code should apply operations through ``QuantumStateManager``:

.. code-block:: pycon

   >>> from simyuj.qstate import QuantumStateManager, SubsystemId
   >>> from simyuj.qstate.ops import CNOT, H

   >>> q0 = SubsystemId("q0")
   >>> q1 = SubsystemId("q1")
   >>> qstate = QuantumStateManager()
   >>> qstate.prepare("|0>", subsystems=(q0,))
   0
   >>> qstate.prepare("|0>", subsystems=(q1,))
   1

   >>> qstate.apply(H, targets=(q0,))
   0
   >>> state_ref = qstate.apply(CNOT, targets=(q0, q1))
   >>> qstate.record(state_ref).layout.subsystems
   (SubsystemId(name='q0'), SubsystemId(name='q1'))

The manager resolves subsystem targets, combines compatible states when needed,
and stores the updated record.

Unitary Operations
------------------

``Unitary`` records wrap dense matrices with an arity. Built-in gates such as
``X``, ``H``, ``CNOT``, and ``SWAP`` are ready to use. Rotation factories such
as ``RX(theta)``, ``RY(theta)``, ``RZ(theta)``, and ``CPhase(theta)`` create
parameterized unitaries.

For multi-qubit operations, target order matters. The first target is the
most-significant computational-basis axis for the operation matrix. With
``targets=(q0, q1)``, a two-qubit operation follows the basis order
:math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`,
:math:`|11\rangle`.

Reset and Frames
----------------

Reset helpers prepare qubit states such as :math:`|0\rangle` and
:math:`|+\rangle` after a discard or reduction workflow.

Pauli-frame helpers track classical correction state for teleportation,
entanglement swapping, and Bell-measurement workflows. They are bookkeeping
tools; they do not schedule events or apply protocol policy.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Apply <apply>
   Base <base>
   Frame <frame>
   Gates <gates>
   Reset <reset>
   Rotations <rotations>
   Unitary <unitary>
