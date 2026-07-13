State
=====

The state package contains the quantum-state representations used by
``QuantumStateManager``. Most code reaches these representations through
manager workflows such as ``prepare()``, ``apply()``, ``measure()``, and
``apply_noise()``.

This layer explains what is stored behind a state reference and why some
operations need a particular representation.

Representations
---------------

SimYuj currently uses three state representations.

``KetState``
   A pure state vector :math:`|\psi\rangle`. This is the natural representation for
   ideal unitary evolution and projective measurements without noise.

   For an :math:`n`-qubit state, the vector has :math:`2^n` amplitudes in
   computational-basis order.

``DensityState``
   A density matrix :math:`\rho`. This representation is used when mixed states,
   noise channels, or POVMs are involved.

   For an :math:`n`-qubit state, :math:`\rho` is a
   :math:`2^n \times 2^n` matrix. Noise models act on density states through
   Kraus operators.

``BellDiagState``
   A compact two-qubit Bell-diagonal state. It stores probabilities over the
   Bell basis instead of a full density matrix.

   This is useful for entanglement workflows where the state is known to stay
   Bell diagonal.

Choosing a Representation
-------------------------

For ordinary simulations, start with the manager and let workflows choose the
right lower-level path.

* Use ket states for ideal pure-state evolution.
* Use density states when applying noise or POVMs.
* Use Bell-diagonal states only when the model is explicitly Bell diagonal.

Representation conversion is available in the lower-level state modules, but
component and protocol code should usually stay at the manager level.

Handlers
--------

Each representation has a handler used by ``QuantumStateManager``. Handlers
know how to build, combine, transform, measure, and validate their payload
type.

This keeps representation-specific math out of the timeline and out of
protocol code. The timeline schedules events; qstate handlers operate on
quantum-state records.

Layout and Basis Order
----------------------

State payloads do not name qubits by themselves. The accompanying
``StateLayout`` maps logical subsystem IDs to tensor axes and dimensions.

For qubit states, SimYuj uses computational-basis order with axis 0 as the
most-significant axis. For example, a two-qubit layout ``(q0, q1)`` follows
the basis order :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`,
:math:`|11\rangle`.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Base <base>
   Bell Diagonal <bell_diag>
   Checks <check>
   Conversion <convert>
   Density <density>
   Ket <ket>
   Constructors <make>
   Metrics <metric>
   Reductions <reduce>
   Registry <registry>
