Math
====

The qstate math package contains the small numerical helpers used by state,
operation, measurement, and noise code. It is mostly a developer-facing layer;
normal simulations should go through ``QuantumStateManager`` and the public
qstate workflow APIs.

These helpers keep dense-array behavior in one place: vector normalization,
density-matrix checks, probability cleanup, projectors, tensor products, local
axis-operator application, and operator expansion.

Conventions
-----------

Arrays are NumPy arrays, usually with ``complex128`` dtype. Qubit basis vectors
follow computational-basis order by integer index. Tensor products follow the
operand order supplied by the caller.

For an ``n``-qubit dense state, local operations are applied by moving the
target axes into position, applying the local operator, then restoring the
original layout order. Higher-level qstate code handles subsystem-to-axis
resolution before calling these helpers.

What Belongs Here
-----------------

Use this package for backend work such as:

* checking whether a matrix is unitary, Hermitian, or positive semidefinite;
* normalizing state vectors, density matrices, and probability vectors;
* building projectors and outer products;
* applying dense local operators over selected tensor axes without full
  expansion when possible;
* expanding dense operators when a full Hilbert-space matrix is required.

Protocol logic, component behavior, event scheduling, and ownership tracking do
not belong in this package.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Constants <const>
   Linear Algebra <linalg>
   Matrix <matrix>
   Probability <prob>
   Projector <projector>
   Tensor <tensor>
