Noise
=====

The noise package provides Kraus noise channels and helpers for applying them
to qstate records. The default manager policy applies noise exactly by using
density-matrix evolution.

Most simulation code should apply noise through ``QuantumStateManager``:

.. code-block:: pycon

   >>> from simyuj.qstate import QuantumStateManager, SubsystemId
   >>> from simyuj.qstate.noise import depolarizing

   >>> q0 = SubsystemId("q0")
   >>> qstate = QuantumStateManager()
   >>> qstate.prepare("|0>", rep="density", subsystems=(q0,))
   0

   >>> channel = depolarizing(0.01)
   >>> state_ref = qstate.apply_noise(channel, targets=(q0,))
   >>> qstate.get(state_ref).rho.shape
   (2, 2)

The manager handles representation conversion and target resolution before the
channel is applied.

Exact vs Sampled Ket Noise
--------------------------

``QuantumStateManager`` has a manager-level noise policy for
``apply_noise_models()``:

``noise_mode="density"``
   The default and exact behavior. Ket records are converted to density before
   noise is applied, so mixed states such as amplitude damping are represented
   directly.

``noise_mode="sampled_ket"``
   Existing ket records keep a pure-state trajectory by sampling one Kraus
   branch for each noise channel. A single run stores one possible pure
   outcome, not the exact mixed state. Repeated runs averaged together recover
   the density-channel statistics.

Sampled ket noise uses the chosen Kraus representation, so different Kraus
decompositions of the same density channel can produce different individual
trajectories even when their ensemble density matrix is the same. Density
records remain exact in sampled-ket mode. Bell-diagonal records remain compact
for supported Pauli noise and convert to density for channels that do not
preserve Bell diagonality.

Kraus Channels
--------------

Noise channels are represented with Kraus operators. For a density matrix
:math:`\rho`, a channel acts as:

.. math::

   \rho \mapsto \sum_i K_i \rho K_i^\dagger

The operators must satisfy the usual completeness condition for a
trace-preserving channel:

.. math::

   \sum_i K_i^\dagger K_i = I

The package includes common channels such as depolarizing noise, dephasing,
phase damping, amplitude damping, Pauli channels, imperfect two-qubit gates,
and T1/T2-style time-dependent models.

Target Order
------------

Operators act on target axes in the order supplied by the caller. For two-qubit
channels, ``targets=(q0, q1)`` means the first operand is the most-significant
axis of the channel matrix.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Base <base>
   Damping <damping>
   Dephasing <dephase>
   Depolarizing <depolarize>
   Kraus <kraus>
   Noisy Gates <noisy_gates>
   Pauli <pauli>
   T1/T2 <t1t2>
   Time <time>
