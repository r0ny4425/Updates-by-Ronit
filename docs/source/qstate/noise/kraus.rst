Kraus
=====

Sampled Ket Branches
--------------------

Density application evaluates the exact channel
:math:`\rho \mapsto \sum_i K_i \rho K_i^\dagger`.

Sampled ket application keeps one pure trajectory. For each branch:

.. math::

   p_i = ||K_i |\psi\rangle||^2

.. math::

   |\psi_i\rangle = K_i |\psi\rangle / \sqrt{p_i}

One branch is sampled using the manager's explicit noise RNG. This is a Monte
Carlo trajectory realization; it does not store the mixed state in one ket
record.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/qstate/noise/kraus.py``

.. automodule:: simyuj.qstate.noise.kraus
   :members:
   :show-inheritance:
