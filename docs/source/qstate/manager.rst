Manager
=======

Noise Policy
------------

``QuantumStateManager`` accepts ``noise_mode``. The default ``"density"`` mode
keeps exact mixed-state channel evolution for ``apply_noise_models()``. The
optional ``"sampled_ket"`` mode samples Kraus branches for existing ket records
using ``noise_rng`` and leaves density and Bell-diagonal records as exact
ensemble representations.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/qstate/manager.py``

.. automodule:: simyuj.qstate.manager
   :members:
   :show-inheritance:
