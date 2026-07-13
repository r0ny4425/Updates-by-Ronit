Quantum Targets
===============

``qstate_targets_from_signal(...)`` resolves the qstate subsystem target encoded
in a qstate-backed ``Signal``.

Use this signal-level adapter when a component already has a ``Signal`` and
needs the qstate subsystem id encoded in it. Memory and layout code should use
their own qstate interfaces.

Resolution Rule
---------------

The signal must carry a ``state_ref`` and exactly one ``SubsystemHandle`` in
``signal.state_targets``.

If that handle has metadata key ``"qstate_subsystem"``, the metadata value is
used as the resolved ``SubsystemId``. Otherwise the handle label is used.

.. code-block:: text

   qstate_subsystem metadata -> handle label

The helper raises ``ValueError`` when the signal has no ``state_ref`` or when it
does not carry exactly one state target. It raises ``TypeError`` when the target
is not a ``SubsystemHandle``.

Current signal-level component behavior supports exactly one qstate target per
signal. Multi-target and multi-qubit signal handling is future work. The
returned id is an identity only; existence checks happen through qstate or
memory interfaces.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/quantum_targets.py``

.. automodule:: simyuj.components.quantum_targets
   :members:
   :show-inheritance:
