Qstate
======

``simyuj.qstate`` is SimYuj's quantum-state layer. It gives components and
protocol code a small, explicit API for preparing, transforming, measuring,
and tracking quantum states while keeping the event timeline deterministic and
domain-agnostic.

Use this package when you need to:

* create named quantum subsystems;
* apply unitary operations or noise channels;
* measure projectively, with a POVM, or in the Bell basis;
* track which logical subsystem belongs to which stored state;
* inspect or validate quantum-state records while developing components.

The main entry point is ``QuantumStateManager``. Most application, component,
and protocol code should use the manager instead of manipulating payloads or
store records directly.

Quick Start
-----------

This example prepares two qubits, entangles them, and measures one side.

.. code-block:: pycon

   >>> from random import Random

   >>> from simyuj.qstate import QuantumStateManager, SubsystemId
   >>> from simyuj.qstate.ops import CNOT, H

   >>> q0 = SubsystemId("alice.photon")
   >>> q1 = SubsystemId("bob.photon")

   >>> qstate = QuantumStateManager()
   >>> qstate.prepare("|0>", subsystems=(q0,))
   0
   >>> qstate.prepare("|0>", subsystems=(q1,))
   1

   >>> qstate.apply(H, targets=(q0,))
   0
   >>> bell_ref = qstate.apply(CNOT, targets=(q0, q1))

   >>> result = qstate.measure(targets=(q0,), basis="z", rng=Random(7))

   >>> result.label
   '0'
   >>> result.post_state_ref == bell_ref
   True

The local ``Random`` keeps this standalone example reproducible. In a
timeline-driven simulation, use an RNG stream from ``Timeline.rng(...)`` so
stochastic qstate workflows participate in deterministic replay.

Mental Model
------------

The qstate layer has three main pieces:

``SubsystemId``
   A stable logical name for a quantum subsystem, such as a photon, memory
   slot, or qubit inside a component.

``QuantumStateStore``
   The ownership table for live quantum states. It records which state
   reference owns each subsystem.

``QuantumStateManager``
   The workflow API. It prepares states, combines compatible states when a
   multi-subsystem operation requires it, applies operations and noise,
   performs measurements, and updates the store.

A state operation targets logical subsystems, not tensor axes. Layout objects
map those subsystem names to tensor axes internally. Axis 0 is the first
subsystem and the most-significant computational-basis axis.

Representations
---------------

The manager currently supports:

* dense normalized ket states;
* dense density matrices;
* compact two-qubit Bell-diagonal records.

Representation-specific math lives below the manager. Component and protocol
code should usually choose workflows such as ``prepare()``, ``apply()``,
``apply_noise()``, ``measure()``, ``measure_povm()``, or ``measure_bell()``
instead of calling low-level handlers directly.

Noise Policy
------------

``QuantumStateManager`` defaults to exact noise-model evolution:
``apply_noise_models()`` converts ket records to density before applying noise.
Managers can also be created with ``noise_mode="sampled_ket"`` and an explicit
noise RNG. In that mode, existing ket records sample one Kraus branch and remain
pure ket trajectories, while density records remain exact and Bell-diagonal
records stay compact for supported Pauli noise.

Randomness Boundary
-------------------

Stochastic qstate workflows require an explicit caller-supplied RNG. In
timeline-driven simulations, that RNG should come from ``Timeline.rng(...)``
so measurements, state sampling, and probabilistic choices participate in
deterministic replay.

Boundary Rules
--------------

The qstate package does not schedule events, advance simulation time, or encode
concrete protocol policy. Components and control services decide when a state
operation should happen, then call qstate from scheduled event handlers.

This keeps the timeline generic and keeps quantum-state math in one service
layer.

Module Pages
------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Core <core>
   State <state/index>
   Space <space/index>
   Operations <ops/index>
   Measurement <measure/index>
   Noise <noise/index>
   Math <math/index>
   Debug <debug/index>
