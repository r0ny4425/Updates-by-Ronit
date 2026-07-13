.. _memory-noise:

Memory Storage Noise
====================

The memory noise helpers normalize public ``QuantumMemory.noise_models``
configuration into one ordered storage-noise chain per physical memory
position.

Storage noise here means passive noise accumulated while a memory position is
occupied. Absorb/emit efficiency, memory readout, detector noise, and individual
qstate noise-channel physics are configured elsewhere.

Accepted Noise Models
---------------------

A storage-noise model is accepted structurally. It may be either an instance of
``simyuj.qstate.noise.NoiseChannel`` or any object that exposes a callable
``resolve`` method.

The memory helper only stores and orders these models. Each qstate noise model
owns its own mathematical behavior and interpretation of elapsed ``duration_s``.

Input Shapes
------------

``normalize_memory_noise_models`` accepts three public input shapes:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Input
     - Meaning
     - Normalized form
   * - ``None``
     - No passive storage noise on any position.
     - One empty tuple per position.
   * - One noise model
     - The same single model is used for every position.
     - ``((model,), (model,), ...)``
   * - Position-indexed sequence
     - One entry per physical memory position.
     - One tuple per position.

The outer normalized tuple is indexed by physical memory position. The inner
tuple is the ordered chain of noise models for that position.

Important Sequence Rule
-----------------------

The top-level sequence is always position-indexed. It is not interpreted as one
shared chain.

With ``num_positions=2``, ``noise_models=(dephasing, amplitude_decay)`` means
``dephasing`` for position 0 and ``amplitude_decay`` for position 1. To give the
same chain to both positions, repeat the chain once per position:

.. code-block:: python

   noise_models=(
       (dephasing, amplitude_decay),
       (dephasing, amplitude_decay),
   )

Order matters inside each position's chain because ``QuantumMemory`` passes the
tuple to the qstate manager in that order.

Lazy Application In QuantumMemory
---------------------------------

The normalized chains are stored on the memory component at construction time.
Storage noise is applied later, only when an operation touches an occupied
position.

For one position, ``QuantumMemory`` computes elapsed storage time from
``record.last_noise_update_time`` to ``timeline.current_time``, converts that
duration to seconds, and calls ``timeline.qstate.apply_noise_models(...)`` for
the stable memory subsystem. Afterward, ``last_noise_update_time`` is advanced
to the current simulation tick.

When Noise Is Applied
~~~~~~~~~~~~~~~~~~~~~

``QuantumMemory`` applies pending storage noise before operations that touch an
occupied memory subsystem:

* successful emission;
* operator application;
* memory measurement.

Emission is a special case: the pending interval is applied when the delayed
emit completes, so passive storage noise can accrue during the emit delay.

Delayed operator and measurement requests apply pending storage noise before the
position enters the busy state. The later delay models operation latency, not
additional passive storage time.

Absorption does not apply storage noise to the incoming photon before it is
stored. A successful absorb initializes ``last_noise_update_time`` to the absorb
completion tick.

Pending storage noise is not applied for metadata updates, discard operations,
or stale expiry removal.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/memories/noise.py``

.. automodule:: simyuj.components.memories.noise
   :members: MemoryNoiseModels, MemoryNoiseModelsInput, normalize_memory_noise_models
   :show-inheritance:
