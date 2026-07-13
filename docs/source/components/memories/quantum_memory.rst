.. _quantum-memory:

Quantum Memory
==============

``QuantumMemory`` is the event-driven memory component for storing
qstate-backed photon signals in physical memory positions. It owns the
classical lifecycle of each position, while the actual quantum state remains in
the timeline's ``QuantumStateManager``.

Use it when a protocol needs a finite bank of memory slots that can absorb
incoming photon signals, wait, emit stored photons again, apply qstate
operators, measure stored positions, expire old contents, or update classical
metadata attached to a position.

Typical uses include repeater memories, finite-capacity memory banks, storage
lifetime experiments, delayed memory operations, detector-style memory
measurement, and protocol metadata attached to occupied positions.

For direct measurement of known qstate subsystems, use ``QubitReadoutDevice``.
For operations driven by arriving quantum signals, use detector components such
as ``DetectorArray`` or ``BellStateAnalyzer``.

At a high level, one successful storage cycle looks like this:

.. code-block:: text

   incoming photon Signal
      -> absorb into an available memory position
      -> relabel photon qstate target to a stable memory subsystem
      -> optionally apply lazy storage noise while the qubit waits
      -> emit, measure, operate on, expire, or discard the stored qubit
      -> produce an operation report and optionally transmit it on the notice port

Ports And Event Flow
--------------------

``QuantumMemory`` is a timeline component. Public operations are scheduled as
explicit events targeted at the memory.

.. list-table::
   :header-rows: 1
   :widths: 22 20 58

   * - Surface
     - Name
     - Meaning
   * - Quantum input port
     - ``in``
     - Receives photon ``Signal`` objects for absorption when the event payload
       is a ``PortDelivery``.
   * - Quantum output port
     - ``out``
     - Transmits emitted photon ``Signal`` objects after successful emission.
   * - Classical notice port
     - ``notice``
     - Transmits memory reports when connected.
   * - Stored reports
     - ``reports``
     - Local history for reports the component produces. Operator reports are
       produced only when the ``notice`` port is connected.
   * - Position snapshots
     - ``positions``
     - Tuple of immutable ``MemoryPositionRecord`` snapshots, one per physical
       memory position.

The public actions are:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Action
     - Payload
     - Effect
   * - ``MEMORY_ABSORB``
     - ``MemoryAbsorbRequest`` or ``PortDelivery`` carrying a ``Signal``
     - Store one incoming photon target in an available memory position.
   * - ``MEMORY_EMIT``
     - ``MemoryEmitRequest``
     - Emit one occupied position as a new photon signal.
   * - ``MEMORY_APPLY_OPERATOR``
     - ``MemoryApplyOperatorRequest``
     - Apply an operator to ordered occupied positions.
   * - ``MEMORY_MEASURE``
     - ``MemoryMeasureRequest``
     - Measure ordered occupied positions through the detector readout
       primitive.
   * - ``MEMORY_DISCARD``
     - ``MemoryDiscardRequest``
     - Explicitly discard one occupied position.
   * - ``MEMORY_EXPIRE``
     - ``MemoryExpireRequest``
     - Expire one occupied position when its occupancy token still matches.
   * - ``MEMORY_UPDATE_META``
     - ``MemoryUpdateMetaRequest``
     - Update classical metadata for one occupied position.

Delayed absorb, emit, operator, and measurement operations schedule private
completion actions internally. User code should schedule the public actions
above, not the private completion actions.

Minimal Memory
--------------

The smallest useful memory has one or more physical positions. Every position
starts as ``EMPTY``. In the scheduling examples below, ``controller`` means the
protocol or component that is scheduling the memory operation, and
``incoming_signal`` means a qstate-backed photon ``Signal`` already prepared by
the surrounding simulation.

.. code-block:: python

   memory = QuantumMemory(
       memory_id="mem0",
       num_positions=4,
       storage_lifetime_ticks=10_000,
       recovery_ticks=50,
   )

   memory.bind(BindingContext(timeline=timeline))

``memory_id`` is used for ports, reports, logs, RNG streams, and stable qstate
subsystem labels. ``num_positions`` is the finite physical capacity of the
memory.

Position Lifecycle
------------------

A memory position is a classical lifecycle record around a qstate subsystem.
The usual lifecycle is:

.. code-block:: text

   EMPTY -> ABSORBING -> OCCUPIED -> EMITTING -> EMPTY
                         OCCUPIED -> MEASURING -> OCCUPIED or EMPTY
                         OCCUPIED -> APPLYING_OPERATOR -> OCCUPIED
                         OCCUPIED -> EMPTY  (discard, expiry, destructive measurement)

The busy states appear only when the corresponding delay is nonzero. For
example, with ``absorb_delay_ticks=0``, a successful absorb goes directly from
``EMPTY`` to ``OCCUPIED`` during the handling event. With
``absorb_delay_ticks > 0``, the position becomes ``ABSORBING`` and completes at
a later internal event.

``EMPTY`` positions are available only when the current tick is at or after
``ready_at``. ``ready_at`` is advanced by ``recovery_ticks`` whenever a stored
quantum carrier is removed.

Absorbing A Photon Signal
-------------------------

Absorption stores a qstate-backed photon signal into a physical memory
position. The incoming signal must resolve to exactly one qstate target.
Successful absorption relabels that photon subsystem to the stable memory
subsystem for the selected position:

.. code-block:: text

   memory:<memory_id>:position:<position>

If ``position=None``, the memory selects the first available position by
physical index. If a position is provided explicitly, that position must be
``EMPTY`` and past its ``ready_at`` recovery tick.

.. code-block:: python

   request = MemoryAbsorbRequest(
       request_id="store-0",
       memory_id="mem0",
       signal=incoming_signal,   # qstate-backed photon Signal
       position=None,            # let the memory choose the first available slot
       session_id="session-a",
       meta=(("purpose", "temporary_storage"),),
   )

   timeline.schedule(
       Event(
           time=timeline.current_time,
           priority=0,
           target_ref=memory,
           action=MEMORY_ABSORB,
           payload_ref=request,
           source=controller,
           subsystem_id="components",
       )
   )

``MEMORY_ABSORB`` also accepts a ``PortDelivery`` whose target port is
``memory.input_port`` and whose payload is a ``Signal``. In that case the
memory builds a ``MemoryAbsorbRequest`` internally and chooses the position
automatically.

Absorb Success And Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~

``absorb_success_probability`` is a Bernoulli success probability:

.. math::

   P(\text{absorb succeeds}) = p_\mathrm{absorb}

When the probability is ``1.0``, absorption always succeeds and does not
consume the absorb RNG. When it is ``0.0``, absorption always fails and does not
consume the absorb RNG. Intermediate probabilities consume the timeline-owned
absorb RNG stream.

On absorb failure, the incoming photon target is discarded from qstate, the
position is returned to ``EMPTY``, and a failed ``MemoryAbsorbReport`` is
stored.

Storage Lifetime And Expiry
---------------------------

If ``storage_lifetime_ticks`` is configured, a successful absorb schedules an
expiry event for the occupied position:

.. math::

   t_\mathrm{expire} = t_\mathrm{stored} + L_\mathrm{storage}

where :math:`L_\mathrm{storage}` is ``storage_lifetime_ticks``.

The expiry request carries the position's occupancy token. If the position was
emitted, discarded, measured destructively, or reused before the expiry event
fires, the stale expiry is ignored. A current expiry discards the memory
subsystem, clears the position into recovery, and stores a
``MemoryExpireReport``. Stale expiry requests are trace-logged and do not
produce reports.

Emitting A Stored Photon
------------------------

Emission converts one occupied memory position back into an outgoing photon
``Signal``. Before emission is attempted, pending storage noise is applied for
elapsed storage time.

.. code-block:: python

   emit_request = MemoryEmitRequest(
       request_id="emit-0",
       memory_id="mem0",
       position=0,
       session_id="session-a",
   )

   timeline.schedule(
       Event(
           time=timeline.current_time + 100,
           priority=0,
           target_ref=memory,
           action=MEMORY_EMIT,
           payload_ref=emit_request,
           source=controller,
           subsystem_id="components",
       )
   )

Successful emission requires ``memory.output_port`` to be connected. The memory
relabels the stored memory subsystem to a unique emitted-photon subsystem:

.. code-block:: text

   photon:<memory_id>:position:<position>:emit:<counter>

It then transmits a photon ``Signal`` through the quantum output port, clears
the memory position into recovery, stores a ``MemoryEmitReport``, and optionally
transmits that report through the ``notice`` port.

``emit_success_probability`` is sampled the same way as
``absorb_success_probability``. On emission failure, the position remains
occupied and an unsuccessful ``MemoryEmitReport`` is stored.

Applying Operators To Memory Positions
--------------------------------------

``MEMORY_APPLY_OPERATOR`` applies an operator to the qstate targets stored in
ordered occupied memory positions. Request ordering is preserved when building
the qstate target tuple.

.. code-block:: python

   request = MemoryApplyOperatorRequest(
       request_id="op-0",
       memory_id="mem0",
       positions=(0, 1),
       operator=two_qubit_operator,
   )

   timeline.schedule(
       Event(
           time=timeline.current_time,
           priority=0,
           target_ref=memory,
           action=MEMORY_APPLY_OPERATOR,
           payload_ref=request,
           source=controller,
           subsystem_id="components",
       )
   )

Pending storage noise is applied before the operator operation enters its busy
state. If ``operator_delay_ticks`` is nonzero, the affected positions become
``APPLYING_OPERATOR`` and the operator is applied at the delayed completion
event. No additional passive storage noise is accrued during that operation
latency; the delay models operation time, not extra idle storage.

Operator reports are produced only when the classical ``notice`` port is
connected. This avoids creating operator reports when no downstream component is
listening for them.

Measuring Memory Positions
--------------------------

``MEMORY_MEASURE`` measures ordered occupied memory positions using the same
readout primitive as ``QubitReadoutDevice``. The resulting
``MemoryMeasurementReport`` wraps a detector-style ``DetectionReport``.

.. code-block:: python

   request = MemoryMeasureRequest(
       request_id="measure-0",
       memory_id="mem0",
       positions=(0,),
       measurement="z",
       collapse=True,
       destructive=True,
   )

   timeline.schedule(
       Event(
           time=timeline.current_time,
           priority=0,
           target_ref=memory,
           action=MEMORY_MEASURE,
           payload_ref=request,
           source=controller,
           subsystem_id="components",
       )
   )

The measurement used for the operation comes from
``MemoryMeasureRequest.measurement``, which defaults to ``"z"`` when omitted.
As with operator operations, pending storage noise is advanced before a delayed
measurement enters its busy state; measurement delay models operation latency,
not extra idle storage.

Destructive And Non-Destructive Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``destructive=True`` discards the measured memory subsystems after the qstate
measurement and clears their positions into recovery. This clearing happens
even if the detector readout reports no usable outcome.

``destructive=False`` leaves the measured positions occupied. If the qstate
measurement itself collapses the state, the memory keeps the collapsed state.
Set ``collapse=False`` only when the selected measurement supports non-collapsing
behavior and the protocol really needs it.

Readout Distortion
~~~~~~~~~~~~~~~~~~

``readout_model`` is passed to the detector readout primitive during memory
measurement. Use it for classical readout distortion such as confusion maps.
The true qstate result remains available inside the nested ``DetectionReport``;
the reported outcome may differ when the readout model distorts it.

.. code-block:: python

   from simyuj.components.memories import QuantumMemory

   memory = QuantumMemory(
       memory_id="mem0",
       num_positions=2,
       readout_model={
           "0": {"0": 0.98, "1": 0.02},
           "1": {"0": 0.05, "1": 0.95},
       },
   )

Discarding A Position
---------------------

Use ``MEMORY_DISCARD`` when protocol logic decides that one occupied position
should be removed immediately.

.. code-block:: python

   discard_request = MemoryDiscardRequest(
       request_id="discard-0",
       memory_id="mem0",
       position=0,
       reason="protocol_rejected",
   )

   timeline.schedule(
       Event(
           time=timeline.current_time,
           priority=0,
           target_ref=memory,
           action=MEMORY_DISCARD,
           payload_ref=discard_request,
           source=controller,
           subsystem_id="components",
       )
   )

Discard requires the position to be ``OCCUPIED``. It discards the memory
subsystem from qstate, clears the position into recovery, and stores a
``MemoryDiscardReport``.

Updating Position Metadata
--------------------------

Metadata updates are classical bookkeeping only. They do not touch qstate, do
not apply storage noise, and do not change timing fields.

.. code-block:: python

   update_request = MemoryUpdateMetaRequest(
       request_id="tag-0",
       memory_id="mem0",
       position=0,
       updates=(("basis", "Z"), ("role", "left_link")),
       remove_keys=("temporary",),
       expected_occupancy_token=None,
   )

   timeline.schedule(
       Event(
           time=timeline.current_time,
           priority=0,
           target_ref=memory,
           action=MEMORY_UPDATE_META,
           payload_ref=update_request,
           source=controller,
           subsystem_id="components",
       )
   )

The position must be ``OCCUPIED`` for a successful update. If
``expected_occupancy_token`` is provided, it must match the current token.
Failed metadata updates produce failure reports instead of mutating the
position.

Storage Noise
-------------

Storage noise is applied lazily. The memory does not continuously schedule noise
events. Instead, before operations that touch an occupied position, it computes
elapsed time since the last noise update:

.. math::

   \Delta t_\mathrm{s}
      = \mathrm{ticks\_to\_seconds}(t_\mathrm{now} - t_\mathrm{last})

where :math:`t_\mathrm{last}` is the position's
``last_noise_update_time``.

and applies the configured noise-model chain for that position to the stored
memory subsystem. The component advances storage noise before emit, operator,
and measurement operations. Metadata updates are classical-only and do not
advance storage noise; discard and expiry remove the subsystem without first
advancing storage noise.

A noise configuration may be:

* ``None`` for no storage noise;
* one noise model shared by every position;
* a position-indexed sequence where each entry is ``None``, one model, or an
  ordered sequence of models.

The exact accepted noise model forms are documented with the memory noise
helpers.

Delays And Stale Completions
----------------------------

The memory supports operation delays:

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - Delay field
     - Busy status
     - Completion behavior
   * - ``absorb_delay_ticks``
     - ``ABSORBING``
     - Relabels or discards the incoming photon at completion.
   * - ``emit_delay_ticks``
     - ``EMITTING``
     - Emits or reports failed emission at completion.
   * - ``operator_delay_ticks``
     - ``APPLYING_OPERATOR``
     - Applies the qstate operator at completion.
   * - ``measure_delay_ticks``
     - ``MEASURING``
     - Runs readout at completion.

Each delayed operation records the position occupancy token. If a position is
cleared, reused, or otherwise no longer has the expected token and status when
the completion event fires, the stale completion is logged and skipped.

Reports And Notices
-------------------

Every produced report is appended to ``memory.reports`` and logged. If the
classical ``notice`` port is connected, the same report object is transmitted on
that port at the current timeline tick using ``notice_priority``.

The report type depends on the operation:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Operation
     - Report
   * - Absorb
     - ``MemoryAbsorbReport``
   * - Emit
     - ``MemoryEmitReport``
   * - Apply operator
     - ``MemoryOperatorReport`` when the notice port is connected
   * - Measure
     - ``MemoryMeasurementReport`` containing a nested ``DetectionReport``
   * - Discard
     - ``MemoryDiscardReport``
   * - Expire
     - ``MemoryExpireReport``
   * - Update metadata
     - ``MemoryMetaUpdateReport``

A report describes the result after any qstate mutation has already happened.
For example, a successful emit report is produced after the memory subsystem has
already been relabeled to an emitted-photon subsystem and the memory position
has already been cleared.

Reproducibility
---------------

Call ``bind(context)`` before event execution. Binding declares deterministic
RNG streams for:

* absorption success;
* emission success;
* measurement selection;
* qstate measurement;
* readout-model sampling.

With a fixed timeline seed, fixed memory configuration, and the same event
sequence, memory stochastic behavior is reproducible. Binding is idempotent for
the same timeline and rejects rebinding to a different timeline.

Important Behavior Notes
------------------------

* ``num_positions`` must be positive.
* Absorb requires exactly one qstate target in the incoming photon signal.
* ``position=None`` selects the first available position by physical index.
* Explicit position selection fails if the position is not empty or is still
  recovering.
* Successful absorption relabels the photon subsystem to a stable memory
  subsystem label.
* Successful emission relabels the memory subsystem to a unique emitted-photon
  subsystem label.
* Successful emission requires the quantum output port to be connected.
* Storage noise is lazy; it is applied when occupied positions are touched.
* Destructive measurement clears positions even when readout reports no usable
  outcome.
* Stale delayed completions and stale expiry requests are ignored rather than
  clearing newer contents.
* Operator reports are stored only when the ``notice`` port is connected.
* One physical position stores at most one quantum carrier.
* Protocol code decides when to absorb, emit, measure, discard, operate on, or
  update memory positions.

API Reference
-------------

.. rubric:: Source File

``src/simyuj/components/memories/quantum_memory.py``

.. automodule:: simyuj.components.memories.quantum_memory
   :members: MEMORY_ABSORB, MEMORY_EMIT, MEMORY_APPLY_OPERATOR, MEMORY_MEASURE, MEMORY_DISCARD, MEMORY_EXPIRE, MEMORY_UPDATE_META, QuantumMemory
   :show-inheritance:
