Signal
======

A ``Signal`` is an immutable transport envelope.

Signals are the in-flight carriers passed between sources, channels, memories,
and detectors. They carry identity, timing, qstate references, and metadata;
qstate math and protocol decisions happen in the layers that consume them.

When To Use A Signal
--------------------

Create a ``Signal`` when a component emits a physical or logical carrier such
as a photon, pulse, or entangled-pair member.

Use it to carry:

* where the signal came from;
* when it was emitted;
* what kind of physical carrier it represents;
* which encoding scheme describes it;
* which qstate record and subsystem handles it refers to;
* protocol, timing, and transport metadata needed by later components.

Do not use it to store mutable simulation state or protocol control flow. Those
belong in components, qstate stores, control agents, or timeline events.

A Small Example
---------------

A source can create a photon signal that points at a qstate record and names
the specific subsystem carried by the signal:

.. code-block:: python

   from simyuj.primitives.subsystems import SubsystemHandle
   from simyuj.signal import EncodingScheme, Signal, SignalKind


   target = SubsystemHandle(
       label="alice:photon:0",
       kind="qubit",
       index=0,
   )

   signal = Signal(
       id="alice-pulse-0",
       signal_kind=SignalKind.PHOTON,
       encoding_scheme=EncodingScheme.POLARIZATION,
       emission_time=0,
       origin="alice.source",
       wavelength_nm=1550.0,
       state_ref=12,
       state_targets=(target,),
       protocol_params=(("bb84.basis", "Z"),),
       timing_meta=(("emission_offset_ticks", 0),),
   )

The ``state_ref`` tells a consuming component where the qstate-backed payload
lives. The ``state_targets`` field tells it which subsystem inside that state
the signal represents.

In real component code, ``state_ref`` should come from the qstate layer. It is
shown as a fixed number here only to keep the example small.

The signal does not measure, mutate, or validate membership in the qstate
store. A detector or memory component must use the qstate layer for that.

Signal Records
--------------

``Signal`` is a frozen, slot-backed dataclass. Equality compares every field,
and hashing works when the values inside tuple fields are hashable.

Its fields are easier to read in groups:

.. list-table::
   :header-rows: 1

   * - Group
     - Fields
     - Meaning
   * - Identity
     - ``id``, ``origin``, ``emission_time``
     - Who produced the signal and when it entered the simulation.
   * - Physical description
     - ``signal_kind``, ``encoding_scheme``, ``wavelength_nm``
     - What carrier the signal represents. These fields are descriptive
       metadata, not operations.
   * - Qstate linkage
     - ``state_ref``, ``state_targets``
     - Where the qstate-backed payload lives and which subsystem or subsystems
       the signal represents.
   * - Correlation
     - ``correlation_id``, ``correlation_meta``
     - Optional metadata that links related signals, such as two members of an
       entangled pair.

Metadata
--------

Signals carry three tuple-shaped metadata channels:

.. list-table::
   :header-rows: 1

   * - Field
     - Intended Use
   * - ``protocol_params``
     - Protocol-level symbolic metadata such as basis labels.
   * - ``meta``
     - General simulator metadata.
   * - ``timing_meta``
     - Timing and debug metadata accumulated by transport components.

Each metadata container must be a tuple of ``(key, value)`` pairs with string
keys. Values are accepted as supplied and are not recursively validated or
copied.

Use ``protocol_params`` for symbolic protocol facts, such as
``("bb84.basis", "Z")``. Use ``meta`` for general simulator metadata. Use
``timing_meta`` for timing and debug details accumulated by transport paths.

Kinds and Encodings
-------------------

``SignalKind`` currently includes ``PHOTON``, ``PULSE``, and
``ENTANGLED_MEMBER``. ``EncodingScheme`` currently includes ``PHASE``,
``POLARIZATION``, ``FREQUENCY``, and ``TIME_BIN``. These enums are descriptive
transport metadata; they do not apply quantum operations by themselves.

Validation
----------

Construction validates enum fields, metadata tuple shape, non-negative
``emission_time``, positive ``wavelength_nm``, optional correlation IDs,
optional integer ``state_ref``, tuple-shaped ``state_targets`` containing
``SubsystemHandle`` instances, identifier type, and non-empty origin.

When ``validation_flag`` is false, construction-time validation is skipped.
That fast path is intended for callers that have already established the same
invariants.

Common Mistakes
---------------

Do not use ``correlation_id`` to identify a qubit. Use ``state_targets`` for
subsystem identity.

Do not put mutable values into metadata if the signal needs to be hashable.

Do not skip validation with ``validation_flag=False`` unless the caller has
already established the same invariants.

Module Pages
------------

.. toctree::
   :maxdepth: 1

   Signal Records <signal>
