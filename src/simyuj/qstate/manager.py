from __future__ import annotations

"""Workflow entry point for preparing, transforming, and measuring qstate records.

The manager coordinates the ownership store, representation registry, operation
handlers, noise channels, and measurement routines. It summarizes the public
state lifecycle without re-documenting the math implemented by lower layers.
Noise-model chains use a manager-level policy: the default ``"density"`` mode
keeps exact mixed-state evolution, while ``"sampled_ket"`` samples one Kraus
branch for existing ket records and leaves density/Bell-diagonal records as
exact ensemble representations.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from .check import Meta, MetaInput, coerce_meta, normalize_rep
from .errors import (
    DimensionError,
    InvalidLayoutError,
    InvalidOperationError,
    MeasurementError,
    NoiseError,
    StateOwnershipError,
)
from .ids import StateRef, SubsystemId
from .measure.basis import MeasurementBasis
from .measure.povm import POVM, measure_povm_density
from .measure.result import BellResult, MeasurementResult, POVMResult
from .noise.base import NoiseChannel, check_noise_channel, check_noise_models
from .ops.unitary import Unitary, check_unitary
from .record import QuantumStateRecord, SubsystemLocation
from .space import StateLayout, qubit_dims, resolve_targets
from .state import (
    BellDiagHandler,
    BellDiagState,
    DensityHandler,
    DensityState,
    KetHandler,
    KetState,
    StateRegistry,
    as_rep,
)
from .state.check import (
    _assert_payload_layout_compatible_trusted,
    _payload_num_qubits_trusted,
)
from .state.reduce import discard_density, reset_density
from .store import QuantumStateStore

_UNSET = object()
_NOISE_MODES = frozenset(("density", "sampled_ket"))


def _normalize_noise_mode(noise_mode: object) -> str:
    if not isinstance(noise_mode, str):
        raise TypeError("noise_mode must be str")
    normalized = noise_mode.strip().lower()
    if normalized not in _NOISE_MODES:
        raise ValueError("noise_mode must be 'density' or 'sampled_ket'")
    return normalized


class QuantumStateManager:
    """Coordinate qstate records through store-backed workflows.

    The manager owns a :class:`QuantumStateStore` and a representation registry.
    Public methods either delegate store operations or run a state workflow:
    prepare a payload, apply a unitary or noise channel, measure, convert,
    discard, or reset. Multi-state unitary, POVM, noise, and Bell workflows
    combine separately owned states when their representations match.

    Notes
    -----
    The example below constructs a local ``Random`` only to make a standalone
    qstate call deterministic. In timeline-driven simulations, pass an RNG
    stream obtained from ``Timeline.rng(...)`` so sampling participates in
    deterministic replay.

    Examples
    --------
    >>> from random import Random
    >>> from simyuj.qstate import QuantumStateManager, SubsystemId
    >>> from simyuj.qstate.ops import CNOT, H
    >>> q0 = SubsystemId("q0")
    >>> q1 = SubsystemId("q1")
    >>> manager = QuantumStateManager()
    >>> _ = manager.prepare("|0>", subsystems=(q0,))
    >>> _ = manager.prepare("|0>", subsystems=(q1,))
    >>> _ = manager.apply(H, targets=(q0,))
    >>> bell_ref = manager.apply(CNOT, targets=(q0, q1))
    >>> manager.record(bell_ref).layout.subsystems
    (SubsystemId(name='q0'), SubsystemId(name='q1'))
    >>> result = manager.measure(targets=(q0,), rng=Random(1))
    >>> result.label, result.post_state_ref
    ('0', 2)
    """

    __slots__ = ("_store", "_registry", "_noise_mode", "_noise_rng")

    def __init__(
        self,
        *,
        store: QuantumStateStore | None = None,
        noise_mode: str = "density",
        noise_rng: Any | None = None,
    ) -> None:
        """Create a manager with ket, density, and Bell-diagonal handlers.

        Parameters
        ----------
        store : QuantumStateStore or None, optional
            Existing store to use. If omitted, a new empty store is created.
        noise_mode : {"density", "sampled_ket"}, default="density"
            Policy used by :meth:`apply_noise_models`.  ``"density"`` preserves
            exact channel evolution by converting ket records to density before
            applying noise.  ``"sampled_ket"`` samples one Kraus branch when
            the owning record is already a ket; density and Bell-diagonal
            records remain exact ensemble representations.
        noise_rng : object or None, optional
            Explicit RNG stream used only for sampled ket noise. It must
            provide ``random()`` or ``rand()`` when sampling is required.
        """
        self._store = store if store is not None else QuantumStateStore()
        self._registry = StateRegistry()
        self._registry.register(KetHandler())
        self._registry.register(DensityHandler())
        self._registry.register(BellDiagHandler())
        self._noise_mode = _normalize_noise_mode(noise_mode)
        self._noise_rng = noise_rng

    @property
    def noise_mode(self) -> str:
        """Return the manager-level noise-model policy."""
        return self._noise_mode

    @property
    def store(self) -> QuantumStateStore:
        """Return the underlying ownership store."""
        return self._store

    def put(self, record: QuantumStateRecord) -> StateRef:
        """Insert ``record`` into the underlying store."""
        return self._store.put(record)

    def get(self, state_ref: StateRef) -> object:
        """Return the payload stored under ``state_ref``."""
        return self._store.get(state_ref)

    def record(self, state_ref: StateRef) -> QuantumStateRecord:
        """Return the record stored under ``state_ref``."""
        return self._store.record(state_ref)

    def get_record(self, state_ref: StateRef) -> QuantumStateRecord:
        """Return the record stored under ``state_ref``.

        This is an alias for :meth:`record`.
        """
        return self._store.get_record(state_ref)

    def replace(self, state_ref: StateRef, record: QuantumStateRecord) -> None:
        """Replace a live record in the underlying store."""
        self._store.replace(state_ref, record)

    def _replace_same_layout(
        self,
        state_ref: StateRef,
        record: QuantumStateRecord,
        *,
        payload: object = _UNSET,
        rep: str | None = None,
        meta: Meta | object = _UNSET,
    ) -> QuantumStateRecord:
        """Replace a record when payload metadata changes but layout does not."""
        metadata = cast(Meta, record.meta) if meta is _UNSET else cast(Meta, meta)
        updated = QuantumStateRecord._from_trusted(
            payload=record.payload if payload is _UNSET else payload,
            rep=record.rep if rep is None else rep,
            layout=record.layout,
            meta=metadata,
        )
        self._store._replace_same_layout(state_ref, updated)
        return updated

    def consume_and_put(
        self,
        consumed_refs: tuple[StateRef, ...],
        record: QuantumStateRecord,
    ) -> StateRef:
        """Consume live references and insert one replacement record."""
        return self._store.consume_and_put(consumed_refs, record)

    def delete(self, state_ref: StateRef) -> None:
        """Delete a live state reference from the underlying store."""
        self._store.delete(state_ref)

    def clear(self) -> None:
        """Remove all live records from the underlying store."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of live records in the store."""
        return self._store.size()

    def state_of(self, subsystem: SubsystemId) -> StateRef:
        """Return the live state reference that owns ``subsystem``."""
        return self._store.state_of(subsystem)

    def location_of(self, subsystem: SubsystemId) -> SubsystemLocation:
        """Return the store location for ``subsystem``."""
        return self._store.location_of(subsystem)

    def relabel_subsystem(
        self,
        old: SubsystemId,
        new: SubsystemId,
    ) -> StateRef:
        """Replace one subsystem identity while preserving its qstate record.

        The quantum payload, state reference, tensor axis, and subsystem
        dimension are unchanged. Only the layout identity and ownership index
        are updated.
        """
        state_ref = self.state_of(old)
        if self.store.contains_subsystem(new):
            raise StateOwnershipError(f"subsystem is already owned: {new}")

        record = self.record(state_ref)
        new_layout = StateLayout(
            tuple(
                new if subsystem == old else subsystem
                for subsystem in record.layout.subsystems
            ),
            record.layout.dims,
        )
        self.replace(state_ref, replace(record, layout=new_layout))
        return state_ref

    def prepare(
        self,
        state: object = "|0>",
        *,
        subsystems: tuple[SubsystemId | str, ...] | None = None,
        layout: StateLayout | None = None,
        rep: str = "ket",
        meta: MetaInput = None,
    ) -> StateRef:
        """Prepare a state payload and store it under a new reference.

        Parameters
        ----------
        state : object, optional
            Representation-specific state initializer.
        subsystems : tuple[SubsystemId | str, ...] or None, optional
            Subsystems used to build a qubit layout when ``layout`` is omitted.
        layout : StateLayout or None, optional
            Explicit tensor layout for the prepared payload.
        rep : str, optional
            Representation name used to select the state handler.
        meta : MetaInput, optional
            Metadata stored on the resulting record.

        Returns
        -------
        StateRef
            New state reference.

        Raises
        ------
        TypeError
            If layout or subsystem inputs have the wrong type.
        ValueError
            If neither ``subsystems`` nor ``layout`` is provided.
        DimensionError
            If the prepared payload and layout are incompatible.
        """
        normalized = normalize_rep(rep)
        handler = self._registry.get(normalized)
        payload = handler.make(state)
        prepared_layout = self._prepare_layout(
            payload,
            subsystems=subsystems,
            layout=layout,
        )
        return self.put(
            QuantumStateRecord(
                payload=payload,
                rep=normalized,
                layout=prepared_layout,
                meta=meta,
            )
        )

    def apply(
        self,
        operation: Unitary,
        *,
        targets: tuple[SubsystemId, ...],
    ) -> StateRef:
        """Apply a unitary operation to owned target subsystems.

        Targets in separate live records are first combined into one record when
        their representations match. The resulting record keeps the same state
        reference when no combine is needed, or receives a new reference after a
        consume-and-put combine.

        Parameters
        ----------
        operation : Unitary
            Unitary operation whose arity must match ``targets``.
        targets : tuple[SubsystemId, ...]
            Unique target subsystems.

        Returns
        -------
        StateRef
            State reference containing the updated payload.

        Raises
        ------
        TypeError
            If ``operation`` or ``targets`` has the wrong type.
        ValueError
            If targets are empty or duplicated.
        StateNotFoundError
            If a target subsystem is not owned.
        InvalidOperationError
            If arity or representation compatibility checks fail.
        DimensionError
            If a target axis is not a qubit axis.
        """
        operation = check_unitary(operation)
        checked_targets = self._check_subsystem_targets(targets)
        if len(checked_targets) != operation.arity:
            raise InvalidOperationError("target count must match operation arity")

        state_ref = self._owning_state_for_targets(checked_targets)
        record = self.record(state_ref)
        handler = self._registry.get(record.rep)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        payload = handler.apply(
            record.payload,
            operation,
            layout=record.layout,
            axes=axes,
        )
        self._replace_same_layout(state_ref, record, payload=payload)
        return state_ref

    def apply_noise(
        self,
        channel: NoiseChannel,
        *,
        targets: tuple[SubsystemId, ...],
    ) -> StateRef:
        """Apply a noise channel to target subsystems.

        Density records support general Kraus-channel application. Bell-diagonal
        records support Bell-diagonal-preserving Pauli channels. The manager
        does not auto-convert representations for this method; unsupported
        representation/channel pairs raise from their handlers.

        Parameters
        ----------
        channel : NoiseChannel
            Noise channel whose arity must match ``targets``.
        targets : tuple[SubsystemId, ...]
            Unique target subsystems.

        Returns
        -------
        StateRef
            State reference containing the updated density payload.

        Raises
        ------
        TypeError
            If ``channel`` or ``targets`` has the wrong type.
        ValueError
            If targets are empty or duplicated.
        StateNotFoundError
            If a target subsystem is not owned.
        NoiseError
            If arity fails or the representation cannot apply the channel.
        InvalidOperationError
            If separate states cannot be combined.
        DimensionError
            If a target axis is not a qubit axis.

        Examples
        --------
        >>> from simyuj.qstate import QuantumStateManager, SubsystemId
        >>> from simyuj.qstate.noise import amplitude_damping
        >>> q0 = SubsystemId("q0")
        >>> manager = QuantumStateManager()
        >>> state_ref = manager.prepare("|1>", rep="density", subsystems=(q0,))
        >>> _ = manager.apply_noise(amplitude_damping(1.0), targets=(q0,))
        >>> float(manager.get(state_ref).rho[0, 0].real)
        1.0
        """
        channel = check_noise_channel(channel)
        checked_targets = self._check_subsystem_targets(targets)
        if len(checked_targets) != channel.arity:
            raise NoiseError("target count must match noise channel arity")

        state_ref = self._owning_state_for_targets(checked_targets)
        record = self.record(state_ref)
        handler = self._registry.get(record.rep)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        payload = handler.channel(
            record.payload,
            channel,
            layout=record.layout,
            axes=axes,
        )
        self._replace_same_layout(state_ref, record, payload=payload)
        return state_ref

    def apply_noise_models(
        self,
        noise_models: Sequence[object],
        *,
        targets: tuple[SubsystemId, ...],
        auto_convert: bool = True,
        duration_s: float | None = None,
    ) -> StateRef:
        """Apply multiple noise channels to target subsystems in order.

        The manager's ``noise_mode`` controls ket handling. In ``"density"``
        mode, ket records are converted to density before the first channel
        when ``auto_convert`` is true. In ``"sampled_ket"`` mode, ket records
        sample one Kraus branch per channel using ``noise_rng`` and remain ket
        records; density records use exact channels, and Bell-diagonal records
        use compact exact Pauli updates where possible before falling back to
        density.

        Parameters
        ----------
        noise_models : sequence of NoiseChannel
            Noise channels to apply in order.
        targets : tuple[SubsystemId, ...]
            Target subsystems. The number of targets must match every channel arity.
        auto_convert : bool, default=True
            If true, convert the owning state to density representation before
            applying the channels. This option is used only in ``"density"``
            mode; ``"sampled_ket"`` mode keeps ket records pure by sampling.
        duration_s : float or None, optional
            Elapsed physical duration in seconds. Required only when a noise
            model needs time to resolve into a concrete NoiseChannel.

        Returns
        -------
        StateRef
            State reference containing the updated payload.
        """
        checked_targets = self._check_subsystem_targets(targets)
        channels = check_noise_models(
            noise_models,
            arity=len(checked_targets),
            field_name="noise_models",
            duration_s=duration_s,
        )

        state_ref = self._owning_state_for_targets(checked_targets)

        if not channels:
            return state_ref

        if self._noise_mode == "sampled_ket":
            return self._apply_noise_models_sampled_ket_policy(
                state_ref,
                channels,
                targets=checked_targets,
            )

        if auto_convert:
            self.convert(state_ref, "density")

        for channel in channels:
            state_ref = self.apply_noise(
                channel,
                targets=checked_targets,
            )

        return state_ref

    def _apply_noise_models_sampled_ket_policy(
        self,
        state_ref: StateRef,
        channels: tuple[NoiseChannel, ...],
        *,
        targets: tuple[SubsystemId, ...],
    ) -> StateRef:
        record = self.record(state_ref)
        if isinstance(record.payload, KetState):
            if self._noise_rng is None:
                raise ValueError("sampled ket noise requires noise_rng")
            for channel in channels:
                state_ref = self._apply_noise_sampled_ket(
                    channel,
                    targets=targets,
                    rng=self._noise_rng,
                )
            return state_ref

        if isinstance(record.payload, BellDiagState):
            for channel in channels:
                try:
                    state_ref = self.apply_noise(channel, targets=targets)
                except NoiseError as exc:
                    if "bell_diag supports Pauli channels only" not in str(exc):
                        raise
                    self.convert(state_ref, "density")
                    state_ref = self.apply_noise(channel, targets=targets)
            return state_ref

        for channel in channels:
            state_ref = self.apply_noise(channel, targets=targets)
        return state_ref

    def _apply_noise_sampled_ket(
        self,
        channel: NoiseChannel,
        *,
        targets: tuple[SubsystemId, ...],
        rng: Any,
    ) -> StateRef:
        state_ref = self._owning_state_for_targets(targets)
        record = self.record(state_ref)
        if not isinstance(record.payload, KetState):
            raise NoiseError("sampled ket noise requires ket representation")
        axes = resolve_targets(record.layout, targets)
        self._check_qubit_axes(record.layout, axes)
        handler = self._registry.get(record.rep)
        channel_sampled = getattr(handler, "channel_sampled", None)
        if not callable(channel_sampled):
            raise NoiseError("sampled ket noise requires ket representation")
        payload = channel_sampled(
            record.payload,
            channel,
            layout=record.layout,
            axes=axes,
            rng=rng,
        )
        self._replace_same_layout(state_ref, record, payload=payload)
        return state_ref

    def measure(
        self,
        *,
        targets: tuple[SubsystemId, ...],
        basis: str | MeasurementBasis = "z",
        rng: Any | None = None,
        collapse: bool = True,
    ) -> MeasurementResult:
        """Measure targets that already belong to one live state.

        Projective measurement does not combine separate states. When collapse
        is true and the handler returns a post-measurement state, the owning
        record is replaced in place.

        Parameters
        ----------
        targets : tuple[SubsystemId, ...]
            Unique target subsystems in one live state.
        basis : str or MeasurementBasis, optional
            Measurement basis passed to the representation handler.
        rng : Any or None, optional
            Explicit RNG stream used by stochastic measurement paths.
        collapse : bool, optional
            Whether to write back a collapsed post-measurement state.

        Returns
        -------
        MeasurementResult
            Measurement result annotated with state reference fields.

        Raises
        ------
        TypeError
            If ``targets`` has the wrong type.
        ValueError
            If targets are empty or duplicated.
        StateNotFoundError
            If a target subsystem is not owned.
        InvalidOperationError
            If targets span multiple live states.
        DimensionError
            If a target axis is not a qubit axis.
        """
        checked_targets = self._check_subsystem_targets(targets)
        state_ref = self._state_ref_for_same_state_targets(checked_targets)
        record = self.record(state_ref)
        handler = self._registry.get(record.rep)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        result = handler.measure(
            record.payload,
            layout=record.layout,
            axes=axes,
            basis=basis,
            rng=rng,
            collapse=collapse,
        )

        post_state_ref = None
        if result.collapsed and result.post_state is not None:
            self._replace_same_layout(state_ref, record, payload=result.post_state)
            post_state_ref = state_ref
        return result._with_refs(
            state_ref=state_ref,
            post_state_ref=post_state_ref,
        )

    def measure_and_discard(
        self,
        *,
        targets: tuple[SubsystemId, ...],
        basis: str | MeasurementBasis = "z",
        rng: Any | None = None,
        collapse: bool = True,
    ) -> MeasurementResult:
        """Measure targets and then trace those targets out of their state.

        This is equivalent to calling ``measure(...)`` followed by
        ``discard(targets=targets)``. It exists for detector-style workflows
        that immediately consume a measured signal, avoiding an intermediate
        manager write of the full collapsed state.
        """
        checked_targets = self._check_subsystem_targets(targets)
        state_ref = self._state_ref_for_same_state_targets(checked_targets)
        record = self.record(state_ref)
        handler = self._registry.get(record.rep)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        result = handler.measure(
            record.payload,
            layout=record.layout,
            axes=axes,
            basis=basis,
            rng=rng,
            collapse=collapse,
        )

        post_state_ref = None
        if result.collapsed and result.post_state is not None:
            payload_to_reduce = result.post_state
            post_state_ref = state_ref
        else:
            payload_to_reduce = record.payload

        remaining_layout = record.layout.without(checked_targets)
        if remaining_layout.size == 0:
            self.delete(state_ref)
        else:
            density_payload = as_rep(payload_to_reduce, "density")
            reduced_payload = discard_density(
                cast(DensityState, density_payload),
                drop_axes=axes,
            )
            self.replace(
                state_ref,
                replace(
                    record,
                    payload=reduced_payload,
                    rep="density",
                    layout=remaining_layout,
                ),
            )

        return result._with_refs(
            state_ref=state_ref,
            post_state_ref=post_state_ref,
        )

    def measure_povm(
        self,
        povm: POVM,
        *,
        targets: tuple[SubsystemId, ...],
        rng: Any | None = None,
        collapse: bool = True,
    ) -> POVMResult:
        """Measure targets with a POVM, converting the working payload to density.

        Targets may come from separate live states when they can be combined.
        The stored representation changes to density only when a collapsed
        post-state is written back.

        Parameters
        ----------
        povm : POVM
            POVM applied to the resolved target axes.
        targets : tuple[SubsystemId, ...]
            Unique target subsystems.
        rng : Any or None, optional
            Explicit RNG stream used for stochastic sampling.
        collapse : bool, optional
            Whether to write back a collapsed density post-state.

        Returns
        -------
        POVMResult
            POVM result annotated with state reference fields.

        Examples
        --------
        >>> import numpy as np
        >>> from simyuj.qstate import QuantumStateManager, SubsystemId
        >>> from simyuj.qstate.measure import POVM, POVMElement
        >>> q0 = SubsystemId("q0")
        >>> povm = POVM((
        ...     POVMElement("zero", np.diag([1.0, 0.0])),
        ...     POVMElement("one", np.diag([0.0, 1.0])),
        ... ), name="z")
        >>> manager = QuantumStateManager()
        >>> _ = manager.prepare("|0>", subsystems=(q0,))
        >>> manager.measure_povm(povm, targets=(q0,)).label
        'zero'
        """
        checked_targets = self._check_subsystem_targets(targets)
        state_ref = self._owning_state_for_targets(checked_targets)
        record = self.record(state_ref)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        density_payload = as_rep(record.payload, "density")
        result = measure_povm_density(
            cast(DensityState, density_payload),
            povm,
            axes=axes,
            rng=rng,
            collapse=collapse,
        )

        post_state_ref = None
        if result.collapsed and result.post_state is not None:
            self._replace_same_layout(
                state_ref,
                record,
                payload=result.post_state,
                rep="density",
            )
            post_state_ref = state_ref
        return result._with_refs(
            state_ref=state_ref,
            post_state_ref=post_state_ref,
        )

    def measure_bell(
        self,
        *,
        targets: tuple[SubsystemId, ...],
        rng: Any | None = None,
        collapse: bool = True,
    ) -> BellResult:
        """Measure exactly two targets in the Bell basis.

        Targets may come from separate live states when their representations
        can be combined. Collapsed post-states are written back to the owning
        record.

        Parameters
        ----------
        targets : tuple[SubsystemId, ...]
            Exactly two unique target subsystems.
        rng : Any or None, optional
            Explicit RNG stream used for stochastic sampling.
        collapse : bool, optional
            Whether to write back a collapsed post-state.

        Returns
        -------
        BellResult
            Bell measurement result annotated with state reference fields.

        Raises
        ------
        MeasurementError
            If the target count is not two.

        Examples
        --------
        >>> from simyuj.qstate import QuantumStateManager, SubsystemId
        >>> q0 = SubsystemId("q0")
        >>> q1 = SubsystemId("q1")
        >>> manager = QuantumStateManager()
        >>> _ = manager.prepare("phi+", subsystems=(q0, q1))
        >>> manager.measure_bell(targets=(q0, q1)).label
        'phi+'
        """
        checked_targets = self._check_subsystem_targets(targets)
        if len(checked_targets) != 2:
            raise MeasurementError("Bell measurement requires exactly two targets")

        state_ref = self._owning_state_for_targets(checked_targets)
        record = self.record(state_ref)
        handler = self._registry.get(record.rep)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        result = handler.measure_bell(
            record.payload,
            layout=record.layout,
            axes=cast(tuple[int, int], axes),
            rng=rng,
            collapse=collapse,
        )

        post_state_ref = None
        if result.collapsed and result.post_state is not None:
            self._replace_same_layout(state_ref, record, payload=result.post_state)
            post_state_ref = state_ref
        return result._with_refs(
            state_ref=state_ref,
            post_state_ref=post_state_ref,
        )

    def convert(
        self,
        state_ref: StateRef,
        rep: str,
        *,
        meta: MetaInput = None,
    ) -> StateRef:
        """Convert a live record to another representation in place.

        Parameters
        ----------
        state_ref : StateRef
            Live state reference.
        rep : str
            Target representation name.
        meta : MetaInput, optional
            Replacement metadata. If omitted, existing metadata is preserved.

        Returns
        -------
        StateRef
            The same state reference.

        Notes
        -----
        If the record is already in the requested representation, the method
        returns without changing payload or metadata.
        """
        normalized = normalize_rep(rep)
        record = self.record(state_ref)

        if record.rep == normalized:
            return state_ref

        converted = as_rep(record.payload, normalized)
        metadata = record.meta if meta is None else coerce_meta(meta)
        self._replace_same_layout(
            state_ref,
            record,
            payload=converted,
            rep=normalized,
            meta=metadata,
        )
        return state_ref

    def discard(self, *, targets: tuple[SubsystemId, ...]) -> StateRef | None:
        """Trace out target subsystems from one live state.

        The workflow converts the working payload to density form, removes the
        requested axes, and stores the reduced density record. If every
        subsystem is discarded, the state reference is deleted.

        Parameters
        ----------
        targets : tuple[SubsystemId, ...]
            Unique target subsystems that must belong to one live state.

        Returns
        -------
        StateRef or None
            Original state reference, or ``None`` when the record is deleted.

        Examples
        --------
        >>> from simyuj.qstate import QuantumStateManager, SubsystemId
        >>> q0 = SubsystemId("q0")
        >>> q1 = SubsystemId("q1")
        >>> manager = QuantumStateManager()
        >>> state_ref = manager.prepare("|00>", subsystems=(q0, q1))
        >>> manager.discard(targets=(q1,))
        0
        >>> manager.record(state_ref).layout.subsystems
        (SubsystemId(name='q0'),)
        """
        checked_targets = self._check_subsystem_targets(targets)
        state_ref = self._state_ref_for_same_state_targets(checked_targets)
        record = self.record(state_ref)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        remaining_layout = record.layout.without(checked_targets)
        if remaining_layout.size == 0:
            self.delete(state_ref)
            return None

        density_payload = as_rep(record.payload, "density")
        reduced_payload = discard_density(
            cast(DensityState, density_payload),
            drop_axes=axes,
        )
        self.replace(
            state_ref,
            replace(
                record,
                payload=reduced_payload,
                rep="density",
                layout=remaining_layout,
            ),
        )
        return state_ref

    def reset(
        self,
        *,
        targets: tuple[SubsystemId, ...],
        state: object = "|0>",
    ) -> StateRef:
        """Reset target subsystems inside one live state.

        The workflow converts the working payload to density form and writes the
        reset density payload back to the same record layout.

        Parameters
        ----------
        targets : tuple[SubsystemId, ...]
            Unique target subsystems that must belong to one live state.
        state : object, optional
            Representation-specific initializer for the reset state.

        Returns
        -------
        StateRef
            The same state reference.

        Examples
        --------
        >>> from simyuj.qstate import QuantumStateManager, SubsystemId
        >>> q0 = SubsystemId("q0")
        >>> manager = QuantumStateManager()
        >>> state_ref = manager.prepare("|1>", subsystems=(q0,))
        >>> manager.reset(targets=(q0,), state="|0>")
        0
        >>> float(manager.get(state_ref).rho[0, 0].real)
        1.0
        """
        checked_targets = self._check_subsystem_targets(targets)
        state_ref = self._state_ref_for_same_state_targets(checked_targets)
        record = self.record(state_ref)
        axes = resolve_targets(record.layout, checked_targets)
        self._check_qubit_axes(record.layout, axes)
        density_payload = as_rep(record.payload, "density")
        reset_payload = reset_density(
            cast(DensityState, density_payload),
            axes=axes,
            state=state,
        )
        self._replace_same_layout(
            state_ref,
            record,
            payload=reset_payload,
            rep="density",
        )
        return state_ref

    def _owning_state_for_targets(
        self,
        targets: tuple[SubsystemId, ...],
    ) -> StateRef:
        state_refs = tuple(self.state_of(target) for target in targets)
        unique_refs = tuple(sorted(set(state_refs)))
        if len(unique_refs) == 1:
            return unique_refs[0]
        return self._combine_states(unique_refs)

    def _combine_states(self, state_refs: tuple[StateRef, ...]) -> StateRef:
        ordered_refs = tuple(sorted(state_refs))
        records = tuple(self.record(state_ref) for state_ref in ordered_refs)
        first = records[0]
        rep = first.rep
        handler = self._registry.get(rep)
        layout = first.layout
        payload = first.payload

        for record in records[1:]:
            if record.rep != rep:
                raise InvalidOperationError(
                    "cannot combine states with different representations"
                )
            layout = layout.combine(record.layout)
            payload = handler.tensor(payload, record.payload)

        return self.consume_and_put(
            ordered_refs,
            QuantumStateRecord(payload=payload, rep=rep, layout=layout),
        )

    def _state_ref_for_same_state_targets(
        self,
        targets: tuple[SubsystemId, ...],
    ) -> StateRef:
        state_refs = tuple(self.state_of(target) for target in targets)
        unique_refs = set(state_refs)
        if len(unique_refs) != 1:
            raise InvalidOperationError("targets must belong to one live state")
        return state_refs[0]

    @staticmethod
    def _check_subsystem_targets(
        targets: tuple[SubsystemId, ...],
    ) -> tuple[SubsystemId, ...]:
        if not isinstance(targets, tuple):
            raise TypeError("targets must be tuple")
        if not targets:
            raise ValueError("targets must be non-empty")
        for target in targets:
            if not isinstance(target, SubsystemId):
                raise TypeError("targets must be SubsystemId")
        if len(set(targets)) != len(targets):
            raise ValueError("targets must be unique")
        return targets

    @staticmethod
    def _coerce_subsystems(
        subsystems: tuple[SubsystemId | str, ...],
    ) -> tuple[SubsystemId, ...]:
        if not isinstance(subsystems, tuple):
            raise TypeError("subsystems must be tuple")
        coerced = []
        for subsystem in subsystems:
            if isinstance(subsystem, SubsystemId):
                coerced.append(subsystem)
            elif isinstance(subsystem, str):
                coerced.append(SubsystemId(subsystem.strip()))
            else:
                raise TypeError("subsystems must be SubsystemId or str")
        return tuple(coerced)

    @classmethod
    def _prepare_layout(
        cls,
        payload: object,
        *,
        subsystems: tuple[SubsystemId | str, ...] | None,
        layout: StateLayout | None,
    ) -> StateLayout:
        if layout is not None and not isinstance(layout, StateLayout):
            raise TypeError("layout must be StateLayout or None")
        num_qubits = cls._payload_num_qubits(payload)
        if layout is None:
            if subsystems is None:
                raise ValueError("subsystems or layout must be provided")
            checked_subsystems = cls._coerce_subsystems(subsystems)
            if len(checked_subsystems) != num_qubits:
                raise DimensionError("layout does not match state payload")
            layout = StateLayout(
                checked_subsystems,
                qubit_dims(num_qubits),
            )
        try:
            _assert_payload_layout_compatible_trusted(payload, layout)
        except InvalidLayoutError as exc:
            if "qubit axes" in str(exc):
                raise DimensionError(
                    "qstate operations support qubit axes only"
                ) from exc
            raise DimensionError("layout does not match state payload") from exc
        return layout

    @staticmethod
    def _payload_num_qubits(payload: object) -> int:
        try:
            num_qubits = _payload_num_qubits_trusted(payload)
        except DimensionError as exc:
            if "integer num_qubits" in str(exc):
                raise TypeError("payload must expose integer num_qubits") from exc
            raise
        if num_qubits <= 0:
            raise DimensionError("payload num_qubits must be positive")
        return num_qubits

    @staticmethod
    def _check_qubit_axes(layout: StateLayout, axes: tuple[int, ...]) -> None:
        for axis in axes:
            if layout.dim_at(axis) != 2:
                raise DimensionError("qstate operations support qubit axes only")


__all__ = ["QuantumStateManager"]
