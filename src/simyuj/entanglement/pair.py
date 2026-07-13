"""Protocol-neutral records for entangled pairs stored in memories.

This module describes pair identity, endpoint memory references, lifecycle
state, and optional metadata.  It deliberately does not store quantum-state
objects or implement entanglement-generation physics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import (
    require_optional_probability,
    validate_optional_non_negative_int,
)
from simyuj.resources.memory import MemoryRef


class PairState(Enum):
    """Resource-layer lifecycle state for a known entangled pair.

    This does not describe the quantum state itself. It only describes whether
    the pair record is usable by higher-level protocol code.
    """

    AVAILABLE = "available"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EntangledPairRecord:
    """Protocol-neutral record for an entangled pair stored in memory positions.

    The record stores where the pair lives and optional lifecycle metadata. It
    must not store qstate objects, density matrices, qubits, photons, or
    protocol-specific controller state.

    Parameters
    ----------
    pair_id : str
        Stable identifier for this entangled-pair record.
    left, right : MemoryRef
        Distinct memory positions containing the pair endpoints. Left and right
        are labels for record storage; node-connection checks treat the pair as
        undirected.
    state : PairState, optional
        Bookkeeping lifecycle state.
    fidelity : float or None, optional
        Optional probability-like fidelity estimate in ``[0, 1]``.
    created_at : int or None, optional
        Optional non-negative creation tick.
    expires_at : int or None, optional
        Optional non-negative expiration tick. When both times are present,
        ``expires_at`` must not be earlier than ``created_at``.
    generation_link_id : str or None, optional
        Optional link identifier that produced the pair.
    left_occupancy_token : int or None, optional
        Optional non-negative token identifying the specific occupancy of the
        left memory position. When present, distinguishes the current pair
        from historical pairs that reused the same physical slot.
    right_occupancy_token : int or None, optional
        Same as ``left_occupancy_token`` for the right endpoint.
    metadata : tuple[tuple[str, object], ...], optional
        Metadata entries for traceability.

    Raises
    ------
    TypeError
        If endpoint references, lifecycle state, metadata, or identifiers use
        unsupported types.
    ValueError
        If identifiers are empty, fidelity is outside ``[0, 1]``, endpoints are
        the same memory reference, or times are inconsistent.

    Notes
    -----
    State helpers such as ``reserved()``, ``available()``, and ``consumed()``
    only create replacement records. Legal lifecycle transitions are enforced
    by ``EntangledPairRegistry``.

    ``metadata`` shape and keys are validated, but metadata values are stored by
    reference. Mutable values are not deep-frozen or hashability-checked.

    Examples
    --------
        >>> left = MemoryRef("alice", "qmem", 0)
        >>> right = MemoryRef("bob", "qmem", 0)
        >>> pair = EntangledPairRecord("pair:0", left, right, fidelity=0.95)
        >>> pair.node_ids
        ('alice', 'bob')
    """

    pair_id: str
    left: MemoryRef
    right: MemoryRef
    state: PairState = PairState.AVAILABLE
    fidelity: float | None = None
    created_at: int | None = None
    expires_at: int | None = None
    generation_link_id: str | None = None
    left_occupancy_token: int | None = None
    right_occupancy_token: int | None = None
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pair_id",
            ensure_nonempty_id(self.pair_id, field_name="pair_id"),
        )
        object.__setattr__(
            self,
            "fidelity",
            require_optional_probability(
                self.fidelity,
                field_name="fidelity",
            ),
        )
        object.__setattr__(
            self,
            "generation_link_id",
            self._resolve_optional_id(
                self.generation_link_id,
                field_name="generation_link_id",
            ),
        )

        self._validate_memory_ref(self.left, field_name="left")
        self._validate_memory_ref(self.right, field_name="right")

        if self.left == self.right:
            raise ValueError("left and right memory refs must differ")

        if not isinstance(self.state, PairState):
            raise TypeError("state must be PairState")

        validate_optional_non_negative_int(
            self.created_at,
            field_name="created_at",
        )
        validate_optional_non_negative_int(
            self.expires_at,
            field_name="expires_at",
        )

        validate_optional_non_negative_int(
            self.left_occupancy_token,
            field_name="left_occupancy_token",
        )
        validate_optional_non_negative_int(
            self.right_occupancy_token,
            field_name="right_occupancy_token",
        )

        if (
            self.created_at is not None
            and self.expires_at is not None
            and self.expires_at < self.created_at
        ):
            raise ValueError("expires_at cannot be earlier than created_at")

        self._validate_metadata(self.metadata)

    @property
    def is_available(self) -> bool:
        """Return whether this pair can be selected by query helpers."""

        return self.state is PairState.AVAILABLE

    @property
    def is_active(self) -> bool:
        """Return whether this pair still occupies usable lifecycle state."""

        return self.state in (PairState.AVAILABLE, PairState.RESERVED)

    @property
    def is_terminal(self) -> bool:
        """Return whether this pair is consumed, expired, or failed."""

        return self.state in (
            PairState.CONSUMED,
            PairState.EXPIRED,
            PairState.FAILED,
        )

    @property
    def has_fidelity(self) -> bool:
        """Return whether a fidelity estimate is attached to this record."""

        return self.fidelity is not None

    @property
    def memory_refs(self) -> tuple[MemoryRef, MemoryRef]:
        """Return the pair endpoint references as ``(left, right)``."""

        return (self.left, self.right)

    @property
    def memory_ref_keys(self) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
        """Return tuple keys for the left and right endpoint references."""

        return (self.left.key, self.right.key)

    @property
    def node_ids(self) -> tuple[str, str]:
        """Return endpoint node identifiers as ``(left.node_id, right.node_id)``."""

        return (self.left.node_id, self.right.node_id)

    def uses_memory(self, memory_ref: MemoryRef) -> bool:
        """Return whether ``memory_ref`` is one of this pair's endpoints."""

        self._validate_memory_ref(memory_ref, field_name="memory_ref")
        return memory_ref == self.left or memory_ref == self.right

    def other_memory(self, memory_ref: MemoryRef) -> MemoryRef:
        """Return the endpoint opposite ``memory_ref``.

        Raises
        ------
        ValueError
            If ``memory_ref`` is not part of this pair.
        """

        self._validate_memory_ref(memory_ref, field_name="memory_ref")

        if memory_ref == self.left:
            return self.right

        if memory_ref == self.right:
            return self.left

        raise ValueError(f"memory ref {memory_ref.key!r} is not part of pair")

    def has_node(self, node_id: str) -> bool:
        """Return whether either endpoint is stored at ``node_id``."""

        resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")
        return (
            self.left.node_id == resolved_node_id
            or self.right.node_id == resolved_node_id
        )

    def connects_nodes(self, first_node_id: str, second_node_id: str) -> bool:
        """Return whether this pair connects two node IDs, ignoring direction."""

        first = ensure_nonempty_id(first_node_id, field_name="first_node_id")
        second = ensure_nonempty_id(second_node_id, field_name="second_node_id")

        return (self.left.node_id == first and self.right.node_id == second) or (
            self.left.node_id == second and self.right.node_id == first
        )

    def connects_memory_refs(
        self,
        first: MemoryRef,
        second: MemoryRef,
    ) -> bool:
        """Return whether this pair connects two exact memory refs."""

        self._validate_memory_ref(first, field_name="first")
        self._validate_memory_ref(second, field_name="second")

        return (self.left == first and self.right == second) or (
            self.left == second and self.right == first
        )

    def with_state(self, state: PairState) -> EntangledPairRecord:
        """Return a copy of this pair record with ``state`` applied."""

        if not isinstance(state, PairState):
            raise TypeError("state must be PairState")

        return replace(self, state=state)

    def available(self) -> EntangledPairRecord:
        """Return a copy marked available."""

        return self.with_state(PairState.AVAILABLE)

    def reserved(self) -> EntangledPairRecord:
        """Return a copy marked reserved."""

        return self.with_state(PairState.RESERVED)

    def consumed(self) -> EntangledPairRecord:
        """Return a copy marked consumed."""

        return self.with_state(PairState.CONSUMED)

    def expired(self) -> EntangledPairRecord:
        """Return a copy marked expired."""

        return self.with_state(PairState.EXPIRED)

    def failed(self) -> EntangledPairRecord:
        """Return a copy marked failed."""

        return self.with_state(PairState.FAILED)

    @staticmethod
    def _validate_memory_ref(memory_ref: MemoryRef, *, field_name: str) -> None:
        if not isinstance(memory_ref, MemoryRef):
            raise TypeError(f"{field_name} must be MemoryRef")

    @staticmethod
    def _resolve_optional_id(
        value: str | None,
        *,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        return ensure_nonempty_id(value, field_name=field_name)

    @staticmethod
    def _validate_metadata(metadata: tuple[tuple[str, object], ...]) -> None:
        if not isinstance(metadata, tuple):
            raise TypeError("metadata must be tuple[tuple[str, object], ...]")

        for item in metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("metadata must contain only two-item tuple entries")

            key, _value = item
            ensure_nonempty_id(key, field_name="metadata key")


__all__ = [
    "EntangledPairRecord",
    "PairState",
]
