"""Registry for protocol-neutral entangled-pair records.

The registry owns pair lookup and lifecycle bookkeeping.  It enforces that an
active memory position is used by at most one active pair record, while keeping
historical terminal records queryable.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import (
    require_optional_probability,
    validate_non_negative_int,
)
from simyuj.resources.memory import MemoryRef

from .pair import EntangledPairRecord, PairState


class EntangledPairRegistry:
    """Protocol-neutral registry for known entangled pair records.

    The registry tracks pair lifecycle and lookup. It does not create quantum
    states, perform swapping, run purification, reserve memory slots, or submit
    runtime memory operations.

    Notes
    -----
    Pair records are returned in deterministic ``pair_id`` order by query
    methods. Active records may not share a memory position with another active
    record, but terminal records remain stored for historical lookup.
    Lifecycle methods replace the stored immutable record and return the
    replacement; any previously held ``EntangledPairRecord`` object is stale
    after a transition.
    """

    __slots__ = ("_pairs",)

    def __init__(self) -> None:
        self._pairs: dict[str, EntangledPairRecord] = {}

    @property
    def pairs(self) -> Mapping[str, EntangledPairRecord]:
        """Read-only live mapping of pair ID to pair record.

        Use tuple-returning query methods such as ``all_pairs()`` when callers
        need a deterministic snapshot.
        """

        return MappingProxyType(self._pairs)

    def register(self, pair: EntangledPairRecord) -> EntangledPairRecord:
        """Register a new pair record.

        Active pairs may not reuse a memory position already used by another
        active pair.

        Parameters
        ----------
        pair : EntangledPairRecord
            Pair record to add.

        Returns
        -------
        EntangledPairRecord
            The registered pair record.

        Raises
        ------
        TypeError
            If ``pair`` is not an ``EntangledPairRecord``.
        ValueError
            If the pair ID already exists or an active endpoint conflicts with
            another active pair.
        """

        if not isinstance(pair, EntangledPairRecord):
            raise TypeError("pair must be EntangledPairRecord")

        if pair.pair_id in self._pairs:
            raise ValueError(f"pair id '{pair.pair_id}' already exists")

        self._ensure_no_active_memory_conflict(pair)

        self._pairs[pair.pair_id] = pair
        return pair

    def get(self, pair_id: str) -> EntangledPairRecord:
        """Return the pair record for ``pair_id``.

        Raises
        ------
        KeyError
            If ``pair_id`` is unknown.
        """

        resolved_pair_id = ensure_nonempty_id(pair_id, field_name="pair_id")

        if resolved_pair_id not in self._pairs:
            raise KeyError(f"unknown pair id '{resolved_pair_id}'")

        return self._pairs[resolved_pair_id]

    def all_pairs(
        self,
        *,
        state: PairState | None = None,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return pair records in deterministic pair-id order.

        Parameters
        ----------
        state : PairState or None, optional
            Optional lifecycle-state filter.

        Returns
        -------
        tuple[EntangledPairRecord, ...]
            Matching records sorted by pair ID.
        """

        if state is not None and not isinstance(state, PairState):
            raise TypeError("state must be PairState or None")

        return tuple(
            pair for pair in self._pairs_by_id() if state is None or pair.state is state
        )

    def active_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return available and reserved pairs in deterministic pair-id order."""

        return tuple(pair for pair in self._pairs_by_id() if pair.is_active)

    def available_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs in deterministic pair-id order."""

        return self.all_pairs(state=PairState.AVAILABLE)

    def reserved_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return reserved pairs in deterministic pair-id order."""

        return self.all_pairs(state=PairState.RESERVED)

    def available_between(
        self,
        first_node_id: str,
        second_node_id: str,
        *,
        min_fidelity: float | None = None,
        link_id: str | None = None,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs connecting two nodes, independent of direction.

        Pairs without a fidelity estimate are excluded when ``min_fidelity`` is
        provided.

        Parameters
        ----------
        first_node_id, second_node_id : str
            Node IDs that the pair must connect.
        min_fidelity : float or None, optional
            Optional minimum fidelity in ``[0, 1]``.
        link_id : str or None, optional
            Optional identifier for the generation link. If provided, only pairs
            with a matching ``generation_link_id`` are returned.

        Returns
        -------
        tuple[EntangledPairRecord, ...]
            Available pairs sorted by pair ID.
        """

        first = ensure_nonempty_id(first_node_id, field_name="first_node_id")
        second = ensure_nonempty_id(second_node_id, field_name="second_node_id")
        resolved_min_fidelity = require_optional_probability(
            min_fidelity,
            field_name="min_fidelity",
        )
        resolved_link_id = (
            None
            if link_id is None
            else ensure_nonempty_id(link_id, field_name="link_id")
        )

        pairs: list[EntangledPairRecord] = []

        for pair in self._pairs_by_id():
            if not pair.is_available or not pair.connects_nodes(first, second):
                continue

            if resolved_min_fidelity is not None and (
                pair.fidelity is None or pair.fidelity < resolved_min_fidelity
            ):
                continue

            if (
                resolved_link_id is not None
                and pair.generation_link_id != resolved_link_id
            ):
                continue

            pairs.append(pair)

        return tuple(pairs)

    def available_for_memory_refs(
        self,
        first: MemoryRef,
        second: MemoryRef,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs connecting two exact memory positions.

        The left/right order of the pair record is ignored.
        """

        resolved_first = self._validate_memory_ref(first, field_name="first")
        resolved_second = self._validate_memory_ref(second, field_name="second")

        if resolved_first == resolved_second:
            raise ValueError("first and second memory refs must differ")

        return tuple(
            pair
            for pair in self._pairs_by_id()
            if pair.is_available
            and pair.connects_memory_refs(resolved_first, resolved_second)
        )

    def pair_using_memory(self, memory_ref: MemoryRef) -> EntangledPairRecord | None:
        """Return the active pair using ``memory_ref``, if any.

        Because the registry enforces one active pair per memory position, this
        returns at most one pair.
        """

        ref = self._validate_memory_ref(memory_ref, field_name="memory_ref")

        for pair in self._pairs_by_id():
            if pair.is_active and pair.uses_memory(ref):
                return pair

        return None

    def pairs_using_memory(
        self,
        memory_ref: MemoryRef,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return all historical records using ``memory_ref``.

        This can include terminal records.
        """

        ref = self._validate_memory_ref(memory_ref, field_name="memory_ref")

        return tuple(pair for pair in self._pairs_by_id() if pair.uses_memory(ref))

    def reserve(self, pair_id: str) -> EntangledPairRecord:
        """Mark an available pair as reserved.

        Raises
        ------
        ValueError
            If the pair is not currently available.
        """

        pair = self.get(pair_id)

        if pair.state is not PairState.AVAILABLE:
            raise ValueError("only available pairs can be reserved")

        return self._replace(pair.reserved())

    def release(self, pair_id: str) -> EntangledPairRecord:
        """Return a reserved pair to available state.

        Raises
        ------
        ValueError
            If the pair is not currently reserved.
        """

        pair = self.get(pair_id)

        if pair.state is not PairState.RESERVED:
            raise ValueError("only reserved pairs can be released")

        return self._replace(pair.available())

    def consume(self, pair_id: str) -> EntangledPairRecord:
        """Mark an active pair as consumed.

        Raises
        ------
        ValueError
            If the pair is already terminal.
        """

        pair = self.get(pair_id)

        if not pair.is_active:
            raise ValueError("only active pairs can be consumed")

        return self._replace(pair.consumed())

    def expire(self, pair_id: str) -> EntangledPairRecord:
        """Mark an active pair as expired.

        Raises
        ------
        ValueError
            If the pair is already terminal.
        """

        pair = self.get(pair_id)

        if not pair.is_active:
            raise ValueError("only active pairs can be expired")

        return self._replace(pair.expired())

    def fail(self, pair_id: str) -> EntangledPairRecord:
        """Mark an active pair as failed.

        Raises
        ------
        ValueError
            If the pair is already terminal.
        """

        pair = self.get(pair_id)

        if not pair.is_active:
            raise ValueError("only active pairs can be failed")

        return self._replace(pair.failed())

    def expire_before(self, now: int) -> tuple[EntangledPairRecord, ...]:
        """Expire active pairs whose ``expires_at`` time has been reached.

        A pair expires when ``expires_at <= now``.

        Parameters
        ----------
        now : int
            Non-negative current simulation tick.

        Returns
        -------
        tuple[EntangledPairRecord, ...]
            Records that were transitioned to ``PairState.EXPIRED``.
        """

        validate_non_negative_int(now, field_name="now")

        expired: list[EntangledPairRecord] = []

        for pair in self._pairs_by_id():
            if (
                pair.is_active
                and pair.expires_at is not None
                and pair.expires_at <= now
            ):
                expired.append(self._replace(pair.expired()))

        return tuple(expired)

    def _replace(self, pair: EntangledPairRecord) -> EntangledPairRecord:
        if not isinstance(pair, EntangledPairRecord):
            raise TypeError("pair must be EntangledPairRecord")

        if pair.pair_id not in self._pairs:
            raise KeyError(f"unknown pair id '{pair.pair_id}'")

        self._ensure_no_active_memory_conflict(
            pair,
            exclude_pair_id=pair.pair_id,
        )

        self._pairs[pair.pair_id] = pair
        return pair

    def _ensure_no_active_memory_conflict(
        self,
        pair: EntangledPairRecord,
        *,
        exclude_pair_id: str | None = None,
    ) -> None:
        if not pair.is_active:
            return

        for existing in self._pairs_by_id():
            if exclude_pair_id is not None and existing.pair_id == exclude_pair_id:
                continue

            if not existing.is_active:
                continue

            for memory_ref in pair.memory_refs:
                if existing.uses_memory(memory_ref):
                    raise ValueError(
                        f"memory ref {memory_ref.key!r} is already used by "
                        f"active pair '{existing.pair_id}'"
                    )

    def _pairs_by_id(self) -> tuple[EntangledPairRecord, ...]:
        return tuple(self._pairs[pair_id] for pair_id in sorted(self._pairs))

    @staticmethod
    def _validate_memory_ref(
        memory_ref: MemoryRef,
        *,
        field_name: str,
    ) -> MemoryRef:
        if not isinstance(memory_ref, MemoryRef):
            raise TypeError(f"{field_name} must be MemoryRef")

        return memory_ref


__all__ = [
    "EntangledPairRegistry",
]
