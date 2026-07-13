"""Protocol-neutral memory resource manager.

``ResourceManager`` keeps resource-layer slot state and reservation ownership
separate from ``QuantumMemory`` runtime operations.  It is deterministic
bookkeeping only: callers still submit memory operations through the component
and timeline layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from simyuj.components.memories import (
    MemoryPositionRecord,
    MemoryPositionStatus,
    QuantumMemory,
)
from simyuj.network import Network
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import (
    validate_non_negative_int,
    validate_positive_int,
)

from .memory import MemoryRef, MemorySlotState, MemorySlotView, memory_refs
from .reservation import Reservation, ReservationState


class UnauthorizedError(Exception):
    """Raised when a caller does not own the requested reservation."""


class ResourceManager:
    """Protocol-neutral memory resource bookkeeping.

    The manager tracks resource-layer slot state and reservations. It does not
    absorb photons, emit photons, apply operators, measure memories, create
    entanglement, or submit runtime memory requests.

    Notes
    -----
    Reservation state and physical slot state are intentionally distinct.
    Committing a reservation marks ownership handoff only; callers must use
    :meth:`mark_occupied`, :meth:`mark_consumed`, :meth:`mark_expired`,
    :meth:`mark_failed`, or :meth:`mark_free` to mirror physical memory
    lifecycle changes.
    """

    __slots__ = (
        "_slots",
        "_reservations",
        "_memory_holders",
        "_next_reservation_index",
    )

    def __init__(self) -> None:
        self._slots: dict[MemoryRef, MemorySlotView] = {}
        self._reservations: dict[str, Reservation] = {}
        self._memory_holders: dict[MemoryRef, str] = {}
        self._next_reservation_index = 0

    @property
    def slots(self) -> Mapping[MemoryRef, MemorySlotView]:
        """Read-only live mapping of registered memory refs to slot views."""

        return MappingProxyType(self._slots)

    @property
    def reservations(self) -> Mapping[str, Reservation]:
        """Read-only live mapping of reservation ID to reservation record."""

        return MappingProxyType(self._reservations)

    @classmethod
    def from_network(cls, network: Network) -> ResourceManager:
        """Build a manager by scanning node-local ``QuantumMemory`` devices.

        Each node device name becomes the MemoryRef.device_id. The underlying
        QuantumMemory.memory_id is kept only as metadata.

        This is a one-time scan of the current network. Later component state
        changes are not mirrored unless callers update the manager explicitly.

        Parameters
        ----------
        network : Network
            Network whose nodes are scanned for ``QuantumMemory`` devices.

        Returns
        -------
        ResourceManager
            Manager populated with one slot per memory position.

        Raises
        ------
        TypeError
            If ``network`` is not a ``Network``.
        ValueError
            If a memory device reports a position count inconsistent with its
            configured ``num_positions`` or exposes an unsupported position
            status.
        """

        if not isinstance(network, Network):
            raise TypeError("network must be Network")

        manager = cls()

        for node_id in sorted(network.nodes):
            node = network.nodes[node_id]

            for device_id in sorted(node.devices):
                device = node.devices[device_id]

                if not isinstance(device, QuantumMemory):
                    continue

                refs = manager.register_memory(
                    node_id=node_id,
                    device_id=device_id,
                    num_positions=device.num_positions,
                    metadata=(("memory_id", device.memory_id),),
                )

                if len(device.positions) != len(refs):
                    raise ValueError(
                        f"QuantumMemory '{device.memory_id}' position count "
                        "does not match num_positions"
                    )

                for position_record in device.positions:
                    ref = MemoryRef(
                        node_id=node_id,
                        device_id=device_id,
                        position=position_record.position,
                    )
                    manager._replace_slot_from_position_record(ref, position_record)

        return manager

    def register_memory(
        self,
        node_id: str,
        device_id: str,
        *,
        num_positions: int,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> tuple[MemoryRef, ...]:
        """Register one node-local memory device as addressable memory slots.

        Parameters
        ----------
        node_id, device_id : str
            Network node and node-local memory-device identifiers.
        num_positions : int
            Number of memory positions to register.
        metadata : tuple[tuple[str, object], ...], optional
            Metadata copied onto every created ``MemorySlotView``.

        Returns
        -------
        tuple[MemoryRef, ...]
            Registered memory refs ordered by position.

        Raises
        ------
        ValueError
            If any generated memory ref is already registered.
        """

        refs = memory_refs(
            node_id=node_id,
            device_id=device_id,
            num_positions=num_positions,
        )
        self._validate_metadata(metadata)

        for ref in refs:
            if ref in self._slots:
                raise ValueError(f"memory ref {ref.key!r} is already registered")

        for ref in refs:
            self._slots[ref] = MemorySlotView(
                ref=ref,
                state=MemorySlotState.FREE,
                metadata=metadata,
            )

        return refs

    def get_slot(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Return the current slot view for ``memory_ref``.

        Raises
        ------
        KeyError
            If the memory ref has not been registered.
        """

        ref = self._require_registered_memory_ref(memory_ref)
        return self._slots[ref]

    def registered_memories(
        self,
        node_id: str | None = None,
        *,
        device_id: str | None = None,
    ) -> tuple[MemoryRef, ...]:
        """Return registered memory refs filtered by optional node or device.

        Results are sorted by ``(node_id, device_id, position)``.
        """

        resolved_node_id = self._resolve_optional_id(node_id, field_name="node_id")
        resolved_device_id = self._resolve_optional_id(
            device_id,
            field_name="device_id",
        )

        return tuple(
            ref
            for ref in sorted(self._slots, key=lambda item: item.key)
            if self._matches_ref(
                ref,
                node_id=resolved_node_id,
                device_id=resolved_device_id,
            )
        )

    def available_memories(
        self,
        now: int,
        node_id: str | None = None,
        *,
        device_id: str | None = None,
        link_id: str | None = None,
    ) -> tuple[MemoryRef, ...]:
        """Return physically free, unheld memory refs filtered by optional node or
        device.

        A memory is available if it is logically FREE in the ledger and its
        recovery ready_at time is None or <= now. If a link_id is provided, only
        slots that explicitly have ``("link_id", link_id)`` in their metadata
        are returned.

        Results follow the same deterministic ordering as
        :meth:`registered_memories`.
        """
        validate_non_negative_int(now, field_name="now")

        resolved_node_id = self._resolve_optional_id(node_id, field_name="node_id")
        resolved_device_id = self._resolve_optional_id(
            device_id,
            field_name="device_id",
        )
        resolved_link_id = (
            None
            if link_id is None
            else ensure_nonempty_id(link_id, field_name="link_id")
        )

        return tuple(
            ref
            for ref in self.registered_memories(
                resolved_node_id,
                device_id=resolved_device_id,
            )
            if self._is_available(ref, now)
            and (
                resolved_link_id is None
                or ("link_id", resolved_link_id) in self._slots[ref].metadata
            )
        )

    def reserve_memories(
        self,
        now: int,
        requirements: Mapping[str, int | Mapping[str, int]],
        *,
        owner: str,
        reservation_id: str | None = None,
        created_at: int | None = None,
        expires_at: int | None = None,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> Reservation:
        """Reserve a deterministic set of physically free memory slots by node
        requirement.

        Requirements are sorted by node ID before selection. Within each node,
        the lowest sorted available memory refs are selected.

        Parameters
        ----------
        now : int
            Current timeline tick used to evaluate physical hardware readiness.
        requirements : Mapping[str, int | Mapping[str, int]]
            Mapping from node ID to either a positive number of memory slots
            or a mapping from device ID to positive number of slots.
        owner : str
            Owner recorded on the reservation.
        reservation_id : str or None, optional
            Explicit reservation ID. When omitted, a deterministic
            ``reservation:<n>`` ID is generated.
        created_at, expires_at : int or None, optional
            Optional non-negative reservation times.
        metadata : tuple[tuple[str, object], ...], optional
            Metadata attached to the reservation.

        Returns
        -------
        Reservation
            Active reservation holding the selected memory refs.

        Examples
        --------
        >>> manager = ResourceManager()
        >>> manager.register_memory("alice", "qmem", num_positions=1)
        (MemoryRef(node_id='alice', device_id='qmem', position=0),)
        >>> reservation = manager.reserve_memories(10, {"alice": 1}, owner="demo")
        >>> reservation.memory_ref_keys
        (('alice', 'qmem', 0),)
        """

        resolved_requirements = self._resolve_requirements(requirements)
        selected_refs: list[MemoryRef] = []

        for node_id, req_value in resolved_requirements:
            self._require_registered_node(node_id)

            if isinstance(req_value, tuple):
                for device_id, count in req_value:
                    available_refs = self.available_memories(
                        now, node_id, device_id=device_id
                    )
                    if len(available_refs) < count:
                        raise ValueError(
                            f"device '{device_id}' on node '{node_id}' has "
                            f"{len(available_refs)} available memory slot(s), "
                            f"but {count} requested"
                        )
                    selected_refs.extend(available_refs[:count])
            else:
                available_refs = self.available_memories(now, node_id)
                if len(available_refs) < req_value:
                    raise ValueError(
                        f"node '{node_id}' has {len(available_refs)} available "
                        f"memory slot(s), but {req_value} requested"
                    )
                selected_refs.extend(available_refs[:req_value])

        return self.reserve_memory_refs(
            now,
            tuple(selected_refs),
            owner=owner,
            reservation_id=reservation_id,
            created_at=created_at,
            expires_at=expires_at,
            metadata=metadata,
        )

    def reserve_memory_refs(
        self,
        now: int,
        memory_refs: tuple[MemoryRef, ...],
        *,
        owner: str,
        reservation_id: str | None = None,
        created_at: int | None = None,
        expires_at: int | None = None,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> Reservation:
        """Reserve exact memory refs.

        This is useful when routing or user protocol code has already selected
        concrete memory positions.

        Parameters
        ----------
        now : int
            Current timeline tick used to evaluate physical hardware readiness.
        memory_refs : tuple[MemoryRef, ...]
            Non-empty tuple of registered, available memory refs.
        owner : str
            Owner recorded on the reservation.
        reservation_id : str or None, optional
            Explicit reservation ID, or ``None`` for generated IDs.
        created_at, expires_at : int or None, optional
            Optional non-negative reservation times.
        metadata : tuple[tuple[str, object], ...], optional
            Metadata attached to the reservation.

        Returns
        -------
        Reservation
            Active reservation holding exactly the supplied memory refs.
        """

        resolved_memory_refs = self._resolve_memory_refs(memory_refs)
        self._validate_metadata(metadata)

        if reservation_id is not None:
            resolved_reservation_id = self._resolve_reservation_id(reservation_id)

            if resolved_reservation_id in self._reservations:
                raise ValueError(
                    f"reservation id '{resolved_reservation_id}' already exists"
                )
        else:
            resolved_reservation_id = None

        for ref in resolved_memory_refs:
            self._require_available(ref, now)

        if resolved_reservation_id is None:
            resolved_reservation_id = self._resolve_reservation_id(None)

        reservation = Reservation(
            reservation_id=resolved_reservation_id,
            memory_refs=resolved_memory_refs,
            link_ids=(),
            owner=owner,
            created_at=created_at,
            expires_at=expires_at,
            state=ReservationState.ACTIVE,
            metadata=metadata,
        )

        for ref in resolved_memory_refs:
            self._replace_slot_state(ref, MemorySlotState.RESERVED)
            self._memory_holders[ref] = reservation.reservation_id

        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def get_reservation(self, reservation_id: str) -> Reservation:
        """Return a reservation by ID.

        Raises
        ------
        KeyError
            If the reservation ID is unknown.
        """

        resolved_reservation_id = ensure_nonempty_id(
            reservation_id,
            field_name="reservation_id",
        )

        if resolved_reservation_id not in self._reservations:
            raise KeyError(f"unknown reservation id '{resolved_reservation_id}'")

        return self._reservations[resolved_reservation_id]

    def reservation_for_memory(self, memory_ref: MemoryRef) -> Reservation | None:
        """Return the reservation currently holding ``memory_ref``, if any."""

        ref = self._require_registered_memory_ref(memory_ref)
        reservation_id = self._memory_holders.get(ref)

        if reservation_id is None:
            return None

        try:
            return self._reservations[reservation_id]
        except KeyError as exc:
            raise RuntimeError(
                f"memory ref {ref.key!r} is held by unknown reservation "
                f"{reservation_id!r}"
            ) from exc

    def commit_reservation(self, reservation_id: str, *, owner: str) -> Reservation:
        """Mark a reservation as handed off to user/runtime protocol code.

        This does not mark memory as occupied. Use mark_occupied(...) when the
        underlying memory slot is actually used.
        """

        reservation = self.get_reservation(reservation_id)

        if reservation.owner != owner:
            raise UnauthorizedError(
                f"unauthorized: caller '{owner}' does not own reservation "
                f"'{reservation_id}'"
            )

        if reservation.state is not ReservationState.ACTIVE:
            raise ValueError("only active reservations can be committed")

        committed = reservation.committed()
        self._reservations[committed.reservation_id] = committed
        return committed

    def release_reservation(self, reservation_id: str, *, owner: str) -> Reservation:
        """Release an active or committed reservation.

        Only slots still in ``RESERVED`` become ``FREE``. Slots already marked
        ``OCCUPIED``, ``CONSUMED``, ``EXPIRED``, or ``FAILED`` keep their
        resource-layer state.
        """

        reservation = self.get_reservation(reservation_id)
        if reservation.owner != owner:
            raise UnauthorizedError(
                f"unauthorized: caller '{owner}' does not own reservation "
                f"'{reservation_id}'"
            )

        return self._close_reservation(
            reservation_id,
            state=ReservationState.RELEASED,
        )

    def cancel_reservation(self, reservation_id: str, *, owner: str) -> Reservation:
        """Cancel an active or committed reservation.

        Only slots still in ``RESERVED`` become ``FREE``. Slots already marked
        ``OCCUPIED``, ``CONSUMED``, ``EXPIRED``, or ``FAILED`` keep their
        resource-layer state.
        """

        reservation = self.get_reservation(reservation_id)
        if reservation.owner != owner:
            raise UnauthorizedError(
                f"unauthorized: caller '{owner}' does not own reservation "
                f"'{reservation_id}'"
            )

        return self._close_reservation(
            reservation_id,
            state=ReservationState.CANCELLED,
        )

    def expire_reservation(self, reservation_id: str) -> Reservation:
        """Expire an active or committed reservation.

        Only slots still in ``RESERVED`` become ``FREE``. Slots already marked
        ``OCCUPIED``, ``CONSUMED``, ``EXPIRED``, or ``FAILED`` keep their
        resource-layer state.
        """

        return self._close_reservation(
            reservation_id,
            state=ReservationState.EXPIRED,
        )

    def expire_before(self, now: int) -> tuple[Reservation, ...]:
        """Expire active/committed reservations with expires_at <= now.

        Returns
        -------
        tuple[Reservation, ...]
            The expired reservations, ordered deterministically by ID.
        """
        validate_non_negative_int(now, field_name="now")

        expired = []
        for r_id in sorted(self._reservations.keys()):
            res = self._reservations[r_id]
            if res.state in (ReservationState.ACTIVE, ReservationState.COMMITTED):
                if res.expires_at is not None and res.expires_at <= now:
                    expired.append(self.expire_reservation(r_id))

        return tuple(expired)

    def mark_occupied(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a free or reserved slot as physically occupied.

        A free slot may be marked occupied without creating a reservation. Use
        that path when mirroring physical memory state discovered outside the
        reservation flow.
        """

        ref = self._require_registered_memory_ref(memory_ref)
        current_state = self._slots[ref].state

        if current_state not in (MemorySlotState.FREE, MemorySlotState.RESERVED):
            raise ValueError(
                f"cannot mark memory ref {ref.key!r} occupied from "
                f"state {current_state.value!r}"
            )

        return self._replace_slot_state(ref, MemorySlotState.OCCUPIED)

    def mark_consumed(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark an occupied slot as consumed."""

        ref = self._require_registered_memory_ref(memory_ref)
        current_state = self._slots[ref].state

        if current_state is not MemorySlotState.OCCUPIED:
            raise ValueError(
                f"cannot consume memory ref {ref.key!r} from "
                f"state {current_state.value!r}"
            )

        return self._replace_slot_state(ref, MemorySlotState.CONSUMED)

    def mark_expired(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a reserved or occupied slot as expired."""

        ref = self._require_registered_memory_ref(memory_ref)
        current_state = self._slots[ref].state

        if current_state not in (
            MemorySlotState.RESERVED,
            MemorySlotState.OCCUPIED,
        ):
            raise ValueError(
                f"cannot expire memory ref {ref.key!r} from "
                f"state {current_state.value!r}"
            )

        return self._replace_slot_state(ref, MemorySlotState.EXPIRED)

    def mark_failed(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a registered slot as failed from any current state."""

        ref = self._require_registered_memory_ref(memory_ref)
        return self._replace_slot_state(ref, MemorySlotState.FAILED)

    def mark_free(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a slot physically free.

        If an active/committed reservation still holds the slot, the slot
        returns to RESERVED rather than FREE. Release the reservation to make it
        available to other callers.
        """

        ref = self._require_registered_memory_ref(memory_ref)
        next_state = (
            MemorySlotState.RESERVED
            if ref in self._memory_holders
            else MemorySlotState.FREE
        )

        return self._replace_slot_state(ref, next_state)

    def _close_reservation(
        self,
        reservation_id: str,
        *,
        state: ReservationState,
    ) -> Reservation:
        reservation = self.get_reservation(reservation_id)

        if reservation.state not in (
            ReservationState.ACTIVE,
            ReservationState.COMMITTED,
        ):
            raise ValueError("only active or committed reservations can be closed")

        if state is ReservationState.RELEASED:
            updated = reservation.released()
        elif state is ReservationState.CANCELLED:
            updated = reservation.cancelled()
        elif state is ReservationState.EXPIRED:
            updated = reservation.expired()
        else:
            raise ValueError("unsupported reservation close state")

        for ref in reservation.memory_refs:
            holder = self._memory_holders.get(ref)

            if holder is None:
                raise RuntimeError(
                    f"memory ref {ref.key!r} is not held by reservation "
                    f"{reservation.reservation_id!r}"
                )

            if holder != reservation.reservation_id:
                raise RuntimeError(
                    f"memory ref {ref.key!r} is held by reservation "
                    f"{holder!r}, not {reservation.reservation_id!r}"
                )

            del self._memory_holders[ref]

            if self._slots[ref].state is MemorySlotState.RESERVED:
                self._replace_slot_state(ref, MemorySlotState.FREE)

        self._reservations[updated.reservation_id] = updated
        return updated

    def _require_available(self, memory_ref: MemoryRef, now: int) -> MemoryRef:
        ref = self._require_registered_memory_ref(memory_ref)

        if not self._is_available(ref, now):
            slot = self._slots[ref]
            raise ValueError(
                f"memory ref {ref.key!r} is not available "
                f"(state={slot.state.value!r})"
            )

        return ref

    def _require_registered_memory_ref(self, memory_ref: MemoryRef) -> MemoryRef:
        if not isinstance(memory_ref, MemoryRef):
            raise TypeError("memory_ref must be MemoryRef")

        if memory_ref not in self._slots:
            raise KeyError(f"unknown memory ref {memory_ref.key!r}")

        return memory_ref

    def _require_registered_node(self, node_id: str) -> None:
        if not any(ref.node_id == node_id for ref in self._slots):
            raise KeyError(f"unknown node id '{node_id}'")

    def _is_available(self, ref: MemoryRef, now: int) -> bool:
        slot = self._slots[ref]
        return (
            slot.state is MemorySlotState.FREE
            and (slot.ready_at is None or slot.ready_at <= now)
            and ref not in self._memory_holders
        )

    def _replace_slot_state(
        self,
        memory_ref: MemoryRef,
        state: MemorySlotState,
    ) -> MemorySlotView:
        ref = self._require_registered_memory_ref(memory_ref)

        if not isinstance(state, MemorySlotState):
            raise TypeError("state must be MemorySlotState")

        current = self._slots[ref]
        updated = MemorySlotView(
            ref=current.ref,
            state=state,
            ready_at=current.ready_at,
            expires_at=current.expires_at,
            metadata=current.metadata,
        )
        self._slots[ref] = updated
        return updated

    def _replace_slot_from_position_record(
        self,
        memory_ref: MemoryRef,
        position_record: MemoryPositionRecord,
    ) -> MemorySlotView:
        ref = self._require_registered_memory_ref(memory_ref)

        if not isinstance(position_record, MemoryPositionRecord):
            raise TypeError("position_record must be MemoryPositionRecord")

        current = self._slots[ref]
        updated = MemorySlotView(
            ref=current.ref,
            state=self._slot_state_from_position_status(position_record.status),
            ready_at=position_record.ready_at,
            expires_at=position_record.expires_at,
            metadata=current.metadata,
        )
        self._slots[ref] = updated
        return updated

    def _resolve_reservation_id(self, reservation_id: str | None) -> str:
        if reservation_id is not None:
            return ensure_nonempty_id(
                reservation_id,
                field_name="reservation_id",
            )

        while True:
            candidate = f"reservation:{self._next_reservation_index}"
            self._next_reservation_index += 1

            if candidate not in self._reservations:
                return candidate

    @staticmethod
    def _resolve_memory_refs(
        refs: tuple[MemoryRef, ...],
    ) -> tuple[MemoryRef, ...]:
        if not isinstance(refs, tuple):
            raise TypeError("memory_refs must be tuple[MemoryRef, ...]")

        if not refs:
            raise ValueError("memory_refs must be non-empty")

        seen: set[MemoryRef] = set()

        for ref in refs:
            if not isinstance(ref, MemoryRef):
                raise TypeError("memory_refs must contain only MemoryRef entries")

            if ref in seen:
                raise ValueError(f"duplicate memory ref {ref.key!r}")

            seen.add(ref)

        return refs

    @staticmethod
    def _resolve_requirements(
        requirements: Mapping[str, int | Mapping[str, int]],
    ) -> tuple[tuple[str, int | tuple[tuple[str, int], ...]], ...]:
        if not isinstance(requirements, Mapping):
            raise TypeError(
                "requirements must be a mapping of node_id to count or device mapping"
            )

        if not requirements:
            raise ValueError("requirements must be non-empty")

        resolved: list[tuple[str, int | tuple[tuple[str, int], ...]]] = []
        seen_node_ids: set[str] = set()

        for node_id, value in requirements.items():
            resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")

            if resolved_node_id in seen_node_ids:
                raise ValueError(f"duplicate node requirement {resolved_node_id!r}")

            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(
                        f"device mapping for node '{resolved_node_id}' cannot be empty"
                    )
                device_reqs = []
                for dev_id, dev_count in value.items():
                    resolved_dev_id = ensure_nonempty_id(dev_id, field_name="device_id")
                    validate_positive_int(dev_count, field_name="count")
                    device_reqs.append((resolved_dev_id, dev_count))
                resolved_value: int | tuple[tuple[str, int], ...] = tuple(
                    sorted(device_reqs)
                )
            else:
                validate_positive_int(value, field_name="count")
                resolved_value = value

            seen_node_ids.add(resolved_node_id)
            resolved.append((resolved_node_id, resolved_value))

        return tuple(sorted(resolved, key=lambda item: item[0]))

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
    def _matches_ref(
        ref: MemoryRef,
        *,
        node_id: str | None,
        device_id: str | None,
    ) -> bool:
        if node_id is not None and ref.node_id != node_id:
            return False

        if device_id is not None and ref.device_id != device_id:
            return False

        return True

    @staticmethod
    def _validate_metadata(metadata: tuple[tuple[str, object], ...]) -> None:
        if not isinstance(metadata, tuple):
            raise TypeError("metadata must be tuple[tuple[str, object], ...]")

        for item in metadata:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("metadata must contain only two-item tuple entries")

            key, _value = item
            ensure_nonempty_id(key, field_name="metadata key")

    @staticmethod
    def _slot_state_from_position_status(
        status: MemoryPositionStatus,
    ) -> MemorySlotState:
        if not isinstance(status, MemoryPositionStatus):
            raise TypeError("status must be MemoryPositionStatus")

        if status is MemoryPositionStatus.EMPTY:
            return MemorySlotState.FREE

        occupied_statuses = (
            MemoryPositionStatus.ABSORBING,
            MemoryPositionStatus.OCCUPIED,
            MemoryPositionStatus.EMITTING,
            MemoryPositionStatus.MEASURING,
            MemoryPositionStatus.APPLYING_OPERATOR,
        )

        if status in occupied_statuses:
            return MemorySlotState.OCCUPIED

        raise ValueError(f"unsupported memory position status {status!r}")


__all__ = [
    "ResourceManager",
    "UnauthorizedError",
]
