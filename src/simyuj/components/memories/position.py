"""Physical memory-position records and deterministic qstate carrier labels.

Positions are immutable snapshots owned by ``QuantumMemory``. They record the
classical lifecycle state for a physical memory slot and the stable qstate
subsystem label used while that slot is occupied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import Meta, validate_meta
from simyuj.primitives.validation import validate_non_negative_int
from simyuj.qstate import SubsystemId
from simyuj.signal import Signal


class MemoryPositionStatus(Enum):
    """
    Lifecycle status for one physical memory position.

    Notes
    -----
    ``EMPTY`` positions are available only at or after ``ready_at``. Busy
    statuses reserve a position while delayed memory operations are pending.
    Stored statuses own a ``memory_subsystem`` and storage timing fields.
    """

    EMPTY = "empty"
    ABSORBING = "absorbing"
    OCCUPIED = "occupied"
    EMITTING = "emitting"
    MEASURING = "measuring"
    APPLYING_OPERATOR = "applying_operator"


_MEMORY_SUBSYSTEM_STATUSES = frozenset(
    (
        MemoryPositionStatus.OCCUPIED,
        MemoryPositionStatus.EMITTING,
        MemoryPositionStatus.MEASURING,
        MemoryPositionStatus.APPLYING_OPERATOR,
    )
)


@dataclass(frozen=True, slots=True)
class MemoryPositionRecord:
    """
    Immutable state snapshot for one physical memory position.

    Parameters
    ----------
    position : int
        Zero-based physical position index.
    status : MemoryPositionStatus, default=MemoryPositionStatus.EMPTY
        Lifecycle status for the position.
    memory_subsystem : SubsystemId or None, default=None
        Stable qstate carrier label while the position stores a qubit.
    stored_signal : Signal or None, default=None
        Incoming signal associated with the stored qubit or pending absorb.
    stored_time : int or None, default=None
        Tick at which storage became occupied.
    last_noise_update_time : int or None, default=None
        Last tick at which pending storage noise was applied.
    expires_at : int or None, default=None
        Scheduled expiry tick for the current occupancy.
    occupancy_token : int, default=0
        Monotonic token used to reject stale delayed completions and expiries.
    ready_at : int, default=0
        Earliest tick at which an empty position can absorb again.
    meta : Meta, default=()
        Classical metadata associated with the position.

    Notes
    -----
    ``memory_subsystem`` is the stable qstate carrier handle for occupied
    memory, independent of the store's current ``StateRef``.

    ``ready_at`` is meaningful only when ``status`` is ``EMPTY``; it models
    automatic recovery dead time after quantum-carrier removal.

    This record validates status-owned fields, not time ordering. ``EMPTY``
    positions own no subsystem, signal, or storage timing; ``ABSORBING`` keeps
    only the pending signal; stored and busy positions require
    ``memory_subsystem``, ``stored_time``, and ``last_noise_update_time``.
    """

    position: int
    status: MemoryPositionStatus = MemoryPositionStatus.EMPTY
    memory_subsystem: SubsystemId | None = None
    stored_signal: Signal | None = None
    stored_time: int | None = None
    last_noise_update_time: int | None = None
    expires_at: int | None = None
    occupancy_token: int = 0
    ready_at: int = 0
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_non_negative_int(self.position, field_name="position")
        _check_status(self.status)
        _check_optional_subsystem(self.memory_subsystem)
        _check_optional_signal(self.stored_signal)
        _check_optional_time(self.stored_time, field_name="stored_time")
        _check_optional_time(
            self.last_noise_update_time,
            field_name="last_noise_update_time",
        )
        _check_optional_time(self.expires_at, field_name="expires_at")
        validate_non_negative_int(self.occupancy_token, field_name="occupancy_token")
        validate_non_negative_int(self.ready_at, field_name="ready_at")
        validate_meta(self.meta)
        _check_status_fields(self)


def memory_subsystem_id(memory_id: str, position: int) -> SubsystemId:
    """
    Return the stable qstate subsystem label for a memory position.

    Notes
    -----
    Successful absorption relabels an incoming photon target to this stable
    label. Reusing the same physical position reuses the same memory subsystem
    label with a newer occupancy token.
    """
    checked_memory_id = ensure_nonempty_id(memory_id, field_name="memory_id")
    validate_non_negative_int(position, field_name="position")
    return SubsystemId(f"memory:{checked_memory_id}:position:{position}")


def emitted_photon_subsystem_id(
    memory_id: str,
    position: int,
    emission_counter: int,
) -> SubsystemId:
    """
    Return a unique emitted-photon subsystem label for one emission.

    Notes
    -----
    Successful emission relabels the memory subsystem to this photon label
    before transmitting the outgoing signal.
    """
    checked_memory_id = ensure_nonempty_id(memory_id, field_name="memory_id")
    validate_non_negative_int(position, field_name="position")
    validate_non_negative_int(emission_counter, field_name="emission_counter")
    return SubsystemId(
        f"photon:{checked_memory_id}:position:{position}:emit:{emission_counter}"
    )


def _check_optional_time(value: object, *, field_name: str) -> None:
    if value is not None:
        validate_non_negative_int(value, field_name=field_name)


def _check_status(status: object) -> None:
    if not isinstance(status, MemoryPositionStatus):
        raise TypeError("status must be MemoryPositionStatus")


def _check_optional_subsystem(subsystem: object) -> None:
    if subsystem is not None and not isinstance(subsystem, SubsystemId):
        raise TypeError("memory_subsystem must be SubsystemId or None")


def _check_optional_signal(signal: object) -> None:
    if signal is not None and not isinstance(signal, Signal):
        raise TypeError("stored_signal must be Signal or None")


def _check_status_fields(record: MemoryPositionRecord) -> None:
    if record.status is MemoryPositionStatus.EMPTY:
        if record.memory_subsystem is not None:
            raise ValueError("empty position cannot own a memory_subsystem")
        if record.stored_signal is not None:
            raise ValueError("empty position cannot store a signal")
        if record.stored_time is not None:
            raise ValueError("empty position cannot have stored_time")
        if record.last_noise_update_time is not None:
            raise ValueError("empty position cannot have last_noise_update_time")
        if record.expires_at is not None:
            raise ValueError("empty position cannot have expires_at")
        return

    if record.status is MemoryPositionStatus.ABSORBING:
        if record.memory_subsystem is not None:
            raise ValueError("absorbing position cannot own a memory_subsystem")
        if record.stored_signal is None:
            raise ValueError("absorbing position requires stored_signal")
        if record.stored_time is not None:
            raise ValueError("absorbing position cannot have stored_time")
        if record.last_noise_update_time is not None:
            raise ValueError("absorbing position cannot have last_noise_update_time")
        if record.expires_at is not None:
            raise ValueError("absorbing position cannot have expires_at")
        return

    if record.status in _MEMORY_SUBSYSTEM_STATUSES:
        if record.memory_subsystem is None:
            raise ValueError("stored memory position requires memory_subsystem")
        if record.stored_time is None:
            raise ValueError("stored memory position requires stored_time")
        if record.last_noise_update_time is None:
            raise ValueError("stored memory position requires last_noise_update_time")
        return


__all__ = [
    "MemoryPositionRecord",
    "MemoryPositionStatus",
    "emitted_photon_subsystem_id",
    "memory_subsystem_id",
]
