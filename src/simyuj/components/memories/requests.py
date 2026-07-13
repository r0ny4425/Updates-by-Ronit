"""Memory-component request payloads keyed by physical position.

These immutable records are passed as ``Event.payload_ref`` values to
``QuantumMemory`` actions. They identify the target memory, physical position
or ordered positions, optional session metadata, and operation-specific
parameters; they do not mutate qstate themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import Meta, validate_meta
from simyuj.primitives.validation import validate_bool, validate_non_negative_int
from simyuj.signal import Signal


@dataclass(frozen=True, slots=True)
class MemoryAbsorbRequest:
    """
    Request absorption of an incoming photon signal into memory.

    Parameters
    ----------
    request_id : str
        Caller-chosen request identifier.
    memory_id : str
        Target ``QuantumMemory.memory_id``.
    signal : Signal
        Incoming qstate-backed photon signal.
    position : int or None, default=None
        Optional physical memory position. ``None`` lets the memory choose the
        first available position by physical index, subject to recovery
        ``ready_at`` timing.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied into position state and reports.

    Notes
    -----
    Successful absorption relabels the signal's qstate target into the stable
    memory-position subsystem. Failed absorption discards the incoming photon
    target.
    """

    request_id: str
    memory_id: str
    signal: Signal
    position: int | None = None
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be Signal")
        _check_optional_position(self.position)


@dataclass(frozen=True, slots=True)
class MemoryEmitRequest:
    """
    Request emission of one stored memory position as an outgoing photon.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    position : int
        Occupied memory position to emit.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied into the emitted signal and report.

    Notes
    -----
    Successful emission relabels the memory subsystem into a unique emitted
    photon subsystem, transmits a photon ``Signal`` through the memory output
    port, and clears the position into recovery.
    """

    request_id: str
    memory_id: str
    position: int
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_position(self.position)


@dataclass(frozen=True, slots=True)
class MemoryApplyOperatorRequest:
    """
    Request applying one operator to exact ordered memory positions.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    positions : tuple[int, ...]
        Non-empty unique physical positions. Order is preserved for qstate
        target ordering.
    operator : object
        Operator object passed to the qstate manager.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied to reports.
    """

    request_id: str
    memory_id: str
    positions: tuple[int, ...]
    operator: object
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_positions(self.positions)
        if self.operator is None:
            raise ValueError("operator must be supplied")


@dataclass(frozen=True, slots=True)
class MemoryMeasureRequest:
    """
    Request measurement of exact ordered memory positions.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    positions : tuple[int, ...]
        Non-empty unique physical positions. Order is preserved for qstate
        measurement targets.
    measurement : object, default="z"
        Measurement spec accepted by the detector readout primitive.
    collapse : bool, default=True
        Whether the qstate measurement collapses target state.
    destructive : bool, default=True
        Whether successful measurement discards measured memory subsystems and
        clears their positions.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied to readout and memory reports.
    """

    request_id: str
    memory_id: str
    positions: tuple[int, ...]
    measurement: object = "z"
    collapse: bool = True
    destructive: bool = True
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_positions(self.positions)
        if self.measurement is None:
            raise ValueError("measurement must be supplied")
        validate_bool(self.collapse, field_name="collapse")
        validate_bool(self.destructive, field_name="destructive")


@dataclass(frozen=True, slots=True)
class MemoryDiscardRequest:
    """
    Request explicit discard of one occupied memory position.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    position : int
        Occupied memory position to discard.
    reason : str, default="discarded"
        Non-empty reason stored in the discard report and position metadata.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied to reports.
    """

    request_id: str
    memory_id: str
    position: int
    reason: str = "discarded"
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_position(self.position)
        ensure_nonempty_id(self.reason, field_name="reason")


@dataclass(frozen=True, slots=True)
class MemoryExpireRequest:
    """
    Request expiry of one position for a matching occupancy token.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    position : int
        Memory position to expire.
    occupancy_token : int
        Token that must match the current position occupancy. Stale expiry
        requests are logged and ignored.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied to reports.
    """

    request_id: str
    memory_id: str
    position: int
    occupancy_token: int
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_position(self.position)
        validate_non_negative_int(
            self.occupancy_token,
            field_name="occupancy_token",
        )


@dataclass(frozen=True, slots=True)
class MemoryUpdateMetaRequest:
    """
    Update classical protocol metadata for one memory position.

    Parameters
    ----------
    request_id, memory_id : str
        Request and target memory identifiers.
    position : int
        Occupied memory position whose metadata should change.
    updates : Meta, default=()
        Metadata entries to append or replace by key.
    remove_keys : tuple[str, ...], default=()
        Metadata keys removed before updates are appended.
    expected_occupancy_token : int or None, default=None
        Optional current-token guard. A mismatch reports failure without
        mutating position metadata.
    session_id : str or None, default=None
        Optional session identifier copied to reports.
    meta : Meta, default=()
        Request metadata copied to reports.

    Notes
    -----
    Metadata updates are classical bookkeeping only. They do not touch qstate
    or position timing fields. ``updates`` changes position metadata;
    request-level ``meta`` is copied only into reports and log context.
    """

    request_id: str
    memory_id: str
    position: int
    updates: Meta = field(default_factory=tuple)
    remove_keys: tuple[str, ...] = field(default_factory=tuple)
    expected_occupancy_token: int | None = None
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_request(self.request_id, self.memory_id, self.session_id, self.meta)
        _check_position(self.position)
        validate_meta(self.updates)
        _check_unique_meta_keys(self.updates, field_name="updates")
        _check_remove_keys(self.remove_keys)
        if self.expected_occupancy_token is not None:
            validate_non_negative_int(
                self.expected_occupancy_token,
                field_name="expected_occupancy_token",
            )


def _check_request(
    request_id: str,
    memory_id: str,
    session_id: str | None,
    meta: Meta,
) -> None:
    ensure_nonempty_id(request_id, field_name="request_id")
    ensure_nonempty_id(memory_id, field_name="memory_id")
    if session_id is not None:
        ensure_nonempty_id(session_id, field_name="session_id")
    validate_meta(meta)


def _check_optional_position(position: int | None) -> None:
    if position is not None:
        _check_position(position)


def _check_position(position: int) -> None:
    validate_non_negative_int(position, field_name="position")


def _check_positions(positions: object) -> None:
    if not isinstance(positions, tuple):
        raise TypeError("positions must be tuple[int, ...]")
    if not positions:
        raise ValueError("positions must be non-empty")
    for position in positions:
        _check_position(position)
    if len(set(positions)) != len(positions):
        raise ValueError("positions must be unique")


def _check_unique_meta_keys(meta: Meta, *, field_name: str) -> None:
    seen: set[str] = set()
    for key, _ in meta:
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate key {key!r}")
        seen.add(key)


def _check_remove_keys(remove_keys: object) -> None:
    if not isinstance(remove_keys, tuple):
        raise TypeError("remove_keys must be tuple[str, ...]")
    for key in remove_keys:
        if type(key) is not str or key == "":
            raise ValueError("remove_keys must contain nonempty strings")
    if len(set(remove_keys)) != len(remove_keys):
        raise ValueError("remove_keys contains duplicate keys")


__all__ = [
    "MemoryAbsorbRequest",
    "MemoryApplyOperatorRequest",
    "MemoryDiscardRequest",
    "MemoryEmitRequest",
    "MemoryExpireRequest",
    "MemoryMeasureRequest",
    "MemoryUpdateMetaRequest",
]
