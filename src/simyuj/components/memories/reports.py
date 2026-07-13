"""Classical report payloads emitted by memory components.

Memory reports are immutable outcome records stored by ``QuantumMemory`` and
optionally transmitted through its classical notice port. They describe the
operation result after any qstate mutation has already happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from simyuj.components.detectors.primitives.reports import DetectionReport
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import Meta, validate_meta
from simyuj.primitives.validation import validate_bool, validate_non_negative_int
from simyuj.qstate import SubsystemId


@dataclass(frozen=True, slots=True)
class MemoryAbsorbReport:
    """
    Report the result of absorbing a photon into one memory position.

    Notes
    -----
    A successful report includes the input signal id and stable memory
    subsystem. An unsuccessful report may omit the memory subsystem because the
    incoming photon target was discarded instead of stored.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    position: int
    input_signal_id: object | None
    memory_subsystem: SubsystemId | None
    status: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_position(self.position)
        _check_optional_subsystem(
            self.memory_subsystem,
            field_name="memory_subsystem",
        )
        if self.success:
            _require_present(self.input_signal_id, field_name="input_signal_id")
            _require_present(
                self.memory_subsystem,
                field_name="memory_subsystem",
            )


@dataclass(frozen=True, slots=True)
class MemoryEmitReport:
    """
    Report the result of emitting one memory position as a photon.

    Notes
    -----
    A successful report includes both the previous memory subsystem and the
    emitted photon subsystem. The memory position has already been cleared and
    placed into recovery when this report is produced.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    position: int
    memory_subsystem: SubsystemId | None
    output_signal_id: object | None
    output_subsystem: SubsystemId | None
    status: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_position(self.position)
        _check_optional_subsystem(
            self.memory_subsystem,
            field_name="memory_subsystem",
        )
        _check_optional_subsystem(
            self.output_subsystem,
            field_name="output_subsystem",
        )
        if self.success:
            _require_present(
                self.memory_subsystem,
                field_name="memory_subsystem",
            )
            _require_present(self.output_signal_id, field_name="output_signal_id")
            _require_present(self.output_subsystem, field_name="output_subsystem")


@dataclass(frozen=True, slots=True)
class MemoryOperatorReport:
    """
    Report the result of applying an operator to ordered positions.

    Notes
    -----
    ``positions`` and ``memory_subsystems`` preserve the request ordering used
    for qstate operator targets. Operator reports are currently produced only
    when the memory notice port is connected.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    positions: tuple[int, ...]
    memory_subsystems: tuple[SubsystemId, ...]
    status: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_positions(self.positions)
        _check_subsystems(self.memory_subsystems, field_name="memory_subsystems")
        _check_ordered_subsystems_for_positions(
            self.positions,
            self.memory_subsystems,
            success=self.success,
        )


@dataclass(frozen=True, slots=True)
class MemoryMeasurementReport:
    """
    Report a memory measurement and optional detector readout result.

    Notes
    -----
    ``detection_report`` wraps the detector readout primitive's outcome.
    Destructive measurement clears the measured positions after qstate
    measurement; non-destructive measurement leaves them occupied.
    ``cleared_positions`` records which positions were cleared. When
    ``destructive`` is true, positions are cleared even if the detector readout
    reports no outcome.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    positions: tuple[int, ...]
    memory_subsystems: tuple[SubsystemId, ...]
    detection_report: DetectionReport | None
    destructive: bool
    cleared_positions: tuple[int, ...]
    status: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_positions(self.positions)
        _check_subsystems(self.memory_subsystems, field_name="memory_subsystems")
        _check_ordered_subsystems_for_positions(
            self.positions,
            self.memory_subsystems,
            success=self.success,
        )
        _check_optional_detection_report(self.detection_report)
        validate_bool(self.destructive, field_name="destructive")
        _check_positions(
            self.cleared_positions,
            field_name="cleared_positions",
            allow_empty=True,
        )
        if self.success:
            _require_present(
                self.detection_report,
                field_name="detection_report",
            )


@dataclass(frozen=True, slots=True)
class MemoryDiscardReport:
    """
    Report explicit discard of one memory position.

    Notes
    -----
    Successful discard means the memory subsystem has already been discarded
    from qstate and the position has been cleared into recovery.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    position: int
    memory_subsystem: SubsystemId | None
    status: str
    reason: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_position(self.position)
        _check_optional_subsystem(
            self.memory_subsystem,
            field_name="memory_subsystem",
        )
        ensure_nonempty_id(self.reason, field_name="reason")
        if self.success:
            _require_present(
                self.memory_subsystem,
                field_name="memory_subsystem",
            )


@dataclass(frozen=True, slots=True)
class MemoryExpireReport:
    """
    Report expiry handling for one memory position.

    Notes
    -----
    Successful expiry is token-checked. Stale expiry requests are logged and do
    not create this report or clear the current position contents.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    position: int
    memory_subsystem: SubsystemId | None
    status: str
    reason: str
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_position(self.position)
        _check_optional_subsystem(
            self.memory_subsystem,
            field_name="memory_subsystem",
        )
        ensure_nonempty_id(self.reason, field_name="reason")
        if self.success:
            _require_present(
                self.memory_subsystem,
                field_name="memory_subsystem",
            )


@dataclass(frozen=True, slots=True)
class MemoryMetaUpdateReport:
    """
    Report a classical metadata update attempt for one memory position.

    Notes
    -----
    Metadata update reports may describe success or failure. Failure reports can
    omit ``memory_subsystem`` for empty or busy positions; successful reports
    include changed keys. ``occupancy_token`` is the current position token in
    both success and failure reports, not necessarily the caller's expected
    token.
    """

    report_id: str
    memory_id: str
    time: int
    success: bool
    position: int
    memory_subsystem: SubsystemId | None
    occupancy_token: int
    status: str
    updated_keys: tuple[str, ...] = field(default_factory=tuple)
    removed_keys: tuple[str, ...] = field(default_factory=tuple)
    session_id: str | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _check_common(
            self.report_id,
            self.memory_id,
            self.time,
            self.success,
            self.status,
            self.session_id,
            self.meta,
        )
        _check_position(self.position)
        _check_optional_subsystem(
            self.memory_subsystem,
            field_name="memory_subsystem",
        )
        validate_non_negative_int(
            self.occupancy_token,
            field_name="occupancy_token",
        )
        _check_string_tuple(self.updated_keys, field_name="updated_keys")
        _check_string_tuple(self.removed_keys, field_name="removed_keys")
        if self.success:
            _require_present(
                self.memory_subsystem,
                field_name="memory_subsystem",
            )


MemoryReport: TypeAlias = (
    MemoryAbsorbReport
    | MemoryDiscardReport
    | MemoryEmitReport
    | MemoryExpireReport
    | MemoryMetaUpdateReport
    | MemoryMeasurementReport
    | MemoryOperatorReport
)
"""Union of all memory report payload types."""


def _check_common(
    report_id: str,
    memory_id: str,
    time: int,
    success: bool,
    status: str,
    session_id: str | None,
    meta: Meta,
) -> None:
    ensure_nonempty_id(report_id, field_name="report_id")
    ensure_nonempty_id(memory_id, field_name="memory_id")
    validate_non_negative_int(time, field_name="time")
    validate_bool(success, field_name="success")
    ensure_nonempty_id(status, field_name="status")
    if session_id is not None:
        ensure_nonempty_id(session_id, field_name="session_id")
    validate_meta(meta)


def _check_position(position: int) -> None:
    validate_non_negative_int(position, field_name="position")


def _check_positions(
    positions: object,
    *,
    field_name: str = "positions",
    allow_empty: bool = False,
) -> None:
    if not isinstance(positions, tuple):
        raise TypeError(f"{field_name} must be tuple[int, ...]")
    if not positions and not allow_empty:
        raise ValueError(f"{field_name} must be non-empty")
    for position in positions:
        validate_non_negative_int(position, field_name=field_name)
    if len(set(positions)) != len(positions):
        raise ValueError(f"{field_name} must be unique")


def _check_optional_subsystem(subsystem: object, *, field_name: str) -> None:
    if subsystem is not None and not isinstance(subsystem, SubsystemId):
        raise TypeError(f"{field_name} must be SubsystemId or None")


def _check_subsystems(subsystems: object, *, field_name: str) -> None:
    if not isinstance(subsystems, tuple):
        raise TypeError(f"{field_name} must be tuple[SubsystemId, ...]")
    for subsystem in subsystems:
        if not isinstance(subsystem, SubsystemId):
            raise TypeError(f"{field_name} must contain SubsystemId")


def _check_string_tuple(values: object, *, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be tuple[str, ...]")
    for value in values:
        ensure_nonempty_id(value, field_name=field_name)


def _check_ordered_subsystems_for_positions(
    positions: tuple[int, ...],
    subsystems: tuple[SubsystemId, ...],
    *,
    success: bool,
) -> None:
    if not subsystems and not success:
        return
    if len(subsystems) != len(positions):
        raise ValueError("memory_subsystems must match positions")


def _check_optional_detection_report(report: object) -> None:
    if report is not None and not isinstance(report, DetectionReport):
        raise TypeError("detection_report must be DetectionReport or None")


def _require_present(value: object, *, field_name: str) -> None:
    if value is None:
        raise ValueError(f"{field_name} is required for successful report")


__all__ = [
    "MemoryAbsorbReport",
    "MemoryDiscardReport",
    "MemoryEmitReport",
    "MemoryExpireReport",
    "MemoryMetaUpdateReport",
    "MemoryMeasurementReport",
    "MemoryOperatorReport",
    "MemoryReport",
]
