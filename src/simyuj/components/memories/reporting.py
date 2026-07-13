"""Runtime logging helpers for memory reports.

The helpers translate immutable memory reports into structured timeline log
records. They do not mutate memory state, schedule events, or alter report
payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simyuj.tracing.levels import LogLevel

from .position import MemoryPositionRecord
from .reports import (
    MemoryAbsorbReport,
    MemoryDiscardReport,
    MemoryEmitReport,
    MemoryExpireReport,
    MemoryMeasurementReport,
    MemoryMetaUpdateReport,
    MemoryOperatorReport,
    MemoryReport,
)

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline


def log_memory_report(
    report: MemoryReport,
    timeline: Timeline,
    *,
    positions: tuple[MemoryPositionRecord, ...],
    event_id: int | None,
    action: str | None,
) -> None:
    """Emit the structured timeline log entry for one memory report."""

    category, message, meta = memory_report_log_fields(report, positions=positions)
    timeline.log(
        LogLevel.DEBUG,
        category,
        message,
        event_id=event_id,
        action=action,
        meta=meta,
    )


def memory_report_log_fields(
    report: MemoryReport,
    *,
    positions: tuple[MemoryPositionRecord, ...],
) -> tuple[str, str, dict[str, object]]:
    """
    Return tracing category, message, and metadata for one memory report.

    Notes
    -----
    The caller supplies current position snapshots so absorb reports can expose
    expiry metadata after a successful state transition. ``positions`` must be
    the full snapshot tuple from the same memory.
    """

    if isinstance(report, MemoryAbsorbReport):
        meta = {
            "memory_id": report.memory_id,
            "request_id": report_meta_value(report, "request_id"),
            "report_id": report.report_id,
            "position": report.position,
            "occupancy_token": report_meta_value(report, "occupancy_token"),
            "input_signal_id": report.input_signal_id,
            "success": report.success,
            "status": report.status,
        }
        if report.success:
            meta["expires_at"] = positions[report.position].expires_at
        else:
            meta["absorb_success_probability"] = report_meta_value(
                report,
                "absorb_success_probability",
            )
        return (
            "components.memories.quantum_memory.absorb",
            "memory absorb reported",
            meta,
        )

    if isinstance(report, MemoryEmitReport):
        meta = {
            "memory_id": report.memory_id,
            "request_id": report_meta_value(report, "request_id"),
            "report_id": report.report_id,
            "position": report.position,
            "occupancy_token": report_meta_value(report, "occupancy_token"),
            "output_signal_id": report.output_signal_id,
            "success": report.success,
            "status": report.status,
        }
        if report.success:
            meta["ready_at"] = report_meta_value(report, "ready_at")
        else:
            meta["emit_success_probability"] = report_meta_value(
                report,
                "emit_success_probability",
            )
        return (
            "components.memories.quantum_memory.emit",
            "memory emit reported",
            meta,
        )

    if isinstance(report, MemoryOperatorReport):
        return (
            "components.memories.quantum_memory.operator",
            "memory operator reported",
            {
                "memory_id": report.memory_id,
                "request_id": report_meta_value(report, "request_id"),
                "report_id": report.report_id,
                "positions": report.positions,
                "occupancy_tokens": report_meta_value(
                    report,
                    "occupancy_tokens",
                ),
                "success": report.success,
                "status": report.status,
            },
        )

    if isinstance(report, MemoryMeasurementReport):
        detection_report = report.detection_report
        return (
            "components.memories.quantum_memory.measure",
            "memory measurement reported",
            {
                "memory_id": report.memory_id,
                "request_id": report_meta_value(report, "request_id"),
                "report_id": report.report_id,
                "positions": report.positions,
                "measurement_label": (
                    None
                    if detection_report is None
                    else detection_report.measurement_label
                ),
                "success": report.success,
                "outcome": (
                    None if detection_report is None else detection_report.outcome
                ),
                "destructive": report.destructive,
                "cleared_positions": report.cleared_positions,
                "status": report.status,
                "flags": () if detection_report is None else detection_report.flags,
            },
        )

    if isinstance(report, MemoryDiscardReport):
        return (
            "components.memories.quantum_memory.discard",
            "memory position discarded",
            {
                "memory_id": report.memory_id,
                "request_id": report_meta_value(report, "request_id"),
                "report_id": report.report_id,
                "position": report.position,
                "occupancy_token": report_meta_value(
                    report,
                    "occupancy_token",
                ),
                "success": report.success,
                "status": report.status,
                "reason": report.reason,
                "ready_at": report_meta_value(report, "ready_at"),
            },
        )

    if isinstance(report, MemoryExpireReport):
        return (
            "components.memories.quantum_memory.expire",
            "memory position expired",
            {
                "memory_id": report.memory_id,
                "request_id": report_meta_value(report, "request_id"),
                "report_id": report.report_id,
                "position": report.position,
                "occupancy_token": report_meta_value(
                    report,
                    "occupancy_token",
                ),
                "success": report.success,
                "status": report.status,
                "reason": report.reason,
                "ready_at": report_meta_value(report, "ready_at"),
            },
        )

    if isinstance(report, MemoryMetaUpdateReport):
        return (
            "components.memories.quantum_memory.meta_update",
            "memory metadata updated",
            {
                "memory_id": report.memory_id,
                "request_id": report_meta_value(report, "request_id"),
                "report_id": report.report_id,
                "position": report.position,
                "occupancy_token": report.occupancy_token,
                "success": report.success,
                "status": report.status,
                "updated_keys": report.updated_keys,
                "removed_keys": report.removed_keys,
            },
        )

    raise TypeError("unsupported memory report type")


def report_meta_value(report: MemoryReport, key: str) -> object | None:
    """Return the first report metadata value for ``key``, if present."""

    for meta_key, value in report.meta:
        if meta_key == key:
            return value
    return None


__all__ = [
    "log_memory_report",
    "memory_report_log_fields",
    "report_meta_value",
]
