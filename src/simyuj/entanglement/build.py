"""Construction helpers for memory-backed entangled pair records.

The helpers here translate successful component reports into
``EntangledPairRecord`` bookkeeping. They do not create qstate, run swapping
physics, reserve resources, or schedule component work.
"""

from __future__ import annotations

from simyuj.components.detectors import DetectionReport
from simyuj.components.memories import MemoryAbsorbReport
from simyuj.entanglement.pair import EntangledPairRecord
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.resources.memory import MemoryRef


def pair_from_absorbs(
    pair_id: str,
    left_ref: MemoryRef,
    left_report: MemoryAbsorbReport,
    right_ref: MemoryRef,
    right_report: MemoryAbsorbReport,
    *,
    fidelity: float | None = None,
    created_at: int | None = None,
    expires_at: int | None = None,
    generation_link_id: str | None = None,
    left_memory_id: str | None = None,
    right_memory_id: str | None = None,
    metadata: tuple[tuple[str, object], ...] = (),
) -> EntangledPairRecord:
    """Build a pair record from two successful memory absorb reports."""

    _check_absorb_side(
        "left",
        left_ref,
        left_report,
        expected_memory_id=left_memory_id,
    )
    _check_absorb_side(
        "right",
        right_ref,
        right_report,
        expected_memory_id=right_memory_id,
    )

    return EntangledPairRecord(
        pair_id=pair_id,
        left=left_ref,
        right=right_ref,
        fidelity=fidelity,
        created_at=created_at,
        expires_at=expires_at,
        generation_link_id=generation_link_id,
        left_occupancy_token=_occupancy_token(left_report),
        right_occupancy_token=_occupancy_token(right_report),
        metadata=metadata,
    )


def swapped_pair_from_bsa(
    pair_id: str,
    left_pair: EntangledPairRecord,
    right_pair: EntangledPairRecord,
    *,
    left_outer: MemoryRef,
    right_outer: MemoryRef,
    report: DetectionReport,
    fidelity: float | None = None,
    created_at: int | None = None,
    expires_at: int | None = None,
    metadata: tuple[tuple[str, object], ...] = (),
) -> EntangledPairRecord:
    """Build a swapped-pair record from two input pairs and a BSA report."""

    if not isinstance(left_pair, EntangledPairRecord):
        raise TypeError("left_pair must be EntangledPairRecord")
    if not isinstance(right_pair, EntangledPairRecord):
        raise TypeError("right_pair must be EntangledPairRecord")
    if not isinstance(left_outer, MemoryRef):
        raise TypeError("left_outer must be MemoryRef")
    if not isinstance(right_outer, MemoryRef):
        raise TypeError("right_outer must be MemoryRef")
    if not isinstance(report, DetectionReport):
        raise TypeError("report must be DetectionReport")
    if not report.success:
        raise ValueError("report must indicate success")

    left_token = _outer_token(left_pair, left_outer, side_name="left_outer")
    right_token = _outer_token(right_pair, right_outer, side_name="right_outer")

    return EntangledPairRecord(
        pair_id=pair_id,
        left=left_outer,
        right=right_outer,
        fidelity=fidelity,
        created_at=created_at,
        expires_at=expires_at,
        left_occupancy_token=left_token,
        right_occupancy_token=right_token,
        metadata=metadata
        + (
            ("source_pair_ids", (left_pair.pair_id, right_pair.pair_id)),
            ("bsa_report_id", report.report_id),
            ("bsa_outcome", report.outcome),
        ),
    )


def _check_absorb_side(
    side: str,
    memory_ref: MemoryRef,
    report: MemoryAbsorbReport,
    *,
    expected_memory_id: str | None,
) -> None:
    if not isinstance(memory_ref, MemoryRef):
        raise TypeError(f"{side}_ref must be MemoryRef")
    if not isinstance(report, MemoryAbsorbReport):
        raise TypeError(f"{side}_report must be MemoryAbsorbReport")
    if not report.success:
        raise ValueError(f"{side}_report must indicate success")
    if report.position != memory_ref.position:
        raise ValueError(f"{side}_report position must match {side}_ref position")
    if expected_memory_id is not None:
        expected = ensure_nonempty_id(
            expected_memory_id,
            field_name=f"{side}_memory_id",
        )
        if report.memory_id != expected:
            raise ValueError(f"{side}_report memory_id must match {side}_memory_id")


def _occupancy_token(report: MemoryAbsorbReport) -> int | None:
    for key, value in report.meta:
        if key == "occupancy_token":
            return value if type(value) is int else None
    return None


def _outer_token(
    pair: EntangledPairRecord,
    outer: MemoryRef,
    *,
    side_name: str,
) -> int | None:
    if outer == pair.left:
        return pair.left_occupancy_token
    if outer == pair.right:
        return pair.right_occupancy_token
    raise ValueError(f"{side_name} not found in input pair")


__all__ = ["pair_from_absorbs", "swapped_pair_from_bsa"]
