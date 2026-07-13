from __future__ import annotations

import pytest

from simyuj.components.detectors import DetectionReport
from simyuj.components.memories import MemoryAbsorbReport
from simyuj.entanglement import (
    EntangledPairRecord,
    pair_from_absorbs,
    swapped_pair_from_bsa,
)
from simyuj.qstate import SubsystemId
from simyuj.resources import MemoryRef


def absorb_report(
    memory_id: str,
    *,
    report_id: str,
    position: int,
    success: bool = True,
    occupancy_token: int | None = None,
) -> MemoryAbsorbReport:
    meta = () if occupancy_token is None else (("occupancy_token", occupancy_token),)
    return MemoryAbsorbReport(
        report_id=report_id,
        memory_id=memory_id,
        time=10,
        success=success,
        position=position,
        input_signal_id=f"signal:{report_id}" if success else None,
        memory_subsystem=(
            SubsystemId(f"memory:{memory_id}:position:{position}") if success else None
        ),
        status="occupied" if success else "failed",
        meta=meta,
    )


def test_pair_from_absorbs_preserves_report_occupancy_tokens() -> None:
    left = MemoryRef("alice", "mem", 0)
    right = MemoryRef("bob", "mem", 1)

    pair = pair_from_absorbs(
        "pair:1",
        left,
        absorb_report(
            "alice.mem", report_id="absorb:alice", position=0, occupancy_token=11
        ),
        right,
        absorb_report(
            "bob.mem", report_id="absorb:bob", position=1, occupancy_token=22
        ),
        fidelity=0.98,
        created_at=10,
        expires_at=30,
        generation_link_id="link:alice-bob",
        left_memory_id="alice.mem",
        right_memory_id="bob.mem",
        metadata=(("source", "entangled_pair_source"),),
    )

    assert pair == EntangledPairRecord(
        pair_id="pair:1",
        left=left,
        right=right,
        fidelity=0.98,
        created_at=10,
        expires_at=30,
        generation_link_id="link:alice-bob",
        left_occupancy_token=11,
        right_occupancy_token=22,
        metadata=(("source", "entangled_pair_source"),),
    )


def test_pair_from_absorbs_rejects_failed_position_and_memory_mismatches() -> None:
    left = MemoryRef("alice", "mem", 0)
    right = MemoryRef("bob", "mem", 1)
    good_left = absorb_report("alice.mem", report_id="absorb:alice", position=0)
    good_right = absorb_report("bob.mem", report_id="absorb:bob", position=1)

    with pytest.raises(ValueError, match="right_report must indicate success"):
        pair_from_absorbs(
            "pair:1",
            left,
            good_left,
            right,
            absorb_report("bob.mem", report_id="absorb:bob", position=1, success=False),
        )

    with pytest.raises(ValueError, match="left_report position"):
        pair_from_absorbs(
            "pair:1",
            left,
            absorb_report("alice.mem", report_id="absorb:alice", position=1),
            right,
            good_right,
        )

    with pytest.raises(ValueError, match="right_report memory_id"):
        pair_from_absorbs(
            "pair:1",
            left,
            good_left,
            right,
            good_right,
            right_memory_id="wrong.mem",
        )


def test_swapped_pair_from_bsa_preserves_outer_tokens_and_bsa_metadata() -> None:
    alice = MemoryRef("alice", "mem", 0)
    relay_left = MemoryRef("relay", "left_mem", 0)
    relay_right = MemoryRef("relay", "right_mem", 0)
    bob = MemoryRef("bob", "mem", 0)
    left_pair = EntangledPairRecord(
        "pair:left",
        alice,
        relay_left,
        left_occupancy_token=101,
        right_occupancy_token=102,
    )
    right_pair = EntangledPairRecord(
        "pair:right",
        relay_right,
        bob,
        left_occupancy_token=201,
        right_occupancy_token=202,
    )
    report = DetectionReport(
        report_id="bsa:1",
        device_id="relay.bsa",
        time=20,
        success=True,
        outcome="psi+",
        raw_clicks=(),
    )

    swapped = swapped_pair_from_bsa(
        "pair:swapped",
        left_pair,
        right_pair,
        left_outer=alice,
        right_outer=bob,
        report=report,
        fidelity=0.9,
        created_at=20,
        metadata=(("registered_by", "controller"),),
    )

    assert swapped.left == alice
    assert swapped.right == bob
    assert swapped.left_occupancy_token == 101
    assert swapped.right_occupancy_token == 202
    assert swapped.metadata == (
        ("registered_by", "controller"),
        ("source_pair_ids", ("pair:left", "pair:right")),
        ("bsa_report_id", "bsa:1"),
        ("bsa_outcome", "psi+"),
    )


def test_swapped_pair_from_bsa_rejects_failed_report_and_missing_outer_refs() -> None:
    left_pair = EntangledPairRecord(
        "pair:left",
        MemoryRef("alice", "mem", 0),
        MemoryRef("relay", "left_mem", 0),
    )
    right_pair = EntangledPairRecord(
        "pair:right",
        MemoryRef("relay", "right_mem", 0),
        MemoryRef("bob", "mem", 0),
    )
    success = DetectionReport("bsa:1", "relay.bsa", 20, True, "phi+", ())
    failed = DetectionReport("bsa:failed", "relay.bsa", 20, False, None, ())

    with pytest.raises(ValueError, match="report must indicate success"):
        swapped_pair_from_bsa(
            "pair:swapped",
            left_pair,
            right_pair,
            left_outer=left_pair.left,
            right_outer=right_pair.right,
            report=failed,
        )

    with pytest.raises(ValueError, match="left_outer not found"):
        swapped_pair_from_bsa(
            "pair:swapped",
            left_pair,
            right_pair,
            left_outer=MemoryRef("carol", "mem", 0),
            right_outer=right_pair.right,
            report=success,
        )
