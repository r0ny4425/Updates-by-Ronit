from __future__ import annotations

from dataclasses import replace

import pytest

from simyuj.components.memories import (
    MemoryPositionRecord,
    MemoryPositionStatus,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from simyuj.signal import EncodingScheme, Signal, SignalKind


def _signal() -> Signal:
    return Signal(
        id="signal-1",
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=3,
        origin="source",
    )


def test_memory_position_defaults_to_empty() -> None:
    record = MemoryPositionRecord(position=0)

    assert record.status is MemoryPositionStatus.EMPTY
    assert record.memory_subsystem is None
    assert record.stored_signal is None
    assert record.stored_time is None
    assert record.last_noise_update_time is None
    assert record.expires_at is None
    assert record.occupancy_token == 0
    assert record.ready_at == 0
    assert record.meta == ()


def test_occupied_position_stores_memory_subsystem_and_timing_fields() -> None:
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    signal = _signal()

    record = MemoryPositionRecord(
        position=0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem,
        stored_signal=signal,
        stored_time=10,
        last_noise_update_time=12,
        expires_at=40,
        occupancy_token=7,
        ready_at=9,
        meta=(("basis", "z"),),
    )

    assert record.memory_subsystem == memory_subsystem
    assert record.stored_signal is signal
    assert record.stored_time == 10
    assert record.last_noise_update_time == 12
    assert record.expires_at == 40
    assert record.occupancy_token == 7
    assert record.ready_at == 9


def test_position_ready_at_is_non_negative_plain_int() -> None:
    with pytest.raises(ValueError, match="ready_at"):
        MemoryPositionRecord(position=0, ready_at=-1)

    with pytest.raises(TypeError, match="ready_at"):
        MemoryPositionRecord(position=0, ready_at=True)


def test_position_status_changes_are_explicit() -> None:
    occupied = MemoryPositionRecord(
        position=0,
        status=MemoryPositionStatus.OCCUPIED,
        memory_subsystem=memory_subsystem_id("nodeA.mem0", 0),
        stored_time=10,
        last_noise_update_time=10,
    )

    emitting = replace(occupied, status=MemoryPositionStatus.EMITTING)
    measuring = replace(occupied, status=MemoryPositionStatus.MEASURING)
    applying = replace(occupied, status=MemoryPositionStatus.APPLYING_OPERATOR)

    assert occupied.status is MemoryPositionStatus.OCCUPIED
    assert emitting.status is MemoryPositionStatus.EMITTING
    assert measuring.status is MemoryPositionStatus.MEASURING
    assert applying.status is MemoryPositionStatus.APPLYING_OPERATOR


def test_absorbing_position_requires_only_pending_signal_fields() -> None:
    signal = _signal()

    record = MemoryPositionRecord(
        position=0,
        status=MemoryPositionStatus.ABSORBING,
        stored_signal=signal,
    )

    assert record.stored_signal is signal
    assert record.memory_subsystem is None
    assert record.stored_time is None
    assert record.last_noise_update_time is None
    assert record.expires_at is None


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("memory_subsystem", "memory_subsystem"),
        ("stored_time", "stored_time"),
        ("last_noise_update_time", "last_noise_update_time"),
        ("expires_at", "expires_at"),
    ),
)
def test_absorbing_position_rejects_stored_state_fields(
    field: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        if field == "memory_subsystem":
            MemoryPositionRecord(
                position=0,
                status=MemoryPositionStatus.ABSORBING,
                stored_signal=_signal(),
                memory_subsystem=memory_subsystem_id("nodeA.mem0", 0),
            )
        elif field == "stored_time":
            MemoryPositionRecord(
                position=0,
                status=MemoryPositionStatus.ABSORBING,
                stored_signal=_signal(),
                stored_time=10,
            )
        elif field == "last_noise_update_time":
            MemoryPositionRecord(
                position=0,
                status=MemoryPositionStatus.ABSORBING,
                stored_signal=_signal(),
                last_noise_update_time=10,
            )
        elif field == "expires_at":
            MemoryPositionRecord(
                position=0,
                status=MemoryPositionStatus.ABSORBING,
                stored_signal=_signal(),
                expires_at=20,
            )
        else:
            raise AssertionError(f"unsupported field: {field}")


def test_absorbing_position_requires_stored_signal() -> None:
    with pytest.raises(ValueError, match="requires stored_signal"):
        MemoryPositionRecord(
            position=0,
            status=MemoryPositionStatus.ABSORBING,
        )


def test_memory_subsystem_label_is_stable_for_position_reuse() -> None:
    first = memory_subsystem_id("nodeA.mem0", 0)
    reused = memory_subsystem_id("nodeA.mem0", 0)

    assert str(first) == "memory:nodeA.mem0:position:0"
    assert reused == first


def test_emitted_photon_label_is_unique_by_emission_counter() -> None:
    emitted_7 = emitted_photon_subsystem_id("nodeA.mem0", 0, 7)
    emitted_8 = emitted_photon_subsystem_id("nodeA.mem0", 0, 8)

    assert str(emitted_7) == "photon:nodeA.mem0:position:0:emit:7"
    assert emitted_8 != emitted_7


def test_stored_position_status_requires_memory_subsystem() -> None:
    with pytest.raises(ValueError, match="requires memory_subsystem"):
        MemoryPositionRecord(
            position=0,
            status=MemoryPositionStatus.OCCUPIED,
        )


@pytest.mark.parametrize(
    "status",
    (
        MemoryPositionStatus.OCCUPIED,
        MemoryPositionStatus.EMITTING,
        MemoryPositionStatus.MEASURING,
        MemoryPositionStatus.APPLYING_OPERATOR,
    ),
)
@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("memory_subsystem", "memory_subsystem"),
        ("stored_time", "stored_time"),
        ("last_noise_update_time", "last_noise_update_time"),
    ),
)
def test_stored_position_statuses_require_storage_fields(
    status: MemoryPositionStatus,
    field: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        if field == "memory_subsystem":
            MemoryPositionRecord(
                position=0,
                status=status,
                memory_subsystem=None,
                stored_time=10,
                last_noise_update_time=10,
            )
        elif field == "stored_time":
            MemoryPositionRecord(
                position=0,
                status=status,
                memory_subsystem=memory_subsystem_id("nodeA.mem0", 0),
                stored_time=None,
                last_noise_update_time=10,
            )
        elif field == "last_noise_update_time":
            MemoryPositionRecord(
                position=0,
                status=status,
                memory_subsystem=memory_subsystem_id("nodeA.mem0", 0),
                stored_time=10,
                last_noise_update_time=None,
            )
        else:
            raise AssertionError(f"unsupported field: {field}")


def test_empty_position_rejects_stored_fields() -> None:
    with pytest.raises(ValueError, match="empty position"):
        MemoryPositionRecord(
            position=0,
            memory_subsystem=memory_subsystem_id("nodeA.mem0", 0),
        )

    with pytest.raises(ValueError, match="last_noise_update_time"):
        MemoryPositionRecord(position=0, last_noise_update_time=10)
