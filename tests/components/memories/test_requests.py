from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from simyuj.components.memories import (
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryExpireRequest,
    MemoryMeasureRequest,
    MemoryUpdateMetaRequest,
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


def _field_names(record_type: type[Any]) -> set[str]:
    return {field.name for field in fields(record_type)}


def test_absorb_request_targets_signal_and_optional_position() -> None:
    signal = _signal()

    request = MemoryAbsorbRequest(
        request_id="absorb-1",
        memory_id="nodeA.mem0",
        signal=signal,
        position=None,
        session_id="session-1",
        meta=(("port", "qin"),),
    )

    assert request.signal is signal
    assert request.position is None
    assert request.session_id == "session-1"
    assert request.meta == (("port", "qin"),)


def test_emit_request_uses_position_not_slot_key() -> None:
    request = MemoryEmitRequest(
        request_id="emit-1",
        memory_id="nodeA.mem0",
        position=2,
    )

    assert request.position == 2
    assert "slot_key" not in _field_names(MemoryEmitRequest)


def test_apply_operator_request_preserves_ordered_positions() -> None:
    operator = object()

    request = MemoryApplyOperatorRequest(
        request_id="apply-1",
        memory_id="nodeA.mem0",
        positions=(2, 0, 1),
        operator=operator,
    )

    assert request.positions == (2, 0, 1)
    assert request.operator is operator


def test_measure_request_defaults_separate_collapse_and_destructive() -> None:
    request = MemoryMeasureRequest(
        request_id="measure-1",
        memory_id="nodeA.mem0",
        positions=(0,),
    )

    assert request.measurement == "z"
    assert request.collapse is True
    assert request.destructive is True


def test_measure_request_allows_independent_collapse_and_destructive() -> None:
    request = MemoryMeasureRequest(
        request_id="measure-1",
        memory_id="nodeA.mem0",
        positions=(0,),
        collapse=False,
        destructive=True,
    )

    assert request.collapse is False
    assert request.destructive is True


def test_discard_request_defaults_reason() -> None:
    request = MemoryDiscardRequest(
        request_id="discard-1",
        memory_id="nodeA.mem0",
        position=1,
    )

    assert request.reason == "discarded"


def test_expire_request_carries_occupancy_token() -> None:
    request = MemoryExpireRequest(
        request_id="expire-1",
        memory_id="nodeA.mem0",
        position=1,
        occupancy_token=9,
    )

    assert request.position == 1
    assert request.occupancy_token == 9


def test_update_meta_request_carries_metadata_edits_and_token() -> None:
    request = MemoryUpdateMetaRequest(
        request_id="meta-1",
        memory_id="nodeA.mem0",
        position=1,
        updates=(("pair_id", "ab-1"),),
        remove_keys=("old_pair",),
        expected_occupancy_token=9,
        session_id="session-1",
        meta=(("source", "swap"),),
    )

    assert request.position == 1
    assert request.updates == (("pair_id", "ab-1"),)
    assert request.remove_keys == ("old_pair",)
    assert request.expected_occupancy_token == 9
    assert request.session_id == "session-1"
    assert request.meta == (("source", "swap"),)


def test_update_meta_request_rejects_duplicate_update_keys() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        MemoryUpdateMetaRequest(
            request_id="meta-1",
            memory_id="nodeA.mem0",
            position=0,
            updates=(("pair_id", "A"), ("pair_id", "B")),
        )


def test_update_meta_request_rejects_duplicate_remove_keys() -> None:
    with pytest.raises(ValueError, match="duplicate keys"):
        MemoryUpdateMetaRequest(
            request_id="meta-1",
            memory_id="nodeA.mem0",
            position=0,
            remove_keys=("pair_id", "pair_id"),
        )


@pytest.mark.parametrize(
    "payload",
    (
        MemoryAbsorbRequest("r", "m", _signal(), position=0),
        MemoryEmitRequest("r", "m", position=0),
        MemoryApplyOperatorRequest("r", "m", positions=(0,), operator=object()),
        MemoryMeasureRequest("r", "m", positions=(0,)),
        MemoryDiscardRequest("r", "m", position=0),
        MemoryExpireRequest("r", "m", position=0, occupancy_token=1),
        MemoryUpdateMetaRequest("r", "m", position=0),
    ),
)
def test_memory_requests_do_not_expose_slot_key(payload: object) -> None:
    assert not hasattr(payload, "slot_key")


def test_apply_operator_request_rejects_empty_or_duplicate_positions() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MemoryApplyOperatorRequest(
            request_id="apply-1",
            memory_id="nodeA.mem0",
            positions=(),
            operator=object(),
        )

    with pytest.raises(ValueError, match="unique"):
        MemoryApplyOperatorRequest(
            request_id="apply-1",
            memory_id="nodeA.mem0",
            positions=(0, 0),
            operator=object(),
        )


def test_absorb_request_rejects_non_signal() -> None:
    with pytest.raises(TypeError, match="Signal"):
        MemoryAbsorbRequest(
            request_id="absorb-1",
            memory_id="nodeA.mem0",
            signal=object(),  # type: ignore[arg-type]
        )


def test_requests_validate_common_ids_and_meta() -> None:
    with pytest.raises(ValueError, match="request_id"):
        MemoryEmitRequest("", "nodeA.mem0", position=0)

    with pytest.raises(ValueError, match="memory_id"):
        MemoryEmitRequest("emit-1", "", position=0)

    with pytest.raises(TypeError, match="meta"):
        MemoryEmitRequest(
            "emit-1",
            "nodeA.mem0",
            position=0,
            meta=(("bad", []),),
        )
