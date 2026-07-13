from __future__ import annotations

import pytest

from simyuj.entanglement import EntangledPairRecord, PairState
from simyuj.resources import MemoryRef


def make_pair(
    *,
    state: PairState = PairState.AVAILABLE,
    fidelity: float | None = None,
) -> EntangledPairRecord:
    return EntangledPairRecord(
        pair_id="pair:0",
        left=MemoryRef("alice", "qmem", 0),
        right=MemoryRef("bob", "qmem", 0),
        state=state,
        fidelity=fidelity,
        created_at=3,
        expires_at=9,
        generation_link_id="q_ab",
        metadata=(("source", "test"),),
    )


def test_entangled_pair_record_stores_normalized_metadata() -> None:
    pair = make_pair(fidelity=1)

    assert pair.pair_id == "pair:0"
    assert pair.fidelity == 1.0
    assert pair.generation_link_id == "q_ab"
    assert pair.memory_refs == (
        MemoryRef("alice", "qmem", 0),
        MemoryRef("bob", "qmem", 0),
    )
    assert pair.memory_ref_keys == (
        ("alice", "qmem", 0),
        ("bob", "qmem", 0),
    )
    assert pair.node_ids == ("alice", "bob")
    assert pair.has_fidelity


def test_pair_state_predicates() -> None:
    assert make_pair(state=PairState.AVAILABLE).is_available
    assert make_pair(state=PairState.AVAILABLE).is_active
    assert make_pair(state=PairState.RESERVED).is_active
    assert not make_pair(state=PairState.RESERVED).is_available

    for state in (PairState.CONSUMED, PairState.EXPIRED, PairState.FAILED):
        pair = make_pair(state=state)
        assert pair.is_terminal
        assert not pair.is_active


def test_pair_state_transitions_return_new_records() -> None:
    pair = make_pair()

    assert pair.reserved().state is PairState.RESERVED
    assert pair.reserved().available().state is PairState.AVAILABLE
    assert pair.consumed().state is PairState.CONSUMED
    assert pair.expired().state is PairState.EXPIRED
    assert pair.failed().state is PairState.FAILED
    assert pair.state is PairState.AVAILABLE

    with pytest.raises(TypeError, match="PairState"):
        pair.with_state("available")  # type: ignore[arg-type]


def test_pair_memory_and_node_helpers_are_order_insensitive() -> None:
    pair = make_pair()
    left = MemoryRef("alice", "qmem", 0)
    right = MemoryRef("bob", "qmem", 0)
    unrelated = MemoryRef("carol", "qmem", 0)

    assert pair.uses_memory(left)
    assert pair.other_memory(left) == right
    assert pair.has_node("alice")
    assert not pair.has_node("carol")
    assert pair.connects_nodes("alice", "bob")
    assert pair.connects_nodes("bob", "alice")
    assert pair.connects_memory_refs(left, right)
    assert pair.connects_memory_refs(right, left)
    assert not pair.connects_memory_refs(left, unrelated)

    with pytest.raises(ValueError, match="not part of pair"):
        pair.other_memory(unrelated)


def test_pair_record_validates_constructor_inputs() -> None:
    left = MemoryRef("alice", "qmem", 0)
    right = MemoryRef("bob", "qmem", 0)

    with pytest.raises(ValueError, match="must differ"):
        EntangledPairRecord("pair:0", left, left)

    with pytest.raises(TypeError, match="left must be MemoryRef"):
        EntangledPairRecord("pair:0", "alice", right)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="state must be PairState"):
        EntangledPairRecord(
            "pair:0",
            left,
            right,
            state="available",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        EntangledPairRecord("pair:0", left, right, fidelity=1.1)

    with pytest.raises(ValueError, match="expires_at"):
        EntangledPairRecord("pair:0", left, right, created_at=5, expires_at=4)

    with pytest.raises(TypeError, match="metadata"):
        EntangledPairRecord(
            "pair:0",
            left,
            right,
            metadata=("bad",),  # type: ignore[arg-type]
        )
