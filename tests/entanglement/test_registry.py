from __future__ import annotations

import pytest

from simyuj.entanglement import EntangledPairRecord, EntangledPairRegistry, PairState
from simyuj.resources import MemoryRef, MemorySlotState, ResourceManager


def ref(node_id: str, position: int = 0) -> MemoryRef:
    return MemoryRef(node_id, "qmem", position)


def pair(
    pair_id: str,
    left: MemoryRef | None = None,
    right: MemoryRef | None = None,
    *,
    state: PairState = PairState.AVAILABLE,
    fidelity: float | None = None,
    expires_at: int | None = None,
    generation_link_id: str | None = None,
) -> EntangledPairRecord:
    return EntangledPairRecord(
        pair_id=pair_id,
        left=left or ref("alice"),
        right=right or ref("bob"),
        state=state,
        fidelity=fidelity,
        expires_at=expires_at,
        generation_link_id=generation_link_id,
    )


def test_register_stores_pair_and_get_retrieves_it() -> None:
    registry = EntangledPairRegistry()
    entangled_pair = pair("pair:0")

    assert registry.register(entangled_pair) == entangled_pair
    assert registry.get("pair:0") == entangled_pair
    assert registry.pairs["pair:0"] == entangled_pair


def test_register_rejects_duplicate_pair_id() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0"))

    with pytest.raises(ValueError, match="already exists"):
        registry.register(pair("pair:0", ref("carol"), ref("dave")))


def test_active_pair_cannot_reuse_left_memory_of_another_active_pair() -> None:
    registry = EntangledPairRegistry()
    left = ref("alice")
    registry.register(pair("pair:0", left, ref("bob")))

    with pytest.raises(ValueError, match="already used"):
        registry.register(pair("pair:1", left, ref("carol")))


def test_active_pair_cannot_reuse_right_memory_of_another_active_pair() -> None:
    registry = EntangledPairRegistry()
    right = ref("bob")
    registry.register(pair("pair:0", ref("alice"), right))

    with pytest.raises(ValueError, match="already used"):
        registry.register(pair("pair:1", ref("carol"), right))


def test_terminal_pair_does_not_block_memory_reuse() -> None:
    registry = EntangledPairRegistry()
    left = ref("alice")
    right = ref("bob")
    registry.register(pair("pair:0", left, right, state=PairState.CONSUMED))
    replacement = pair("pair:1", left, right)

    assert registry.register(replacement) == replacement


def test_pair_using_memory_returns_active_pair() -> None:
    registry = EntangledPairRegistry()
    left = ref("alice")
    registry.register(pair("pair:0", left, ref("bob"), state=PairState.CONSUMED))
    active = registry.register(pair("pair:1", left, ref("carol")))

    assert registry.pair_using_memory(left) == active


def test_pairs_using_memory_includes_terminal_historical_records() -> None:
    registry = EntangledPairRegistry()
    left = ref("alice")
    consumed = registry.register(
        pair("pair:0", left, ref("bob"), state=PairState.CONSUMED)
    )
    active = registry.register(pair("pair:1", left, ref("carol")))

    assert registry.pairs_using_memory(left) == (consumed, active)


def test_available_between_is_left_right_independent() -> None:
    registry = EntangledPairRegistry()
    first = registry.register(pair("pair:0", ref("alice"), ref("bob")))
    second = registry.register(pair("pair:1", ref("bob", 1), ref("alice", 1)))
    registry.register(pair("pair:2", ref("alice", 2), ref("carol")))

    assert registry.available_between("alice", "bob") == (first, second)
    assert registry.available_between("bob", "alice") == (first, second)


def test_available_for_memory_refs_is_left_right_independent() -> None:
    registry = EntangledPairRegistry()
    left = ref("alice")
    right = ref("bob")
    entangled_pair = registry.register(pair("pair:0", left, right))

    assert registry.available_for_memory_refs(left, right) == (entangled_pair,)
    assert registry.available_for_memory_refs(right, left) == (entangled_pair,)


def test_available_for_memory_refs_rejects_identical_refs() -> None:
    registry = EntangledPairRegistry()
    memory_ref = ref("alice")

    with pytest.raises(ValueError, match="must differ"):
        registry.available_for_memory_refs(memory_ref, memory_ref)


def test_reserve_only_works_from_available() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0"))
    registry.register(
        pair("pair:1", ref("carol"), ref("dave"), state=PairState.RESERVED)
    )

    assert registry.reserve("pair:0").state is PairState.RESERVED

    with pytest.raises(ValueError, match="only available"):
        registry.reserve("pair:1")


def test_release_only_works_from_reserved() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0", state=PairState.RESERVED))
    registry.register(pair("pair:1", ref("carol"), ref("dave")))

    assert registry.release("pair:0").state is PairState.AVAILABLE

    with pytest.raises(ValueError, match="only reserved"):
        registry.release("pair:1")


def test_consume_works_from_available_and_reserved() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0"))
    registry.register(
        pair("pair:1", ref("carol"), ref("dave"), state=PairState.RESERVED)
    )

    assert registry.consume("pair:0").state is PairState.CONSUMED
    assert registry.consume("pair:1").state is PairState.CONSUMED


def test_expire_before_expires_active_pairs_at_or_before_now() -> None:
    registry = EntangledPairRegistry()
    pair_0 = registry.register(pair("pair:0", expires_at=10))
    pair_1 = registry.register(
        pair(
            "pair:1",
            ref("carol"),
            ref("dave"),
            state=PairState.RESERVED,
            expires_at=11,
        )
    )
    registry.register(pair("pair:2", ref("erin"), ref("frank"), expires_at=12))

    expired = registry.expire_before(11)

    assert expired == (pair_0.expired(), pair_1.expired())
    assert registry.get("pair:0").state is PairState.EXPIRED
    assert registry.get("pair:1").state is PairState.EXPIRED
    assert registry.get("pair:2").state is PairState.AVAILABLE


def test_expire_before_ignores_terminal_pairs() -> None:
    registry = EntangledPairRegistry()
    terminal = registry.register(pair("pair:0", state=PairState.CONSUMED, expires_at=1))

    assert registry.expire_before(10) == ()
    assert registry.get("pair:0") == terminal


def test_available_between_min_fidelity_excludes_unknown_fidelity_pairs() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0", ref("alice"), ref("bob")))

    assert registry.available_between("alice", "bob", min_fidelity=0.9) == ()


def test_available_between_min_fidelity_excludes_lower_fidelity_pairs() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0", ref("alice"), ref("bob"), fidelity=0.89))

    assert registry.available_between("alice", "bob", min_fidelity=0.9) == ()


def test_available_between_min_fidelity_includes_equal_or_higher_pairs() -> None:
    registry = EntangledPairRegistry()
    equal = registry.register(pair("pair:0", ref("alice"), ref("bob"), fidelity=0.9))
    higher = registry.register(
        pair("pair:1", ref("alice", 1), ref("bob", 1), fidelity=0.95)
    )
    registry.register(pair("pair:2", ref("alice", 2), ref("bob", 2), fidelity=0.8))

    assert registry.available_between("bob", "alice", min_fidelity=0.9) == (
        equal,
        higher,
    )


def test_reserved_pairs_returns_reserved_records() -> None:
    registry = EntangledPairRegistry()
    reserved = registry.register(pair("pair:0", state=PairState.RESERVED))
    registry.register(pair("pair:1", ref("carol"), ref("dave")))

    assert registry.reserved_pairs() == (reserved,)


def test_available_between_link_id_filters_pairs() -> None:
    registry = EntangledPairRegistry()
    registry.register(pair("pair:0", ref("alice"), ref("bob")))
    match = registry.register(
        pair("pair:1", ref("alice", 1), ref("bob", 1), generation_link_id="link-1")
    )
    registry.register(
        pair("pair:2", ref("alice", 2), ref("bob", 2), generation_link_id="link-2")
    )

    assert registry.available_between("alice", "bob", link_id="link-1") == (match,)


def test_consumed_and_expired_resource_slots_have_no_available_pairs() -> None:
    manager = ResourceManager()
    alice = manager.register_memory("alice", "qmem", num_positions=1)[0]
    bob = manager.register_memory("bob", "qmem", num_positions=1)[0]
    carol = manager.register_memory("carol", "qmem", num_positions=1)[0]
    dave = manager.register_memory("dave", "qmem", num_positions=1)[0]
    manager.reserve_memory_refs(0, (alice, bob), owner="session", reservation_id="r:ab")
    manager.reserve_memory_refs(
        0, (carol, dave), owner="session", reservation_id="r:cd"
    )
    for memory_ref in (alice, bob, carol, dave):
        manager.mark_occupied(memory_ref)

    registry = EntangledPairRegistry()
    registry.register(pair("pair:consumed", alice, bob))
    registry.register(pair("pair:expired", carol, dave))

    consumed = registry.consume("pair:consumed")
    expired = registry.expire("pair:expired")
    for memory_ref in consumed.memory_refs:
        assert manager.mark_consumed(memory_ref).state is MemorySlotState.CONSUMED
    for memory_ref in expired.memory_refs:
        assert manager.mark_expired(memory_ref).state is MemorySlotState.EXPIRED

    assert registry.available_between("alice", "bob") == ()
    assert registry.available_between("carol", "dave") == ()
    assert registry.pair_using_memory(alice) is None
    assert registry.pair_using_memory(carol) is None


def test_reused_memory_position_uses_new_occupancy_token_not_stale_pair() -> None:
    registry = EntangledPairRegistry()
    alice = ref("alice")
    bob = ref("bob")
    stale = registry.register(
        pair(
            "pair:old",
            alice,
            bob,
            state=PairState.CONSUMED,
        )
    )
    fresh = registry.register(
        EntangledPairRecord(
            "pair:new",
            alice,
            bob,
            left_occupancy_token=7,
            right_occupancy_token=8,
        )
    )

    assert registry.pairs_using_memory(alice) == (fresh, stale)
    assert registry.pair_using_memory(alice) == fresh
    assert registry.available_for_memory_refs(alice, bob) == (fresh,)
    assert stale.left_occupancy_token is None
    assert fresh.left_occupancy_token == 7
