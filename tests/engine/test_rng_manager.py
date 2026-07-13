import pytest

from simyuj.engine.rng_manager import RNGManager
from simyuj.engine.timeline import Timeline

OWNER = "timeline"
OTHER_OWNER = "not_timeline"


# ─────────────────────────────────────────────
# Construction & ownership
# ─────────────────────────────────────────────


def test_rng_manager_requires_non_negative_int_seed():
    with pytest.raises(ValueError):
        RNGManager(master_seed=-1, owner_id=OWNER)

    with pytest.raises(ValueError):
        RNGManager(master_seed="42", owner_id=OWNER)


def test_rng_access_requires_correct_owner():
    mgr = RNGManager(master_seed=123, owner_id=OWNER)

    with pytest.raises(PermissionError):
        mgr.rng("alice", requester_id=OTHER_OWNER)


# ─────────────────────────────────────────────
# Determinism: same seed + same stream
# ─────────────────────────────────────────────


def test_same_seed_same_stream_produces_identical_sequence():
    mgr1 = RNGManager(master_seed=42, owner_id=OWNER)
    mgr2 = RNGManager(master_seed=42, owner_id=OWNER)

    r1 = mgr1.rng("alice", "basis", requester_id=OWNER)
    r2 = mgr2.rng("alice", "basis", requester_id=OWNER)

    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]

    assert seq1 == seq2


# ─────────────────────────────────────────────
# Determinism: same seed + different streams
# ─────────────────────────────────────────────


def test_same_seed_different_streams_are_independent():
    mgr = RNGManager(master_seed=42, owner_id=OWNER)

    r1 = mgr.rng("alice", requester_id=OWNER)
    r2 = mgr.rng("bob", requester_id=OWNER)

    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]

    assert seq1 != seq2


# ─────────────────────────────────────────────
# Determinism: different seeds
# ─────────────────────────────────────────────


def test_different_seeds_produce_different_sequences():
    mgr1 = RNGManager(master_seed=1, owner_id=OWNER)
    mgr2 = RNGManager(master_seed=2, owner_id=OWNER)

    r1 = mgr1.rng("alice", requester_id=OWNER)
    r2 = mgr2.rng("alice", requester_id=OWNER)

    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]

    assert seq1 != seq2


# ─────────────────────────────────────────────
# Stream identity & caching
# ─────────────────────────────────────────────


def test_requesting_same_stream_returns_same_object():
    mgr = RNGManager(master_seed=42, owner_id=OWNER)

    r1 = mgr.rng("alice", "detector", requester_id=OWNER)
    r2 = mgr.rng("alice", "detector", requester_id=OWNER)

    assert r1 is r2


def test_list_streams_returns_sorted_paths():
    mgr = RNGManager(master_seed=42, owner_id=OWNER)

    mgr.rng("bob", requester_id=OWNER)
    mgr.rng("alice", requester_id=OWNER)
    mgr.rng("alice", "basis", requester_id=OWNER)

    assert mgr.list_streams() == (
        ("alice",),
        ("alice", "basis"),
        ("bob",),
    )


# ─────────────────────────────────────────────
# Freeze semantics
# ─────────────────────────────────────────────


def test_freeze_prevents_new_streams():
    mgr = RNGManager(master_seed=42, owner_id=OWNER)

    mgr.rng("alice", requester_id=OWNER)
    mgr.freeze()

    # Existing stream is allowed
    mgr.rng("alice", requester_id=OWNER)

    # New stream is forbidden
    with pytest.raises(RuntimeError):
        mgr.rng("bob", requester_id=OWNER)


# ─────────────────────────────────────────────
# DeterministicRNG restrictions
# ─────────────────────────────────────────────


def test_deterministic_rng_is_immutable_and_restricted():
    mgr = RNGManager(master_seed=42, owner_id=OWNER)
    rng = mgr.rng("alice", requester_id=OWNER)

    with pytest.raises(AttributeError):
        rng.seed = 123

    with pytest.raises(AttributeError):
        _ = rng.bit_generator

    with pytest.raises(AttributeError):
        rng.__dict__


def test_only_owner_can_request_rng():
    m = RNGManager(master_seed=1, owner_id=OWNER)

    with pytest.raises(PermissionError):
        m.rng("alice", requester_id=OTHER_OWNER)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("",),
        ("   ",),
        ("alice/bob",),
        ("alice\\bob",),
    ],
)
def test_invalid_rng_paths_raise(path):
    m = RNGManager(master_seed=1, owner_id="timeline")

    with pytest.raises(ValueError):
        m.rng(*path, requester_id="timeline")


def test_rng_is_immutable():
    m = RNGManager(master_seed=1, owner_id="timeline")
    r = m.rng("alice", requester_id="timeline")

    with pytest.raises(AttributeError):
        r.foo = 123


def test_rng_blocks_unauthorized_methods():
    m = RNGManager(master_seed=1, owner_id="timeline")
    r = m.rng("alice", requester_id="timeline")

    with pytest.raises(AttributeError):
        r.bit_generator


def test_randint_is_deterministic():
    m1 = RNGManager(master_seed=7, owner_id="timeline")
    m2 = RNGManager(master_seed=7, owner_id="timeline")

    r1 = m1.rng("test", requester_id="timeline")
    r2 = m2.rng("test", requester_id="timeline")

    seq1 = [r1.randint(1, 10) for _ in range(3)]
    seq2 = [r2.randint(1, 10) for _ in range(3)]

    assert seq1 == seq2


# ────────────────Timeline based rng tests─────────────────────────────


def test_timeline_accepts_master_seed():
    timeline = Timeline(master_seed=12345)

    rng = timeline.rng("test")

    value = rng.random()
    assert isinstance(value, float)


def test_default_master_seed_is_deterministic():
    """Timeline default master seed should produce deterministic results."""
    t1 = Timeline()
    t2 = Timeline()

    r1 = t1.rng("test")
    r2 = t2.rng("test")

    seq1 = [r1.random() for _ in range(5)]
    seq2 = [r2.random() for _ in range(5)]

    assert seq1 == seq2


def test_hierarchical_naming():
    """Should support hierarchical stream naming."""
    timeline = Timeline(master_seed=42)

    # Different hierarchy levels
    rng1 = timeline.rng("alice")
    rng2 = timeline.rng("alice", "detector")
    rng3 = timeline.rng("alice", "detector", "noise")

    # All should be different streams
    val1 = rng1.random()
    val2 = rng2.random()
    val3 = rng3.random()

    assert val1 != val2
    assert val2 != val3
    assert val1 != val3


def test_order_independence():
    """Stream creation order should not affect sequences."""
    # Scenario A: Create Alice, then Bob
    timeline_A = Timeline(master_seed=12345)
    alice_A = timeline_A.rng("alice")
    bob_A = timeline_A.rng("bob")

    alice_seq_A = [alice_A.random() for _ in range(5)]
    bob_seq_A = [bob_A.random() for _ in range(5)]

    # Scenario B: Create Bob, then Alice (reversed order)
    timeline_B = Timeline(master_seed=12345)
    bob_B = timeline_B.rng("bob")
    alice_B = timeline_B.rng("alice")

    bob_seq_B = [bob_B.random() for _ in range(5)]
    alice_seq_B = [alice_B.random() for _ in range(5)]

    # Sequences should match regardless of creation order
    assert alice_seq_A == alice_seq_B
    assert bob_seq_A == bob_seq_B


def test_many_streams_are_distinct_objects():
    """Each RNG path must produce a distinct RNG stream object."""
    timeline = Timeline(master_seed=42)

    streams = [timeline.rng(f"stream_{i}") for i in range(1000)]

    # Identity-based guarantee (deterministic, non-flaky)
    assert len({id(rng) for rng in streams}) == 1000
