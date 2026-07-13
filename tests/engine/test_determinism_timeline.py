"""
Determinism Test Suite for Timeline

This suite validates that Timeline execution is deterministic across:
- Event ordering (insertion-order independence)
- Cancellation semantics
- RNG determinism
- Replay safety (incremental vs one-shot execution)

All tests use ONLY public APIs and validate observable behavior.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline

# ─────────────────────────────────────────────
# Mock Event Handler (Test Double)
# ─────────────────────────────────────────────


@dataclass
class MockHandler:
    """
    Deterministic event handler that records execution traces.

    This is a test double that implements the handle_event protocol
    expected by Timeline.
    """

    name: str
    trace: List[Tuple[int, int, str]] = field(default_factory=list)

    def handle_event(self, event, timeline):
        """
        Record event execution in trace.

        Trace format: (time, event_id, action)
        """
        self.trace.append((event.time, event.event_id, event.action))

    def get_trace(self) -> Tuple[Tuple[int, int, str], ...]:
        """Return immutable trace for comparison."""
        return tuple(self.trace)

    def clear_trace(self) -> None:
        """Reset trace (for reuse in tests)."""
        self.trace.clear()


def strip_ids(trace):
    "return trace tuple with event_id removed"
    return [(t, action) for (t, _, action) in trace]


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def handler():
    """Provide a fresh MockHandler for each test."""
    return MockHandler(name="test_handler")


@pytest.fixture
def timeline():
    """Provide a fresh Timeline with deterministic seed."""
    return Timeline(master_seed=42)


# ─────────────────────────────────────────────
# TEST GROUP 1: Event Ordering Determinism
# ─────────────────────────────────────────────


@given(actions=st.permutations(["A", "B", "C", "D", "E"]))
@settings(max_examples=40)
def test_insertion_order_does_not_affect_execution_order(actions):
    """
    PROPERTY: Insertion order does not affect execution order
    when time and priority are identical.
    """
    handler = MockHandler("handler")
    tl = Timeline(master_seed=42)

    for action in actions:
        tl.schedule(
            Event(
                time=0,
                priority=0,
                target_ref=handler,
                action=action,
                payload_ref=None,
            )
        )

    tl.run_until(10)

    event_ids = [eid for _, eid, _ in handler.get_trace()]
    assert event_ids == sorted(event_ids)


def test_priority_tiebreaking_with_event_id(handler):
    """
    PROPERTY: When time and priority match, events execute in
    event_id order.

    VALIDATION: Multiple events with same time and priority
    execute in the order they were scheduled.
    """

    tl = Timeline(master_seed=42)

    # Schedule 3 events with identical time and priority
    tl.schedule(
        Event(time=10, priority=0, target_ref=handler, action="FIRST", payload_ref=None)
    )
    tl.schedule(
        Event(
            time=10, priority=0, target_ref=handler, action="SECOND", payload_ref=None
        )
    )
    tl.schedule(
        Event(time=10, priority=0, target_ref=handler, action="THIRD", payload_ref=None)
    )

    tl.run_until(10)
    trace = handler.get_trace()

    # ASSERT: Events execute in scheduling order
    assert trace[0][2] == "FIRST"
    assert trace[1][2] == "SECOND"
    assert trace[2][2] == "THIRD"

    # ASSERT: event_ids are sequential
    assert trace[0][1] < trace[1][1] < trace[2][1]


# ─────────────────────────────────────────────
# TEST GROUP 2: Cancellation Semantics
# ─────────────────────────────────────────────


def test_cancelled_events_never_execute(handler):
    """
    PROPERTY: Cancelled events never execute.

    VALIDATION: Event marked as cancelled does not appear in
    execution trace.
    """

    tl = Timeline(master_seed=42)

    tl.schedule(
        Event(
            time=10, priority=0, target_ref=handler, action="EXECUTE", payload_ref=None
        )
    )
    e_cancel = tl.schedule(
        Event(
            time=10,
            priority=1,
            target_ref=handler,
            action="CANCEL_ME",
            payload_ref=None,
        )
    )
    tl.schedule(
        Event(
            time=10, priority=2, target_ref=handler, action="EXECUTE", payload_ref=None
        )
    )

    # Cancel middle event
    tl.cancel(e_cancel)

    tl.run_until(10)
    trace = handler.get_trace()

    # ASSERT: Only 2 events executed
    assert len(trace) == 2

    # ASSERT: Cancelled event did not execute
    actions = [t[2] for t in trace]
    assert "CANCEL_ME" not in actions
    assert actions == ["EXECUTE", "EXECUTE"]


def test_cancellation_does_not_affect_remaining_order(handler):
    """
    PROPERTY: Cancelling events does not affect the order of
    remaining events.

    VALIDATION: Two timelines (one with cancellation, one without)
    execute remaining events in the same order.
    """

    # Timeline 1: No cancellation
    tl1 = Timeline(master_seed=42)
    tl1.schedule(
        Event(time=10, priority=0, target_ref=handler, action="A", payload_ref=None)
    )
    tl1.schedule(
        Event(time=10, priority=2, target_ref=handler, action="C", payload_ref=None)
    )

    handler.clear_trace()
    tl1.run_until(10)
    trace1 = handler.get_trace()

    # Timeline 2: With cancellation of middle event
    tl2 = Timeline(master_seed=42)
    tl2.schedule(
        Event(time=10, priority=0, target_ref=handler, action="A", payload_ref=None)
    )
    e_cancel = tl2.schedule(
        Event(
            time=10, priority=1, target_ref=handler, action="B_CANCEL", payload_ref=None
        )
    )
    tl2.schedule(
        Event(time=10, priority=2, target_ref=handler, action="C", payload_ref=None)
    )

    tl2.cancel(e_cancel)

    handler.clear_trace()
    tl2.run_until(10)
    trace2 = handler.get_trace()

    # ASSERT: Both timelines execute A then C in same order
    assert len(trace1) == 2
    assert len(trace2) == 2
    assert [t[2] for t in trace1] == ["A", "C"]
    assert [t[2] for t in trace2] == ["A", "C"]


@given(cancel_mask=st.lists(st.booleans(), min_size=1, max_size=20))
@settings(max_examples=80)
def test_arbitrary_cancellation_preserves_relative_order(cancel_mask):
    """
    PROPERTY:
    Arbitrary cancellation of events does NOT change the relative
    execution order of the remaining events.
    """

    handler = MockHandler("handler")
    tl = Timeline(master_seed=42)

    events = []
    expected_actions = []

    # Schedule events with identical (time, priority)
    for i, cancel in enumerate(cancel_mask):
        action = f"E{i}"
        e = tl.schedule(
            Event(
                time=10,
                priority=0,
                target_ref=handler,
                action=action,
                payload_ref=None,
            )
        )
        events.append(e)

        if not cancel:
            expected_actions.append(action)

    # Apply cancellations according to mask
    for e, cancel in zip(events, cancel_mask):
        if cancel:
            tl.cancel(e)

    tl.run_until(10)

    executed_actions = [t[2] for t in handler.get_trace()]

    # ASSERT: relative order preserved
    assert executed_actions == expected_actions


def test_cancel_all_events_in_batch(handler):
    """
    PROPERTY: Cancelling all events in a time batch results in
    no execution at that time.

    VALIDATION: Timeline advances time but trace remains empty.
    """

    tl = Timeline(master_seed=42)

    e1 = tl.schedule(
        Event(
            time=10, priority=0, target_ref=handler, action="CANCEL1", payload_ref=None
        )
    )
    e2 = tl.schedule(
        Event(
            time=10, priority=1, target_ref=handler, action="CANCEL2", payload_ref=None
        )
    )
    _ = tl.schedule(
        Event(time=20, priority=0, target_ref=handler, action="KEEP", payload_ref=None)
    )

    # Cancel all events at time=10
    tl.cancel(e1)
    tl.cancel(e2)

    tl.run_until(20)
    trace = handler.get_trace()

    # ASSERT: Only event at time=20 executed
    assert len(trace) == 1
    assert trace[0][0] == 20
    assert trace[0][2] == "KEEP"


# ─────────────────────────────────────────────
# TEST GROUP 3: RNG Determinism
# ─────────────────────────────────────────────


def test_same_seed_produces_identical_random_sequences():
    """
    PROPERTY: Same master seed produces identical random number
    sequences.

    VALIDATION: Two timelines with same seed generate identical
    random values.
    """

    SEED = 12345

    # Timeline 1
    tl1 = Timeline(master_seed=SEED)
    rng1 = tl1.rng("test", "stream")
    values1 = [rng1.random() for _ in range(10)]

    # Timeline 2 (same seed)
    tl2 = Timeline(master_seed=SEED)
    rng2 = tl2.rng("test", "stream")
    values2 = [rng2.random() for _ in range(10)]

    # ASSERT: Sequences are identical
    assert values1 == values2


def test_different_seeds_produce_different_sequences():
    """
    PROPERTY: Different master seeds produce different random
    sequences.

    VALIDATION: Two timelines with different seeds generate
    different random values.
    """

    # Timeline 1
    tl1 = Timeline(master_seed=111)
    rng1 = tl1.rng("test", "stream")
    values1 = [rng1.random() for _ in range(10)]

    # Timeline 2 (different seed)
    tl2 = Timeline(master_seed=222)
    rng2 = tl2.rng("test", "stream")
    values2 = [rng2.random() for _ in range(10)]

    # ASSERT: Sequences are different
    assert values1 != values2


def test_named_rng_streams_are_independent():
    """
    PROPERTY: Different named RNG streams produce independent
    random sequences.

    VALIDATION: Two streams from same timeline with same seed
    produce different values.
    """

    tl = Timeline(master_seed=42)

    rng_alice = tl.rng("alice", "basis")
    rng_bob = tl.rng("bob", "detector")

    values_alice = [rng_alice.random() for _ in range(10)]
    values_bob = [rng_bob.random() for _ in range(10)]

    # ASSERT: Streams are independent
    assert values_alice != values_bob


def test_rng_stream_reuse_is_deterministic():
    """
    PROPERTY: Requesting the same RNG stream multiple times
    returns the same stream instance.

    VALIDATION: Multiple calls to rng() with same path return
    same values.
    """

    tl = Timeline(master_seed=42)

    # Request same stream twice
    rng1 = tl.rng("alice", "basis")
    val1 = rng1.random()

    rng2 = tl.rng("alice", "basis")
    val2 = rng2.random()

    # ASSERT: Second request continues from same sequence
    # (not restarting from beginning)
    assert val1 != val2

    # ASSERT: This behavior is deterministic
    tl_check = Timeline(master_seed=42)
    rng_check = tl_check.rng("alice", "basis")
    val_check1 = rng_check.random()
    val_check2 = rng_check.random()

    assert val1 == val_check1
    assert val2 == val_check2


# ─────────────────────────────────────────────
# TEST GROUP 4: Replay Safety (Incremental Equivalence)
# ─────────────────────────────────────────────


def test_incremental_execution_equals_oneshot(handler):
    """
    PROPERTY: run_until(a); run_until(b) ≡ run_until(b) for a <= b

    VALIDATION: Incremental execution produces same trace as
    one-shot execution.
    """

    # Setup: Create identical event schedules
    def setup_timeline():
        tl = Timeline(master_seed=42)
        h = MockHandler("handler")
        tl.schedule(
            Event(time=5, priority=0, target_ref=h, action="T5", payload_ref=None)
        )
        tl.schedule(
            Event(time=10, priority=0, target_ref=h, action="T10", payload_ref=None)
        )
        tl.schedule(
            Event(time=15, priority=0, target_ref=h, action="T15", payload_ref=None)
        )
        tl.schedule(
            Event(time=20, priority=0, target_ref=h, action="T20", payload_ref=None)
        )
        return tl, h

    # Timeline 1: Incremental execution
    tl1, h1 = setup_timeline()
    tl1.run_until(5)
    tl1.run_until(10)
    tl1.run_until(15)
    tl1.run_until(20)
    trace1 = h1.get_trace()

    # Timeline 2: One-shot execution
    tl2, h2 = setup_timeline()
    tl2.run_until(20)
    trace2 = h2.get_trace()

    # ASSERT: Traces are identical
    assert trace1 == trace2
    assert len(trace1) == 4


@given(
    boundaries=st.lists(
        st.integers(min_value=1, max_value=50),
        min_size=1,
        max_size=10,
        unique=True,
    )
)
@settings(max_examples=80)
def test_random_incremental_boundaries_equivalent(boundaries):
    """
    PROPERTY:
    Arbitrary incremental run_until boundaries produce the same
    result as one-shot execution.
    """

    # Sort boundaries to ensure monotonic run_until
    boundaries = sorted(boundaries)
    final_time = max(boundaries)

    def setup_timeline():
        tl = Timeline(master_seed=42)
        h = MockHandler("handler")

        # Fixed deterministic schedule
        for t in range(5, 55, 5):  # events at 5,10,...,50
            tl.schedule(
                Event(
                    time=t,
                    priority=0,
                    target_ref=h,
                    action=f"T{t}",
                    payload_ref=None,
                )
            )

        return tl, h

    # Incremental execution
    tl_inc, h_inc = setup_timeline()
    for b in boundaries:
        tl_inc.run_until(b)
    trace_inc = h_inc.get_trace()

    # One-shot execution
    tl_one, h_one = setup_timeline()
    tl_one.run_until(final_time)
    trace_one = h_one.get_trace()

    # ASSERT: traces are identical
    assert trace_inc == trace_one


# ─────────────────────────────────────────────
# TEST GROUP 5: Batch Execution Atomicity
# ─────────────────────────────────────────────


def test_same_time_events_execute_as_atomic_batch(handler):
    """
    PROPERTY: All events with the same time execute in a single
    batch (single run_one_step call).

    VALIDATION: run_one_step() returns summary with all same-time
    events.
    """

    tl = Timeline(master_seed=42)

    # Schedule multiple events at time=10
    for i in range(5):
        tl.schedule(
            Event(
                time=10,
                priority=i,
                target_ref=handler,
                action=f"E{i}",
                payload_ref=None,
            )
        )

    # Execute one batch
    summary = tl.run_one_step()

    # ASSERT: All 5 events executed in one batch
    assert summary.batch_time == 10
    assert summary.num_executed == 5
    assert len(summary.event_ids) == 5


def test_different_time_events_execute_in_separate_batches(handler):
    """
    PROPERTY: Events with different times execute in separate
    batches.

    VALIDATION: Multiple run_one_step() calls needed for different
    times.
    """

    tl = Timeline(master_seed=42)

    tl.schedule(
        Event(time=10, priority=0, target_ref=handler, action="T10", payload_ref=None)
    )
    tl.schedule(
        Event(time=20, priority=0, target_ref=handler, action="T20", payload_ref=None)
    )

    # Execute first batch
    summary1 = tl.run_one_step()
    assert summary1.batch_time == 10
    assert summary1.num_executed == 1

    # Execute second batch
    summary2 = tl.run_one_step()
    assert summary2.batch_time == 20
    assert summary2.num_executed == 1


# ─────────────────────────────────────────────
# TEST GROUP 6: Cross-Cutting Integration Tests
# ─────────────────────────────────────────────


def test_full_determinism_scenario(handler):
    """
    INTEGRATION: Complete determinism test combining ordering,
    cancellation, RNG, and replay.

    VALIDATION: Complex scenario produces identical results
    across multiple runs with same seed.
    """

    def run_complex_scenario(seed):
        tl = Timeline(master_seed=seed)
        h = MockHandler("handler")

        # Schedule events in random order
        tl.schedule(
            Event(time=30, priority=0, target_ref=h, action="LATE", payload_ref=None)
        )
        tl.schedule(
            Event(time=10, priority=1, target_ref=h, action="MID", payload_ref=None)
        )
        tl.schedule(
            Event(time=10, priority=0, target_ref=h, action="EARLY", payload_ref=None)
        )
        e_cancel = tl.schedule(
            Event(time=20, priority=0, target_ref=h, action="CANCEL", payload_ref=None)
        )
        tl.schedule(
            Event(time=30, priority=1, target_ref=h, action="LATE2", payload_ref=None)
        )

        # Cancel one event
        tl.cancel(e_cancel)

        # Get some RNG values
        rng = tl.rng("test", "random")
        random_vals = [rng.random() for _ in range(3)]

        # Execute incrementally
        tl.run_until(15)
        tl.run_until(30)

        return h.get_trace(), random_vals

    # Run scenario 3 times with same seed
    SEED = 777
    trace1, rng1 = run_complex_scenario(SEED)
    trace2, rng2 = run_complex_scenario(SEED)
    trace3, rng3 = run_complex_scenario(SEED)

    # ASSERT: All runs produce identical results
    assert trace1 == trace2 == trace3
    assert rng1 == rng2 == rng3

    # ASSERT: Expected execution order
    actions = [t[2] for t in trace1]
    assert actions == ["EARLY", "MID", "LATE", "LATE2"]


def test_statistics_are_deterministic():
    """
    PROPERTY: Timeline statistics are deterministic and match
    execution.

    VALIDATION: Same schedule produces same statistics across runs.
    """

    def setup_and_run():
        tl = Timeline(master_seed=42)
        h = MockHandler("handler")

        # Schedule 10 events, cancel 3
        events = []
        for i in range(10):
            e = tl.schedule(
                Event(
                    time=i * 5,
                    priority=0,
                    target_ref=h,
                    action=f"E{i}",
                    payload_ref=None,
                )
            )
            events.append(e)

        tl.cancel(events[2])
        tl.cancel(events[5])
        tl.cancel(events[8])

        tl.run_until(100)

        return tl.stats

    # Run twice
    stats1 = setup_and_run()
    stats2 = setup_and_run()

    # ASSERT: Statistics are identical
    assert stats1.total_events_scheduled == stats2.total_events_scheduled == 10
    assert stats1.total_events_executed == stats2.total_events_executed == 7
    assert stats1.current_time == stats2.current_time


# ─────────────────────────────────────────────
# TEST GROUP 7: Edge Cases
# ─────────────────────────────────────────────


def test_empty_timeline_is_deterministic():
    """
    PROPERTY: Empty timeline behaves deterministically.

    VALIDATION: run_until on empty timeline is safe and deterministic.
    """

    tl1 = Timeline(master_seed=42)
    tl2 = Timeline(master_seed=42)

    # Run on empty timelines
    tl1.run_until(100)
    tl2.run_until(100)

    # ASSERT: Both remain at time 0
    assert tl1.current_time == 0
    assert tl2.current_time == 0

    # ASSERT: Statistics are identical
    assert tl1.stats.total_events_executed == 0
    assert tl2.stats.total_events_executed == 0


@given(
    time=st.integers(min_value=0, max_value=100),
    priority=st.integers(min_value=-5, max_value=5),
)
@settings(max_examples=100)
def test_single_event_execution_is_deterministic(time, priority):
    """
    PROPERTY:
    Single-event execution is deterministic for a fixed seed.
    """

    def run_once():
        tl = Timeline(master_seed=42)
        h = MockHandler("handler")
        tl.schedule(
            Event(
                time=time,
                priority=priority,
                target_ref=h,
                action="ONLY",
                payload_ref=None,
            )
        )
        tl.run_until(time)
        return h.get_trace()

    trace1 = run_once()
    trace2 = run_once()
    trace3 = run_once()

    assert trace1 == trace2 == trace3


@given(
    priorities=st.lists(
        st.integers(min_value=0, max_value=10),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=50)
def test_large_batch_determinism(priorities):
    """
    PROPERTY:
    Large batches of same-time events execute deterministically
    regardless of priority distribution.
    """

    def run_once():
        tl = Timeline(master_seed=42)
        h = MockHandler("handler")

        for i, priority in enumerate(priorities):
            tl.schedule(
                Event(
                    time=10,
                    priority=priority,
                    target_ref=h,
                    action=f"E{i}",
                    payload_ref=None,
                )
            )

        tl.run_until(10)
        return h.get_trace()

    trace1 = run_once()
    trace2 = run_once()

    assert trace1 == trace2
    assert len(trace1) == len(priorities)
