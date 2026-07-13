import pytest

from simyuj.engine.event import Event
from simyuj.engine.event_ordering import event_ordering_key
from simyuj.engine.timeline import Timeline


# ─────────────────────────────────────────────
# Helper: create an unscheduled Event
# ─────────────────────────────────────────────
def make_event(*, time: int, priority: int) -> Event:
    """
    Create an Event with event_id=None, suitable for scheduling
    via Timeline.schedule().
    """
    return Event(
        time=time,
        priority=priority,
        event_id=None,  # Timeline will assign
        cancelled=False,
        target_ref=None,
        payload_ref=None,
        action="TEST",
    )


# ─────────────────────────────────────────────
# Tests for Timeline.pop_batch
# ─────────────────────────────────────────────


def test_pop_batch_groups_same_timestamp():
    """Multiple events at the same timestamp are returned as one batch."""
    timeline = Timeline()

    e1 = timeline.schedule(make_event(time=10, priority=1))
    e2 = timeline.schedule(make_event(time=10, priority=0))
    e3 = timeline.schedule(make_event(time=20, priority=0))

    batch = timeline.pop_batch()

    expected = sorted([e1, e2], key=event_ordering_key)
    assert batch == expected
    assert all(e.time == 10 for e in batch)

    next_batch = timeline.pop_batch()
    assert next_batch == [e3]
    assert [e.time for e in next_batch] == [20]


def test_pop_batch_skips_cancelled_events():
    """Cancelled events are excluded from the batch."""
    timeline = Timeline()

    e1 = timeline.schedule(make_event(time=5, priority=0))
    e2 = timeline.schedule(make_event(time=5, priority=1))
    e3 = timeline.schedule(make_event(time=5, priority=2))

    timeline.cancel(e2)

    batch = timeline.pop_batch()
    assert e2 not in batch
    assert all(not e.cancelled for e in batch)
    expected = sorted([e1, e3], key=event_ordering_key)
    assert batch == expected


def test_pop_batch_independent_of_insertion_order():
    """Batch ordering is deterministic, regardless of insertion order."""
    # Timeline 1
    tl1 = Timeline()
    for t, p in [(10, 1), (10, 0), (10, 2)]:
        tl1.schedule(make_event(time=t, priority=p))
    batch1 = tl1.pop_batch()

    # Timeline 2, reversed insertion
    tl2 = Timeline()
    for t, p in [(10, 2), (10, 1), (10, 0)]:
        tl2.schedule(make_event(time=t, priority=p))
    batch2 = tl2.pop_batch()

    # Determine expected ordering
    expected_order1 = sorted(batch1, key=event_ordering_key)
    expected_order2 = sorted(batch2, key=event_ordering_key)

    # Both batches must match their sorted order
    assert batch1 == expected_order1
    assert batch2 == expected_order2

    # Their priorities produce deterministic ordering
    assert [e.priority for e in batch1] == [0, 1, 2]
    assert [e.priority for e in batch2] == [0, 1, 2]


def test_pop_batch_empty_queue_raises():
    """Calling pop_batch on an empty queue raises IndexError."""
    timeline = Timeline()
    with pytest.raises(IndexError):
        timeline.pop_batch()


def test_sequential_pop_batch_increases_timestamp():
    """Repeated pop_batch calls return batches with strictly increasing timestamps."""
    timeline = Timeline()

    timeline.schedule(make_event(time=1, priority=0))
    timeline.schedule(make_event(time=2, priority=0))
    timeline.schedule(make_event(time=3, priority=0))

    batch1 = timeline.pop_batch()
    batch2 = timeline.pop_batch()
    batch3 = timeline.pop_batch()

    t1 = batch1[0].time
    t2 = batch2[0].time
    t3 = batch3[0].time

    assert t1 < t2 < t3
    # Ensure batch timestamp consistency
    assert all(e.time == t1 for e in batch1)
    assert all(e.time == t2 for e in batch2)
    assert all(e.time == t3 for e in batch3)


# ─────────────────────────────────────────────
# tests for edge cases
# ─────────────────────────────────────────────


def test_pop_batch_does_not_advance_current_time():
    """pop_batch must not modify Timeline.current_time."""
    timeline = Timeline()

    timeline.schedule(make_event(time=10, priority=0))
    timeline.schedule(make_event(time=20, priority=0))

    assert timeline.current_time == 0

    timeline.pop_batch()

    # current_time must remain unchanged
    assert timeline.current_time == 0


def test_scheduling_same_time_after_pop_batch_does_not_join_batch():
    """
    Events scheduled at the same timestamp after batch extraction
    must not be included in the already-extracted batch.
    """
    timeline = Timeline()

    e1 = timeline.schedule(make_event(time=5, priority=0))
    e2 = timeline.schedule(make_event(time=5, priority=1))

    batch = timeline.pop_batch()
    assert batch == sorted([e1, e2], key=event_ordering_key)

    # Schedule a new event at the SAME timestamp after batch extraction
    e3 = timeline.schedule(make_event(time=5, priority=2))

    next_batch = timeline.pop_batch()

    # e3 must appear in a new batch, not the previous one
    assert batch == [e1, e2]
    assert next_batch == [e3]


def test_pop_batch_empty_queue_raises_from_timeline():
    """pop_batch on an empty queue raises a Timeline-owned IndexError."""
    timeline = Timeline()

    with pytest.raises(IndexError, match="pop_batch"):
        timeline.pop_batch()
