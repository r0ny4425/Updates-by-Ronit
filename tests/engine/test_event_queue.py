import pytest

from simyuj.engine.event import Event
from simyuj.engine.event_ordering import event_ordering_key
from simyuj.engine.event_queue import EventQueue

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def make_event(
    *,
    time: int,
    priority: int = 0,
    event_id: int | None,
) -> Event:
    return Event(
        time=time,
        priority=priority,
        event_id=event_id,
        target_ref=None,
        action="TEST",
        payload_ref=None,
    )


# ─────────────────────────────────────────────
# Basic queue behavior
# ─────────────────────────────────────────────


def test_push_and_pop_single_event():
    q = EventQueue()
    e = make_event(time=10, event_id=1)

    q.push(e)
    assert q.pop() is e
    assert len(q) == 0


def test_peek_does_not_remove():
    q = EventQueue()
    e = make_event(time=5, event_id=1)

    q.push(e)

    assert q.peek() is e
    assert len(q) == 1
    assert q.pop() is e


def test_pop_empty_queue_raises():
    q = EventQueue()

    with pytest.raises(IndexError):
        q.pop()


def test_peek_empty_queue_raises():
    q = EventQueue()

    with pytest.raises(IndexError):
        q.peek()


# ─────────────────────────────────────────────
# Ordering guarantees
# ─────────────────────────────────────────────


def test_ordering_by_time():
    q = EventQueue()

    e1 = make_event(time=10, event_id=1)
    e2 = make_event(time=5, event_id=2)
    e3 = make_event(time=20, event_id=3)

    q.push(e1)
    q.push(e2)
    q.push(e3)

    assert q.pop() is e2
    assert q.pop() is e1
    assert q.pop() is e3


def test_ordering_by_priority_when_time_equal():
    q = EventQueue()

    e1 = make_event(time=10, priority=5, event_id=1)
    e2 = make_event(time=10, priority=1, event_id=2)
    e3 = make_event(time=10, priority=3, event_id=3)

    q.push(e1)
    q.push(e2)
    q.push(e3)

    assert q.pop() is e2
    assert q.pop() is e3
    assert q.pop() is e1


def test_ordering_by_event_id_when_time_and_priority_equal():
    q = EventQueue()

    e1 = make_event(time=10, priority=1, event_id=3)
    e2 = make_event(time=10, priority=1, event_id=1)
    e3 = make_event(time=10, priority=1, event_id=2)

    q.push(e1)
    q.push(e2)
    q.push(e3)

    assert q.pop() is e2
    assert q.pop() is e3
    assert q.pop() is e1


def test_queue_ordering_matches_event_ordering_key():
    q = EventQueue()

    events = [
        make_event(time=5, priority=2, event_id=3),
        make_event(time=5, priority=1, event_id=2),
        make_event(time=3, priority=9, event_id=1),
        make_event(time=5, priority=1, event_id=1),
    ]

    for e in events:
        q.push(e)

    expected = sorted(events, key=event_ordering_key)
    popped = [q.pop() for _ in range(len(events))]

    assert popped == expected


# ─────────────────────────────────────────────
# Cancellation behavior (lazy)
# ─────────────────────────────────────────────


def test_cancelled_event_at_head_is_skipped():
    q = EventQueue()

    cancelled = make_event(time=1, event_id=1)
    valid = make_event(time=2, event_id=2)

    cancelled._mark_cancelled()

    q.push(cancelled)
    q.push(valid)

    assert q.pop() is valid


def test_cancelled_event_not_at_head_is_skipped_lazily():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)
    e3 = make_event(time=3, event_id=3)

    q.push(e1)
    q.push(e2)
    q.push(e3)

    e2._mark_cancelled()

    assert q.pop() is e1
    assert q.pop() is e3


def test_all_events_cancelled_behaves_as_empty():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)

    e1._mark_cancelled()
    e2._mark_cancelled()

    q.push(e1)
    q.push(e2)

    with pytest.raises(IndexError):
        q.peek()

    with pytest.raises(IndexError):
        q.pop()


def test_len_includes_cancelled_events():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)

    e1._mark_cancelled()

    q.push(e1)
    q.push(e2)

    assert len(q) == 2
    assert q.pop() is e2
    assert len(q) == 0


# ─────────────────────────────────────────────
# Validation / edge cases
# ─────────────────────────────────────────────


def test_push_without_event_id_raises():
    q = EventQueue()
    e = make_event(time=1, event_id=None)

    with pytest.raises(ValueError):
        q.push(e)


def test_peek_does_not_discard_valid_events():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)

    q.push(e1)
    q.push(e2)

    assert q.peek() is e1
    assert q.peek() is e1
    assert q.pop() is e1


def test_cancel_after_peek_before_pop():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)

    q.push(e1)
    q.push(e2)

    assert q.peek() is e1
    e1._mark_cancelled()

    assert q.pop() is e2


def test_interleaved_push_and_cancel():
    q = EventQueue()

    e1 = make_event(time=1, event_id=1)
    e2 = make_event(time=2, event_id=2)
    e3 = make_event(time=3, event_id=3)

    q.push(e1)
    q.push(e2)

    e1._mark_cancelled()
    q.push(e3)

    assert q.pop() is e2
    assert q.pop() is e3
