from hypothesis import given, seed
from hypothesis import strategies as st

from simyuj.engine.event import Event
from simyuj.engine.event_ordering import event_ordering_key
from simyuj.engine.event_queue import EventQueue

# ─────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────

event_ids = st.integers(min_value=0, max_value=10_000)
times = st.integers(min_value=0, max_value=100)
priorities = st.integers(min_value=-10, max_value=10)


@st.composite
def events(draw):
    return Event(
        time=draw(times),
        priority=draw(priorities),
        event_id=draw(event_ids),
        target_ref=None,
        action="TEST",
        payload_ref=None,
    )


@st.composite
def event_sets(draw):
    return draw(
        st.lists(
            events(),
            min_size=1,
            max_size=30,
            unique_by=lambda e: e.event_id,
        )
    )


@st.composite
def event_sets_with_cancellations(draw):
    evts = draw(event_sets())
    cancellations = draw(
        st.lists(st.booleans(), min_size=len(evts), max_size=len(evts))
    )
    return evts, cancellations


# ─────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────


@seed(1234)
@given(event_sets())
def test_pop_order_matches_event_ordering_key(events):
    """
    For any insertion order, popped events are ordered by
    (time, priority, event_id).
    """
    q = EventQueue()

    # Arbitrary insertion order
    for e in events:
        q.push(e)

    popped = [q.pop() for _ in range(len(events))]
    expected = sorted(events, key=event_ordering_key)

    assert popped == expected


@seed(4321)
@given(event_sets_with_cancellations())
def test_cancelled_events_are_never_returned(data):
    """
    Cancelled events must never be returned by the queue.
    """
    events, cancellations = data
    q = EventQueue()

    for e in events:
        q.push(e)

    cancelled = set()
    for e, cancel in zip(events, cancellations):
        if cancel:
            e._mark_cancelled()
            cancelled.add(e)

    popped = []
    while True:
        try:
            popped.append(q.pop())
        except IndexError:
            break

    for e in popped:
        assert e not in cancelled


@seed(9999)
@given(event_sets_with_cancellations())
def test_relative_order_of_non_cancelled_events_is_stable(data):
    """
    Removing cancelled events does not change the relative
    ordering of remaining events.
    """
    events, cancellations = data
    q = EventQueue()

    for e in events:
        q.push(e)

    remaining = []
    for e, cancel in zip(events, cancellations):
        if cancel:
            e._mark_cancelled()
        else:
            remaining.append(e)

    popped = []
    while True:
        try:
            popped.append(q.pop())
        except IndexError:
            break

    expected = sorted(remaining, key=event_ordering_key)
    assert popped == expected


@seed(2024)
@given(event_sets())
def test_queue_is_deterministic_for_same_input(events):
    """
    Given the same events and insertion order, pop order is deterministic.
    """
    q1 = EventQueue()
    q2 = EventQueue()

    for e in events:
        q1.push(e)

    # recreate events for the second queue to avoid shared mutation
    clone = [
        Event(
            time=e.time,
            priority=e.priority,
            event_id=e.event_id,
            target_ref=None,
            action="TEST",
            payload_ref=None,
        )
        for e in events
    ]

    for e in clone:
        q2.push(e)

    out1 = [q1.pop() for _ in range(len(events))]
    out2 = [q2.pop() for _ in range(len(events))]

    assert [e.event_id for e in out1] == [e.event_id for e in out2]
