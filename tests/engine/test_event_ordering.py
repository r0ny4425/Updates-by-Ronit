import pytest

from simyuj.engine.event import Event
from simyuj.engine.event_ordering import event_ordering_key


def make_event(
    *,
    time: int = 0,
    priority: int = 0,
    event_id: int | None = 0,
) -> Event:
    return Event(
        time=time,
        priority=priority,
        event_id=event_id,
        target_ref=None,
        action="TEST",
        payload_ref=None,
    )


def test_ordering_by_time():
    e1 = make_event(time=5, event_id=1)
    e2 = make_event(time=10, event_id=2)

    assert event_ordering_key(e1) < event_ordering_key(e2)


def test_ordering_by_priority_when_time_equal():
    e1 = make_event(time=5, priority=0, event_id=1)
    e2 = make_event(time=5, priority=10, event_id=2)

    assert event_ordering_key(e1) < event_ordering_key(e2)


def test_ordering_by_event_id_when_time_and_priority_equal():
    e1 = make_event(time=5, priority=0, event_id=1)
    e2 = make_event(time=5, priority=0, event_id=2)

    assert event_ordering_key(e1) < event_ordering_key(e2)


def test_event_id_is_required_for_ordering():
    e = make_event(event_id=None)

    with pytest.raises(ValueError):
        event_ordering_key(e)
