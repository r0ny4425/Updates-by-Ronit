import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline

# ─────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────


class MockTarget:
    def __init__(self, action=None):
        self.handled_events = []
        self.action = action

    def handle_event(self, event, timeline):
        self.handled_events.append(event.event_id)
        if callable(self.action):
            self.action(event, timeline)


def make_event(*, time: int, priority: int = 0, target_ref=None) -> Event:
    """
    Create an unscheduled Event (event_id=None).
    """
    return Event(
        time=time,
        priority=priority,
        event_id=None,
        target_ref=target_ref,
        action="TEST",
        payload_ref=None,
    )


def advance_time(timeline: Timeline, t: int) -> None:
    """
    Advance timeline time legally via execution.
    """
    timeline.schedule(make_event(time=t, target_ref=MockTarget()))
    timeline.run_one_step()


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────


def test_event_in_past():
    timeline = Timeline()
    advance_time(timeline, 3)

    event = make_event(time=2, target_ref=MockTarget())
    with pytest.raises(ValueError):
        timeline.schedule(event)


def test_event_currently():
    timeline = Timeline()
    advance_time(timeline, 3)

    event = make_event(time=3, target_ref=MockTarget())
    scheduled = timeline.schedule(event)

    assert scheduled is not None


def test_event_in_future():
    timeline = Timeline()
    advance_time(timeline, 3)

    event = make_event(time=5, target_ref=MockTarget())
    scheduled = timeline.schedule(event)

    assert scheduled is not None


def test_current_time_is_read_only():
    timeline = Timeline()
    with pytest.raises(AttributeError):
        timeline.current_time = 1
