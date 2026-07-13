import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline


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


def test_run_until_noop_when_t_end_in_past():
    """
    Ensures that nothing is done when the end time is less than the current time
    """
    timeline = Timeline()

    # Advance time legally via execution
    timeline.schedule(make_event(time=5, target_ref=MockTarget()))
    timeline.run_one_step()

    assert timeline.current_time == 5

    timeline.run_until(3)  # should do nothing
    assert timeline.current_time == 5


def test_run_until_executes_batch_at_boundary():
    """Tests the boundary time condition.
    If the batch time and end time are equal, the batch should run
    """
    timeline = Timeline()

    event = make_event(time=5, target_ref=MockTarget())
    timeline.schedule(event)

    timeline.run_until(5)

    assert timeline.current_time == 5


def test_run_until_incremental_equals_one_shot():
    """Using run until incrementally
    should give the same result as using it in one shot
    """
    t1 = Timeline()
    t2 = Timeline()

    for t in [t1, t2]:
        t.schedule(make_event(time=1, target_ref=MockTarget()))
        t.schedule(make_event(time=3, target_ref=MockTarget()))
        t.schedule(make_event(time=5, target_ref=MockTarget()))

    t1.run_until(2)
    t1.run_until(5)

    t2.run_until(5)

    assert t1.current_time == t2.current_time


def test_exit_safely_when_queue_empty():
    """Test to see whether run until exits safely
    when the batches are over before the end time
    """
    t1 = Timeline()
    t1.schedule(make_event(time=5, target_ref=MockTarget()))

    t1.run_until(7)
    assert t1.current_time == 5


def test_current_time_is_read_only():
    timeline = Timeline()
    with pytest.raises(AttributeError):
        timeline.current_time = 1
