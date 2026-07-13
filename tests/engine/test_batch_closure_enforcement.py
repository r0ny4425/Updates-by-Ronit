import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline


# Simple mock target that records handled events
class MockTarget:
    def __init__(self, action=None):
        self.handled_events = []
        self.action = action

    def handle_event(self, event, timeline):
        self.handled_events.append(event.event_id)
        if callable(self.action):
            self.action(event, timeline)


# Reuse your make_event helper
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


def test_events_scheduled_during_execution():
    timeline = Timeline()

    def schedule_new(event, tl: Timeline):
        new_event = make_event(time=tl.current_time + 1, target_ref=MockTarget())
        tl.schedule(new_event)

    target = MockTarget(action=schedule_new)
    event = make_event(time=1, target_ref=target)

    timeline.schedule(event)

    summary1 = timeline.run_one_step()
    summary2 = timeline.run_one_step()

    # Event scheduled during execution appears in next batch
    assert summary2.batch_time == summary1.batch_time + 1
    assert summary1.num_executed == 1
    assert summary2.num_executed == 1


def test_events_scheduled_during_execution_at_same_time():
    timeline = Timeline()

    def schedule_new(event, tl: Timeline):
        new_event = make_event(time=tl.current_time, target_ref=MockTarget())
        tl.schedule(new_event)

    target = MockTarget(action=schedule_new)
    event = make_event(time=1, target_ref=target)

    timeline.schedule(event)

    summary1 = timeline.run_one_step()
    summary2 = timeline.run_one_step()

    # Event scheduled during execution appears in next batch,
    # even if scheduled at the same timestamp
    assert summary2.batch_time == summary1.batch_time
    assert summary1.num_executed == 1
    assert summary2.num_executed == 1


def test_events_scheduled_during_execution_in_past():
    timeline = Timeline()

    def schedule_new(event, tl: Timeline):

        new_event = make_event(time=tl.current_time - 100, target_ref=MockTarget())

        tl.schedule(new_event)

    target = MockTarget(action=schedule_new)
    event = make_event(time=1, target_ref=target)

    timeline.schedule(event)

    # Ensures that any error with the handler is not suppressed
    # (here, it is that the handler schedules an event in the past)
    with pytest.raises(ValueError):
        timeline.run_one_step()


def test_handler_exception_aborts_batch_execution():
    timeline = Timeline()

    executed = []

    def failing_handler(event, tl):
        executed.append("first")
        raise RuntimeError()

    def handler_2(event, tl):
        executed.append("second")

    event1 = make_event(time=1, target_ref=MockTarget(action=failing_handler))
    event2 = make_event(time=1, target_ref=MockTarget(action=handler_2))

    timeline.schedule(event1)
    timeline.schedule(event2)

    with pytest.raises(RuntimeError):
        timeline.run_one_step()

    # Only the first event ran
    assert executed == ["first"]

    # Timeline must not be stuck in executing state, and must be usable
    event3 = make_event(time=2, target_ref=MockTarget())
    timeline.schedule(event3)
    summary = timeline.run_one_step()
    assert summary.num_executed == 1
