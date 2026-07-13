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


class MockTimeTarget:
    def __init__(self):
        self.handled_events = []

    def handle_event(self, event, timeline):
        self.handled_events.append(event.event_id)
        timeline._current_time += 1


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


# -------------------------
# TEST: One batch per run
# -------------------------
def test_one_batch_per_run():
    timeline = Timeline()
    target1 = MockTarget()
    target2 = MockTarget()

    event1 = make_event(time=1, target_ref=target1)
    event2 = make_event(time=2, target_ref=target2)

    timeline.schedule(event1)
    timeline.schedule(event2)

    summary = timeline.run_one_step()

    # Only the first batch executes
    assert summary.batch_time == 1
    assert event1.event_id in summary.event_ids
    assert event2.event_id not in summary.event_ids
    assert timeline.current_time == 1


# -------------------------
# TEST: current_time advances once per batch
# -------------------------
def test_current_time_advances_once():
    timeline = Timeline()
    target = MockTarget()
    event = make_event(time=5, target_ref=target)

    timeline.schedule(event)
    old_time = timeline.current_time

    summary = timeline.run_one_step()

    assert timeline.current_time == 5
    assert timeline.current_time >= old_time
    assert summary.batch_time == timeline.current_time


# -------------------------
# TEST: Re-entrancy forbidden
# -------------------------
def test_reentrancy_forbidden():
    timeline = Timeline()
    attempted = {"called": False}

    def nested_run(event, tl: Timeline):
        attempted["called"] = True
        with pytest.raises(RuntimeError, match="already executing"):
            tl.run_one_step()

    target = MockTarget(action=nested_run)
    timeline.schedule(make_event(time=1, target_ref=target))

    timeline.run_one_step()

    assert attempted["called"]


def test_mutate_current_time_by_handle_event_forbidden():
    timeline = Timeline()
    target = MockTimeTarget()
    event = make_event(time=5, target_ref=target)
    timeline.schedule(event)

    with pytest.raises(RuntimeError, match="mutated Timeline.current_time"):
        timeline.run_one_step()
