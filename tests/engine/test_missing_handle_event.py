import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline


class MockTarget:
    # doesn't implement handle_event, generic target(not necessarily Component)
    def __init__(self, action=None):
        self.handled_events = []
        self.action = action


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


def test_missing_handle_event_raises_and_propogates():
    timeline = Timeline()
    target1 = MockTarget()
    timeline.schedule(make_event(time=1, target_ref=target1))
    with pytest.raises(TypeError):
        timeline.run_one_step()
