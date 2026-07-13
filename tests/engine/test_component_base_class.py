from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline


class MockComponent(Component):
    def __init__(self):
        self.call_count = 0

    def handle_event(self, event, timeline):
        self.call_count += 1


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


def test_timeline_dispatches_component_blindly():
    comp = MockComponent()
    timeline = Timeline()
    timeline.schedule(make_event(time=1, target_ref=comp))
    timeline.run_one_step()
    assert comp.call_count == 1
