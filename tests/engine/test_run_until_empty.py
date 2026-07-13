import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline


class RecordingTarget:
    def __init__(self, action=None):
        self.handled_events = []
        self.action = action

    def handle_event(self, event, timeline):
        self.handled_events.append(event.event_id)
        if callable(self.action):
            self.action(event, timeline)


class RaisingTarget:
    def __init__(self, error: Exception):
        self.error = error

    def handle_event(self, event, timeline):
        raise self.error


def make_event(*, time: int, priority: int = 0, target_ref=None) -> Event:
    return Event(
        time=time,
        priority=priority,
        event_id=None,
        target_ref=target_ref,
        action="TEST",
        payload_ref=None,
    )


def test_run_until_empty_returns_empty_tuple_for_empty_timeline():
    timeline = Timeline()

    assert timeline.run_until_empty() == ()


def test_run_until_empty_executes_one_scheduled_event():
    timeline = Timeline()
    target = RecordingTarget()
    event = timeline.schedule(make_event(time=5, target_ref=target))

    summaries = timeline.run_until_empty()

    assert len(summaries) == 1
    assert summaries[0].batch_time == 5
    assert summaries[0].event_ids == (event.event_id,)
    assert target.handled_events == [event.event_id]


def test_run_until_empty_drains_events_scheduled_by_handler():
    timeline = Timeline()
    second_target = RecordingTarget()

    def schedule_later(event, tl: Timeline):
        tl.schedule(make_event(time=7, target_ref=second_target))

    first_target = RecordingTarget(action=schedule_later)
    first_event = timeline.schedule(make_event(time=3, target_ref=first_target))

    summaries = timeline.run_until_empty()

    assert tuple(summary.batch_time for summary in summaries) == (3, 7)
    assert summaries[0].event_ids == (first_event.event_id,)
    assert first_target.handled_events == [first_event.event_id]
    assert second_target.handled_events == [summaries[1].event_ids[0]]


def test_run_until_empty_treats_only_cancelled_events_as_empty():
    timeline = Timeline()
    target = RecordingTarget()
    event = timeline.schedule(make_event(time=5, target_ref=target))
    timeline.cancel(event)

    assert len(timeline._queue) == 1
    assert timeline.run_until_empty() == ()
    assert target.handled_events == []
    assert len(timeline._queue) == 0


def test_run_until_empty_propagates_handler_exceptions_unchanged():
    timeline = Timeline()
    error = ValueError("boom")
    timeline.schedule(make_event(time=1, target_ref=RaisingTarget(error)))

    with pytest.raises(ValueError) as exc_info:
        timeline.run_until_empty()

    assert exc_info.value is error


def test_run_one_step_empty_queue_error_is_unchanged():
    timeline = Timeline()

    with pytest.raises(
        RuntimeError,
        match="Cannot execute: Timeline event queue is empty",
    ):
        timeline.run_one_step()
