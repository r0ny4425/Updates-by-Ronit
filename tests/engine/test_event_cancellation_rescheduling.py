import pytest

from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


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


class MockTarget:
    def __init__(self, action=None):
        self.handled_events = []
        self.action = action

    def handle_event(self, event, timeline):
        self.handled_events.append(event.event_id)
        if callable(self.action):
            self.action(event, timeline)


# ─────────────────────────────────────────────
# Cancellation semantics
# ─────────────────────────────────────────────


def test_cancel_marks_event_as_cancelled():
    timeline = Timeline()

    e = timeline.schedule(make_event(time=10))
    assert e.cancelled is False

    timeline.cancel(e)
    assert e.cancelled is True


def test_cancelled_event_is_not_returned_by_queue():
    timeline = Timeline()
    target1 = MockTarget()
    target2 = MockTarget()

    e1 = timeline.schedule(make_event(time=1, target_ref=target1))
    timeline.schedule(make_event(time=2, target_ref=target2))

    timeline.cancel(e1)
    summary = timeline.run_one_step()

    assert summary.num_executed == 1
    assert summary.batch_time == 2
    with pytest.raises(RuntimeError):
        timeline.run_one_step()


def test_cancelling_event_does_not_affect_other_events():
    timeline = Timeline()
    target1 = MockTarget()
    e1 = timeline.schedule(make_event(time=1, target_ref=target1))
    e2 = timeline.schedule(make_event(time=2, target_ref=target1))
    e3 = timeline.schedule(make_event(time=3, target_ref=target1))

    timeline.cancel(e2)

    summary1 = timeline.run_one_step()
    summary2 = timeline.run_one_step()

    assert summary1.event_ids == (e1.event_id,)
    assert summary2.event_ids == (e3.event_id,)


# ─────────────────────────────────────────────
# Rescheduling semantics
# ─────────────────────────────────────────────


def test_reschedule_cancels_old_event():
    timeline = Timeline()

    old = timeline.schedule(make_event(time=10))
    new = timeline.reschedule(old, new_time=20)

    assert old.cancelled is True
    assert new.cancelled is False


def test_reschedule_creates_new_event_instance():
    timeline = Timeline()

    old = timeline.schedule(make_event(time=10))
    new = timeline.reschedule(old, new_time=20)

    assert new is not old


def test_reschedule_assigns_new_event_id():
    timeline = Timeline()

    old = timeline.schedule(make_event(time=10))
    new = timeline.reschedule(old, new_time=20)

    assert old.event_id != new.event_id


def test_rescheduled_event_is_returned_in_correct_order():
    timeline = Timeline()
    target1 = MockTarget()
    timeline.schedule(make_event(time=5, target_ref=target1))
    e2 = timeline.schedule(make_event(time=10, target_ref=target1))

    e2_rescheduled = timeline.reschedule(e2, new_time=1)

    summary = timeline.run_one_step()

    assert summary.batch_time == e2_rescheduled.time
    assert summary.event_ids == (e2_rescheduled.event_id,)


# ─────────────────────────────────────────────
# Ordering stability guarantees
# ─────────────────────────────────────────────


def test_rescheduling_does_not_reorder_unrelated_events():
    timeline = Timeline()
    target1 = MockTarget()
    e1 = timeline.schedule(make_event(time=1, target_ref=target1))
    e2 = timeline.schedule(make_event(time=2, target_ref=target1))
    e3 = timeline.schedule(make_event(time=3, target_ref=target1))

    timeline.reschedule(e2, new_time=10)
    summary1 = timeline.run_one_step()
    summary2 = timeline.run_one_step()

    assert summary1.event_ids == (e1.event_id,)
    assert summary2.event_ids == (e3.event_id,)


def test_multiple_reschedules_produce_multiple_cancelled_events():
    timeline = Timeline()
    target1 = MockTarget()
    e = timeline.schedule(make_event(time=5, target_ref=target1))

    e2 = timeline.reschedule(e, new_time=6)
    e3 = timeline.reschedule(e2, new_time=7)

    assert e.cancelled is True
    assert e2.cancelled is True
    assert e3.cancelled is False

    summary = timeline.run_one_step()

    assert summary.event_ids == (e3.event_id,)
    with pytest.raises(RuntimeError):
        timeline.run_one_step()


# ─────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────


def test_cancel_then_reschedule_is_allowed():
    timeline = Timeline()
    target1 = MockTarget()
    e = timeline.schedule(make_event(time=10, target_ref=target1))
    timeline.cancel(e)

    new = timeline.reschedule(e, new_time=20)

    assert e.cancelled is True
    assert new.cancelled is False

    summary = timeline.run_one_step()
    assert summary.event_ids == (new.event_id,)
    assert summary.batch_time == new.time


def test_reschedule_preserves_priority_if_not_overridden():
    timeline = Timeline()

    e = timeline.schedule(make_event(time=10, priority=7))
    new = timeline.reschedule(e, new_time=20)

    assert new.priority == 7
