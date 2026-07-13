from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def make_event(time: int) -> Event:
    return Event(
        time=time,
        event_id=None,
        priority=0,
        target_ref=None,
        action="TEST",
        payload_ref=None,
    )


# ─────────────────────────────────────────────
# EventID allocation semantics
# ─────────────────────────────────────────────


def test_event_ids_are_monotonically_increasing():
    timeline = Timeline()

    e1 = timeline.schedule(make_event(1))
    e2 = timeline.schedule(make_event(2))
    e3 = timeline.schedule(make_event(3))

    assert e1.event_id is not None
    assert e2.event_id is not None
    assert e3.event_id is not None

    assert e1.event_id < e2.event_id < e3.event_id


def test_event_ids_are_unique():
    timeline = Timeline()

    events = [timeline.schedule(make_event(i)) for i in range(5)]
    ids = [e.event_id for e in events]

    assert len(ids) == len(set(ids))


def test_event_id_not_reused_after_cancellation():
    timeline = Timeline()

    e1 = timeline.schedule(make_event(1))
    timeline.cancel(e1)

    e2 = timeline.schedule(make_event(2))

    assert e1.event_id is not None
    assert e2.event_id is not None

    assert e2.event_id > e1.event_id


def test_event_id_not_reused_after_reschedule():
    timeline = Timeline()

    e1 = timeline.schedule(make_event(1))
    e2 = timeline.reschedule(e1, new_time=10)

    assert e1.event_id is not None
    assert e2.event_id is not None

    assert e1.event_id != e2.event_id
    assert e2.event_id > e1.event_id


def test_event_id_counter_resets_on_new_timeline():
    t1 = Timeline()
    e1 = t1.schedule(make_event(1))
    e2 = t1.schedule(make_event(2))

    t2 = Timeline()
    e3 = t2.schedule(make_event(1))

    assert e1.event_id == 0
    assert e2.event_id == 1
    assert e3.event_id == 0
