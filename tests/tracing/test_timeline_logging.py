from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink


class NoOpReceiver(Component):
    def __init__(self) -> None:
        self.handled_event_ids: list[int] = []

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        if event.event_id is not None:
            self.handled_event_ids.append(event.event_id)


def test_timeline_logs_include_sim_time_and_event_id():
    sink = MemorySink()
    logger = SimulationLogger(level=LogLevel.TRACE, sinks=[sink])
    timeline = Timeline(logger=logger)
    receiver = NoOpReceiver()

    scheduled = timeline.schedule(
        Event(time=5, target_ref=receiver, action="PING", payload_ref=None)
    )

    summary = timeline.run_one_step()
    assert summary.event_ids == (scheduled.event_id,)

    schedule_record = next(
        record
        for record in sink.records
        if record.category == "engine.timeline.schedule"
        and record.event_id == scheduled.event_id
    )
    execution_record = next(
        record
        for record in sink.records
        if record.category == "engine.timeline.event_execution"
        and record.event_id == scheduled.event_id
    )

    assert schedule_record.sim_time == 0
    assert schedule_record.action == "PING"
    assert schedule_record.target_name == "NoOpReceiver"

    assert execution_record.sim_time == 5
    assert execution_record.action == "PING"
    assert execution_record.target_name == "NoOpReceiver"


def test_run_until_logs_lifecycle_and_queue_exhaustion():
    sink = MemorySink()
    logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])
    timeline = Timeline(logger=logger)

    timeline.run_until(100)

    categories = [record.category for record in sink.records]
    assert categories == [
        "engine.timeline.run_until_start",
        "engine.timeline.queue_exhausted",
        "engine.timeline.run_until_finish",
    ]
    assert all(record.sim_time == 0 for record in sink.records)


def test_timeline_engine_internals_are_trace_level():
    debug_sink = MemorySink()
    debug_timeline = Timeline(
        logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[debug_sink])
    )
    receiver = NoOpReceiver()

    debug_timeline.schedule(
        Event(time=5, target_ref=receiver, action="PING", payload_ref=None)
    )
    debug_timeline.run_one_step()

    assert not any(
        record.category.startswith("engine.timeline.batch_")
        or record.category == "engine.timeline.schedule"
        for record in debug_sink.records
    )

    trace_sink = MemorySink()
    trace_timeline = Timeline(
        logger=SimulationLogger(level=LogLevel.TRACE, sinks=[trace_sink])
    )
    trace_receiver = NoOpReceiver()

    trace_timeline.schedule(
        Event(time=5, target_ref=trace_receiver, action="PING", payload_ref=None)
    )
    trace_timeline.run_one_step()

    categories = {record.category for record in trace_sink.records}
    assert "engine.timeline.schedule" in categories
    assert "engine.timeline.batch_start" in categories
    assert "engine.timeline.batch_complete" in categories
    assert "engine.timeline.event_execution" in categories
