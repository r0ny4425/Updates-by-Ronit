from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.mock_components.channel_stub import ChannelStub
from tests.support.mock_components.detector_stub import DetectorStub
from tests.support.mock_components.emitter_stub import EmitterStub

SOURCE_EMIT_START = "source.emit.start"


def _run_scenario(*, logger: SimulationLogger | None):
    timeline = Timeline(master_seed=42, logger=logger)
    detector = DetectorStub()
    channel = ChannelStub(output_target=detector)
    emitter = EmitterStub(output_target=channel)

    for t in (0, 100, 250):
        timeline.schedule(
            Event(
                time=t,
                target_ref=emitter,
                action=SOURCE_EMIT_START,
                payload_ref=None,
            )
        )

    executed_event_ids: list[int] = []
    while True:
        try:
            summary = timeline.run_one_step()
        except RuntimeError as exc:
            if "event queue is empty" in str(exc):
                break
            raise
        executed_event_ids.extend(summary.event_ids)

    return executed_event_ids, tuple(detector.detections), timeline.stats


def test_logging_does_not_change_execution_order_or_observables():
    baseline_ids, baseline_detections, baseline_stats = _run_scenario(logger=None)

    sink = MemorySink()
    trace_logger = SimulationLogger(level=LogLevel.TRACE, sinks=[sink])
    logged_ids, logged_detections, logged_stats = _run_scenario(logger=trace_logger)

    assert logged_ids == baseline_ids
    assert logged_detections == baseline_detections
    assert logged_stats == baseline_stats
    assert len(sink.records) > 0
