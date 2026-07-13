from io import StringIO

from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.sinks import MemorySink, TextSink


def test_logger_level_filtering():
    sink = MemorySink()
    logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink], session_id="session-x")

    logger.error("test", "error message")
    logger.warning("test", "warning message")
    logger.info("test", "info message")
    logger.debug("test", "debug message")
    logger.trace("test", "trace message")

    assert [record.level for record in sink.records] == [
        LogLevel.ERROR,
        LogLevel.WARNING,
        LogLevel.INFO,
    ]
    assert [record.sequence for record in sink.records] == [0, 1, 2]
    assert [record.session_id for record in sink.records] == [
        "session-x",
        "session-x",
        "session-x",
    ]


def test_filtered_records_do_not_freeze_metadata():
    sink = MemorySink()
    logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])

    record = logger.log(
        level=LogLevel.DEBUG,
        category="test",
        message="filtered",
        meta=[["not-a-tuple", 1]],
    )

    assert record is None
    assert sink.records == []
    assert logger.sequence == 0


def test_null_logger_is_noop():
    logger = NullLogger()
    record = logger.log(level=LogLevel.ERROR, category="test", message="should drop")

    assert record is None
    assert logger.sequence == 0


def test_text_sink_formats_structured_line():
    stream = StringIO()
    sink = TextSink(stream=stream)
    logger = SimulationLogger(
        level=LogLevel.TRACE, sinks=[sink], session_id="session-a"
    )

    logger.log(
        level=LogLevel.INFO,
        category="engine.timeline",
        message="hello",
        sim_time=10,
        event_id=3,
        action="PING",
        target_name="Target",
        source_name="Source",
        meta={"k": 1},
    )

    line = stream.getvalue().strip()

    assert "[INFO]" in line
    assert "[engine.timeline]" in line
    assert line.startswith("[t=10 ps | tick=10] [000000] [INFO]")
    assert "event_id=3" in line
    assert "action=PING" in line
    assert "target=Target" in line
    assert "source=Source" in line
    assert "session=session-a" in line
    assert "meta={k=1}" in line


def test_text_sink_scales_large_picosecond_ticks():
    stream = StringIO()
    sink = TextSink(stream=stream)
    logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])

    logger.info("engine.timeline", "large tick", sim_time=12_500)

    line = stream.getvalue().strip()

    assert line.startswith("[t=12.500 ns | tick=12500] [000000] [INFO]")
