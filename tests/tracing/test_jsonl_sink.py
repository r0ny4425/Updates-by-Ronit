from __future__ import annotations

import json
from pathlib import Path

from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import JsonlSink


def test_jsonl_sink_persists_structured_records(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "sim.jsonl"

    with JsonlSink(path=log_path, auto_flush=True) as sink:
        logger = SimulationLogger(
            level=LogLevel.TRACE,
            sinks=[sink],
            session_id="session-jsonl",
        )
        logger.log(
            level=LogLevel.INFO,
            category="engine.timeline.schedule",
            message="event scheduled",
            sim_time=10,
            event_id=7,
            action="TEST_EVENT",
            target_name="Target",
            source_name="Source",
            node_id="node-a",
            link_id="link-ab",
            meta=(("queue_depth", 3), ("target_ids", (1, 2, 3))),
        )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["sequence"] == 0
    assert record["level"] == "INFO"
    assert record["category"] == "engine.timeline.schedule"
    assert record["session_id"] == "session-jsonl"
    assert record["node_id"] == "node-a"
    assert record["link_id"] == "link-ab"
    assert record["sim_time"] == 10
    assert record["event_id"] == 7
    assert record["meta"] == [["queue_depth", 3], ["target_ids", [1, 2, 3]]]
