import pytest

from simyuj.control.payloads import AgentReport, AgentStart, TimerFired


def test_agent_start_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="agent_id must be non-empty"):
        AgentStart(agent_id="", session_id="session-1")


def test_agent_start_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="session_id must be non-empty"):
        AgentStart(agent_id="alice", session_id="")


def test_agent_start_rejects_empty_node_id_when_present() -> None:
    with pytest.raises(ValueError, match="node_id must be non-empty"):
        AgentStart(agent_id="alice", node_id="", session_id="session-1")


def test_agent_start_rejects_invalid_meta() -> None:
    with pytest.raises(TypeError, match="meta values must be hashable"):
        AgentStart(
            agent_id="alice",
            session_id="session-1",
            meta=(("bad", []),),
        )


def test_timer_fired_rejects_empty_timer_id() -> None:
    with pytest.raises(ValueError, match="timer_id must be non-empty"):
        TimerFired(timer_id="", owner_agent_id="alice", scheduled_at=0, fires_at=0)


def test_timer_fired_rejects_empty_owner_agent_id() -> None:
    with pytest.raises(ValueError, match="owner_agent_id must be non-empty"):
        TimerFired(timer_id="retry", owner_agent_id="", scheduled_at=0, fires_at=0)


def test_timer_fired_rejects_negative_scheduled_at() -> None:
    with pytest.raises(ValueError, match="scheduled_at must be non-negative"):
        TimerFired(
            timer_id="retry", owner_agent_id="alice", scheduled_at=-1, fires_at=0
        )


def test_timer_fired_rejects_negative_fires_at() -> None:
    with pytest.raises(ValueError, match="fires_at must be non-negative"):
        TimerFired(
            timer_id="retry", owner_agent_id="alice", scheduled_at=0, fires_at=-1
        )


def test_timer_fired_rejects_fires_at_before_scheduled_at() -> None:
    with pytest.raises(ValueError, match="fires_at must be >= scheduled_at"):
        TimerFired(timer_id="retry", owner_agent_id="alice", scheduled_at=2, fires_at=1)


def test_timer_fired_rejects_invalid_correlation_id() -> None:
    bad_correlation_id = object()
    with pytest.raises(TypeError, match="correlation_id must be str, int, or None"):
        TimerFired(
            timer_id="retry",
            owner_agent_id="alice",
            scheduled_at=0,
            fires_at=1,
            correlation_id=bad_correlation_id,  # type: ignore[arg-type]
        )


def test_timer_fired_rejects_invalid_meta() -> None:
    with pytest.raises(TypeError, match="meta values must be hashable"):
        TimerFired(
            timer_id="retry",
            owner_agent_id="alice",
            scheduled_at=0,
            fires_at=1,
            meta=(("bad", []),),
        )


def test_agent_report_rejects_empty_source_id_when_present() -> None:
    with pytest.raises(ValueError, match="source_id must be non-empty"):
        AgentReport(report="click", source_id="")


def test_agent_report_rejects_invalid_correlation_id() -> None:
    bad_correlation_id = object()
    with pytest.raises(TypeError, match="correlation_id must be str, int, or None"):
        AgentReport(
            report="click",
            correlation_id=bad_correlation_id,  # type: ignore[arg-type]
        )


def test_agent_report_rejects_invalid_meta() -> None:
    with pytest.raises(TypeError, match="meta values must be hashable"):
        AgentReport(report="click", meta=(("bad", []),))
