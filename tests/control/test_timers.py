from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.control import Agent, AgentContext, SessionRuntime
from simyuj.control.actions import AGENT_MESSAGE, AGENT_TIMER
from simyuj.control.payloads import TimerFired
from simyuj.control.timers import TimerService
from simyuj.engine import Event, Timeline
from simyuj.network import Network, Node
from simyuj.primitives.messages.transport import ClassicalMessage


@dataclass(slots=True)
class TimerAgent(Agent):
    timers_seen: list[TimerFired] = field(default_factory=list)

    def on_timer(self, timer: TimerFired, ctx: AgentContext) -> None:
        del ctx
        self.timers_seen.append(timer)


@dataclass(slots=True)
class CrossEventTimerAgent(Agent):
    mode: str = "replace"
    timers_seen: list[tuple[str, int]] = field(default_factory=list)

    def on_start(self, start, ctx: AgentContext) -> None:
        del start
        assert ctx.timers is not None
        ctx.timers.at("deadline", 10)

    def on_message(self, message, ctx: AgentContext) -> None:
        del message
        assert ctx.timers is not None
        if self.mode == "replace":
            ctx.timers.at("deadline", 20, replace=True)
        elif self.mode == "set_once":
            ctx.timers.at("deadline", 20, set_once=True)
        elif self.mode == "cancel":
            ctx.timers.cancel("deadline")
        else:
            raise AssertionError(self.mode)

    def on_timer(self, timer: TimerFired, ctx: AgentContext) -> None:
        self.timers_seen.append((timer.timer_id, ctx.timeline.current_time))


def run_cross_event_timer(mode: str) -> CrossEventTimerAgent:
    timeline = Timeline(master_seed=1)
    agent = CrossEventTimerAgent(agent_id="alice", mode=mode)
    network = Network()
    node = Node("controller")
    node.add_agent(agent)
    network.add_node(node)
    runtime = SessionRuntime(timeline=timeline, network=network)
    runtime.bind_all()
    runtime.schedule_agent_starts()
    timeline.schedule(
        Event(
            time=1,
            target_ref=agent,
            action=AGENT_MESSAGE,
            payload_ref=ClassicalMessage(
                sender_id="tester",
                receiver_id="alice",
                body="update",
            ),
        )
    )
    runtime.run_until_empty()
    return agent


def test_set_schedules_agent_timer_to_owner_agent() -> None:
    timeline = Timeline(master_seed=1)
    agent = TimerAgent(agent_id="alice")
    service = TimerService(
        owner_agent=agent,
        timeline=timeline,
        session_id="session-1",
    )

    event = service.set("retry", 3, correlation_id="round-1")

    assert event.target_ref is agent
    assert event.action == AGENT_TIMER
    assert isinstance(event.payload_ref, TimerFired)
    assert event.payload_ref.timer_id == "retry"
    assert event.payload_ref.owner_agent_id == "alice"
    assert event.payload_ref.scheduled_at == 0
    assert event.payload_ref.fires_at == 3
    assert event.payload_ref.correlation_id == "round-1"


def test_at_schedules_exact_fires_at() -> None:
    timeline = Timeline(master_seed=1)
    agent = TimerAgent(agent_id="alice")
    service = TimerService(
        owner_agent=agent,
        timeline=timeline,
        session_id="session-1",
        priority=5,
    )

    event = service.at("deadline", 10)

    assert event.time == 10
    assert event.priority == 5
    assert event.payload_ref.fires_at == 10


def test_timer_service_delivers_through_runtime_context() -> None:
    timeline = Timeline(master_seed=1)
    agent = TimerAgent(agent_id="alice")
    network = Network()
    node = Node("controller")
    node.add_agent(agent)
    network.add_node(node)
    runtime = SessionRuntime(timeline=timeline, network=network)
    runtime.bind_all()
    context = agent._context_for(
        runtime.schedule_agent_starts()[0],
        timeline,
    )

    assert context.timers is not None
    context.timers.set("retry", 1)
    runtime.run_until_empty()

    assert [timer.timer_id for timer in agent.timers_seen] == ["retry"]


def test_set_rejects_negative_delay() -> None:
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=Timeline(master_seed=1),
        session_id="session-1",
    )

    with pytest.raises(ValueError, match="delay must be non-negative"):
        service.set("retry", -1)


def test_at_rejects_negative_time() -> None:
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=Timeline(master_seed=1),
        session_id="session-1",
    )

    with pytest.raises(ValueError, match="time must be non-negative"):
        service.at("retry", -1)


def test_at_rejects_invalid_meta() -> None:
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=Timeline(master_seed=1),
        session_id="session-1",
    )

    with pytest.raises(TypeError, match="meta values must be hashable"):
        service.at("retry", 1, meta=(("bad", []),))


def test_cancel_marks_scheduled_event_cancelled() -> None:
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=Timeline(master_seed=1),
        session_id="session-1",
    )
    event = service.at("retry", 1)

    service.cancel(event)

    assert event.cancelled is True


def test_replace_cancels_existing_timer() -> None:
    timeline = Timeline(master_seed=1)
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=timeline,
        session_id="session-1",
    )
    event1 = service.at("retry", 5)
    event2 = service.at("retry", 10, replace=True)

    assert event1.cancelled is True
    assert event2.cancelled is False
    assert event1 is not event2


def test_set_once_returns_existing_timer() -> None:
    timeline = Timeline(master_seed=1)
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=timeline,
        session_id="session-1",
    )
    event1 = service.at("retry", 5)
    event2 = service.at("retry", 10, set_once=True)

    assert event1.cancelled is False
    assert event2 is event1


def test_set_once_and_replace_are_mutually_exclusive() -> None:
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=Timeline(master_seed=1),
        session_id="session-1",
    )
    with pytest.raises(ValueError, match="cannot specify both"):
        service.at("retry", 5, replace=True, set_once=True)


def test_cancel_by_timer_id_string() -> None:
    timeline = Timeline(master_seed=1)
    service = TimerService(
        owner_agent=TimerAgent(agent_id="alice"),
        timeline=timeline,
        session_id="session-1",
    )
    event = service.at("retry", 5)

    service.cancel("retry")
    assert event.cancelled is True

    # Canceling again should be a no-op
    service.cancel("retry")


def test_expired_timers_are_cleaned_up() -> None:
    timeline = Timeline(master_seed=1)
    agent = TimerAgent(agent_id="alice")
    network = Network()
    node = Node("controller")
    node.add_agent(agent)
    network.add_node(node)
    runtime = SessionRuntime(timeline=timeline, network=network)
    runtime.bind_all()
    context = agent._context_for(runtime.schedule_agent_starts()[0], timeline)
    assert context.timers is not None

    event1 = context.timers.at("retry", 5)
    runtime.run_until_empty()

    # At this point event1 has executed.
    assert timeline.current_time == 5

    # Replacing it shouldn't cancel it (it's already fired).
    event2 = context.timers.at("retry", 10, replace=True)
    assert event1.cancelled is False
    assert event2 is not event1


def test_replace_cancels_active_timer_across_runtime_events() -> None:
    agent = run_cross_event_timer("replace")

    assert agent.timers_seen == [("deadline", 20)]


def test_set_once_reuses_active_timer_across_runtime_events() -> None:
    agent = run_cross_event_timer("set_once")

    assert agent.timers_seen == [("deadline", 10)]


def test_cancel_by_timer_id_cancels_active_timer_across_runtime_events() -> None:
    agent = run_cross_event_timer("cancel")

    assert agent.timers_seen == []
