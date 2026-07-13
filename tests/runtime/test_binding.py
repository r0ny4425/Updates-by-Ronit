from __future__ import annotations

import pytest

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.rng_manager import DeterministicRNG
from simyuj.engine.timeline import Timeline
from simyuj.runtime.binding import BindingContext, bind_if_supported, bind_many


class BoundStreamComponent(Component):
    """
    Dummy stochastic component that declares its stream in bind().
    """

    def __init__(self, component_id: str) -> None:
        self.component_id = component_id
        self._rng: DeterministicRNG | None = None
        self.draws: list[float] = []
        self.bound = False

    def bind(self, context: BindingContext) -> None:
        self._rng = context.timeline.rng("dummy", self.component_id, "loss")
        self.bound = True

    def handle_event(self, event, timeline) -> None:
        if self._rng is None:
            raise RuntimeError("BoundStreamComponent must be bound before execution")
        self.draws.append(self._rng.random())


class LateStreamRequestComponent(Component):
    """
    Dummy component that incorrectly asks for a new stream during execution.
    """

    def handle_event(self, event, timeline) -> None:
        timeline.rng("late_component", "new_stream")


class NonCallableBind:
    bind = object()


def make_event(*, time: int, target_ref: object) -> Event:
    return Event(
        time=time,
        target_ref=target_ref,
        action="TEST",
        payload_ref=None,
    )


def test_bind_if_supported_and_bind_many_bind_only_supported_entities():
    timeline = Timeline(master_seed=11)
    stochastic = BoundStreamComponent(component_id="channel_A")
    plain_object = object()

    assert bind_if_supported(plain_object, timeline) is False

    bound_entities = bind_many((plain_object, stochastic), timeline)
    assert bound_entities == (stochastic,)
    assert stochastic.bound is True


def test_bind_if_supported_rejects_non_callable_bind_attribute():
    timeline = Timeline(master_seed=11)

    with pytest.raises(TypeError, match="non-callable 'bind'"):
        bind_if_supported(NonCallableBind(), timeline)


def test_bound_component_uses_predeclared_rng_stream_after_freeze():
    timeline = Timeline(master_seed=1234)
    component = BoundStreamComponent(component_id="channel_A")

    expected_timeline = Timeline(master_seed=1234)
    expected_rng = expected_timeline.rng("dummy", "channel_A", "loss")
    expected_value = expected_rng.random()

    assert bind_if_supported(component, timeline) is True
    timeline.schedule(make_event(time=5, target_ref=component))
    timeline.run_one_step()

    assert component.draws == [expected_value]


def test_requesting_new_stream_after_execution_starts_still_fails():
    timeline = Timeline(master_seed=3)
    timeline.schedule(make_event(time=1, target_ref=LateStreamRequestComponent()))

    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.run_one_step()
