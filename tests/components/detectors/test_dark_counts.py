from __future__ import annotations

import math
from typing import Any, cast

import pytest

from simyuj.components.detectors.primitives.dark_counts import (
    DarkCountProcess,
    OnArrivalWindowDarkCounts,
)


class FakeRNG:
    def __init__(
        self,
        *,
        poisson_values: tuple[int, ...] = (),
        integer_values: tuple[int, ...] = (),
        random_values: tuple[float, ...] = (),
    ) -> None:
        self._poisson_values = list(poisson_values)
        self._integer_values = list(integer_values)
        self._random_values = list(random_values)

        self.poisson_lambdas: list[float] = []

    def poisson(self, lam: float) -> int:
        self.poisson_lambdas.append(lam)

        if not self._poisson_values:
            raise AssertionError("unexpected poisson call")

        return self._poisson_values.pop(0)

    def integers(self, low: int, high: int) -> int:
        if not self._integer_values:
            raise AssertionError("unexpected integers call")

        value = self._integer_values.pop(0)

        assert low <= value < high

        return value

    def random(self) -> float:
        if not self._random_values:
            raise AssertionError("unexpected random call")

        return self._random_values.pop(0)


def test_dark_count_process_rejects_invalid_rate() -> None:
    with pytest.raises(ValueError):
        DarkCountProcess(rate_hz=-1.0)

    with pytest.raises(TypeError):
        DarkCountProcess(rate_hz=True)

    with pytest.raises(ValueError):
        DarkCountProcess(rate_hz=float("inf"))


def test_p_at_least_one_matches_poisson_formula() -> None:
    process = DarkCountProcess(rate_hz=10.0)

    probability = process.p_at_least_one(0.25)

    assert probability == pytest.approx(1.0 - math.exp(-2.5))


def test_p_at_least_one_rejects_invalid_duration() -> None:
    process = DarkCountProcess(rate_hz=10.0)

    with pytest.raises(ValueError):
        process.p_at_least_one(-1.0)

    with pytest.raises(TypeError):
        process.p_at_least_one(True)


def test_sample_count_uses_rng_poisson() -> None:
    process = DarkCountProcess(rate_hz=20.0)
    rng = FakeRNG(poisson_values=(3,))

    count = process.sample_count(0.5, rng)

    assert count == 3
    assert rng.poisson_lambdas == [10.0]


def test_sample_count_rejects_negative_rng_result() -> None:
    process = DarkCountProcess(rate_hz=20.0)
    rng = FakeRNG(poisson_values=(-1,))

    with pytest.raises(ValueError):
        process.sample_count(0.5, rng)


def test_arrival_window_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        OnArrivalWindowDarkCounts(window_duration_ticks=-1)

    with pytest.raises(TypeError):
        OnArrivalWindowDarkCounts(window_duration_ticks=cast(Any, 1.0))

    with pytest.raises(TypeError):
        OnArrivalWindowDarkCounts(
            window_duration_ticks=1,
            time_resolved=cast(Any, 1),
        )

    with pytest.raises(TypeError):
        OnArrivalWindowDarkCounts(
            window_duration_ticks=1,
            return_all_clicks=cast(Any, 1),
        )


def test_zero_window_returns_no_clicks() -> None:
    policy = OnArrivalWindowDarkCounts(window_duration_ticks=0)
    process = DarkCountProcess(rate_hz=1e12)
    rng = FakeRNG()

    assert policy.sample_dark_clicks(time=100, process=process, rng=rng) == ()


def test_zero_rate_returns_no_clicks() -> None:
    policy = OnArrivalWindowDarkCounts(window_duration_ticks=50)
    process = DarkCountProcess(rate_hz=0.0)
    rng = FakeRNG()

    assert policy.sample_dark_clicks(time=100, process=process, rng=rng) == ()


def test_zero_poisson_count_returns_no_clicks() -> None:
    policy = OnArrivalWindowDarkCounts(window_duration_ticks=50)
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(poisson_values=(0,))

    assert policy.sample_dark_clicks(time=100, process=process, rng=rng) == ()


def test_coarse_model_returns_arrival_time_when_any_dark_count_occurs() -> None:
    policy = OnArrivalWindowDarkCounts(
        window_duration_ticks=50,
        time_resolved=False,
    )
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(poisson_values=(3,))

    times = policy.sample_dark_clicks(
        time=100,
        process=process,
        rng=rng,
    )

    assert times == (100,)


def test_time_resolved_model_returns_sampled_offset() -> None:
    policy = OnArrivalWindowDarkCounts(
        window_duration_ticks=50,
        time_resolved=True,
    )
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(
        poisson_values=(1,),
        integer_values=(17,),
    )

    times = policy.sample_dark_clicks(
        time=100,
        process=process,
        rng=rng,
    )

    assert times == (117,)
    assert 100 <= times[0] < 150


def test_time_resolved_model_returns_earliest_click_by_default() -> None:
    policy = OnArrivalWindowDarkCounts(
        window_duration_ticks=50,
        time_resolved=True,
        return_all_clicks=False,
    )
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(
        poisson_values=(3,),
        integer_values=(40, 7, 25),
    )

    times = policy.sample_dark_clicks(
        time=100,
        process=process,
        rng=rng,
    )

    assert times == (107,)


def test_time_resolved_model_can_return_all_clicks() -> None:
    policy = OnArrivalWindowDarkCounts(
        window_duration_ticks=50,
        time_resolved=True,
        return_all_clicks=True,
    )
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(
        poisson_values=(3,),
        integer_values=(40, 7, 25),
    )

    times = policy.sample_dark_clicks(
        time=100,
        process=process,
        rng=rng,
    )

    assert times == (107, 125, 140)


def test_invalid_time_is_rejected() -> None:
    policy = OnArrivalWindowDarkCounts(window_duration_ticks=50)
    process = DarkCountProcess(rate_hz=1e9)
    rng = FakeRNG(poisson_values=(1,))

    with pytest.raises(ValueError):
        policy.sample_dark_clicks(time=-1, process=process, rng=rng)

    with pytest.raises(TypeError):
        policy.sample_dark_clicks(time=cast(Any, 100.0), process=process, rng=rng)


def test_invalid_process_is_rejected() -> None:
    policy = OnArrivalWindowDarkCounts(window_duration_ticks=50)
    rng = FakeRNG(poisson_values=(1,))

    with pytest.raises(TypeError):
        policy.sample_dark_clicks(time=100, process=cast(Any, object()), rng=rng)
