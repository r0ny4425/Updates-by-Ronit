from __future__ import annotations

import pytest

from simyuj.components.detectors.primitives.gate import (
    AlwaysOpenGate,
    GateWindow,
    PeriodicGate,
    ScheduledGate,
)


def test_gate_window_duration_ticks() -> None:
    assert GateWindow(start=4, end=9).duration_ticks == 5


def test_gate_window_contains_tick_uses_half_open_interval() -> None:
    window = GateWindow(start=4, end=9)

    assert not window.contains_tick(3)
    assert window.contains_tick(4)
    assert window.contains_tick(8)
    assert not window.contains_tick(9)


@pytest.mark.parametrize(
    ("start", "end", "duration"),
    [
        (0, 4, 0),
        (0, 5, 1),
        (5, 8, 3),
        (7, 12, 2),
        (9, 12, 0),
    ],
)
def test_gate_window_overlap_duration(
    start: int,
    end: int,
    duration: int,
) -> None:
    assert GateWindow(start=4, end=9).overlap_duration(start, end) == duration


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": -1, "end": 1},
        {"start": 2, "end": 1},
        {"start": 2, "end": 2},
    ],
)
def test_gate_window_rejects_invalid_bounds(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        GateWindow(**kwargs)


def test_always_open_gate_is_open_for_any_nonnegative_time() -> None:
    gate = AlwaysOpenGate()

    assert gate.is_open(0)
    assert gate.is_open(99)
    assert gate.window_containing(7) == GateWindow(start=7, end=8)
    assert gate.active_duration_between(3, 11) == 8


def test_periodic_gate_queries_open_windows() -> None:
    gate = PeriodicGate(period_ticks=10, open_duration_ticks=3, first_open_tick=2)

    assert not gate.is_open(1)
    assert gate.is_open(2)
    assert gate.is_open(4)
    assert not gate.is_open(5)
    assert gate.window_containing(13) == GateWindow(start=12, end=15)
    assert gate.active_duration_between(0, 20) == 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period_ticks": 0, "open_duration_ticks": 1},
        {"period_ticks": 10, "open_duration_ticks": 0},
        {"period_ticks": 10, "open_duration_ticks": 11},
        {"period_ticks": 10, "open_duration_ticks": 1, "first_open_tick": -1},
    ],
)
def test_periodic_gate_rejects_invalid_config(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        PeriodicGate(**kwargs)


def test_scheduled_gate_queries_windows_and_active_duration() -> None:
    gate = ScheduledGate(
        windows=(
            GateWindow(start=5, end=8),
            GateWindow(start=10, end=13),
        )
    )

    assert not gate.is_open(4)
    assert gate.is_open(5)
    assert gate.window_containing(11) == GateWindow(start=10, end=13)
    assert gate.active_duration_between(6, 12) == 4
    assert gate.active_duration_between(8, 10) == 0
    assert gate.active_duration_between(10, 10) == 0


def test_scheduled_gate_rejects_overlapping_windows() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        ScheduledGate(
            windows=(
                GateWindow(start=5, end=8),
                GateWindow(start=7, end=10),
            )
        )
