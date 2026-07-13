"""Detector gate schedules and active-window queries.

Gate models answer whether a detector is active at a simulation tick and how
much of an interval is open. They are pure timing primitives: they do not own
ports, qstate, RNG streams, or timeline scheduling.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Protocol

from simyuj.primitives.validation import require_non_negative_int, validate_positive_int


def _tick(field_name: str, value: object) -> int:
    return require_non_negative_int(value, field_name=field_name)


def _interval(start: object, end: object) -> tuple[int, int]:
    start_tick = _tick("start", start)
    end_tick = _tick("end", end)
    if end_tick < start_tick:
        raise ValueError("end must be >= start")
    return start_tick, end_tick


@dataclass(frozen=True, slots=True)
class GateWindow:
    """
    Half-open detector gate interval in simulation ticks.

    Parameters
    ----------
    start : int
        Inclusive non-negative start tick.
    end : int
        Exclusive end tick. Must be strictly greater than ``start``.

    Notes
    -----
    Detector components use gate windows to decide whether an arrival tick can
    open a detection window and how long that window remains active. The
    interval is ``[start, end)``.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        start_tick, end_tick = _interval(self.start, self.end)
        if end_tick == start_tick:
            raise ValueError("GateWindow end must be > start")

        object.__setattr__(self, "start", start_tick)
        object.__setattr__(self, "end", end_tick)

    @property
    def duration_ticks(self) -> int:
        return self.end - self.start

    def contains_tick(self, tick: int) -> bool:
        return self.start <= tick < self.end

    def overlap_duration(self, start: int, end: int) -> int:
        return max(0, min(self.end, end) - max(self.start, start))


class GateModel(Protocol):
    """
    Protocol for detector gate schedules.

    Notes
    -----
    Gate models are queried with simulation ticks. ``window_containing`` returns
    the active half-open gate window for an arrival tick, while
    ``active_duration_between`` reports the active tick count in an interval.
    Implementations do not schedule events themselves.
    """

    def is_open(self, time: int) -> bool: ...

    def window_containing(self, time: int) -> GateWindow | None: ...

    def active_duration_between(self, start: int, end: int) -> int: ...


@dataclass(frozen=True, slots=True)
class AlwaysOpenGate:
    """
    Gate model that treats every non-negative tick as active.

    Notes
    -----
    ``window_containing`` returns the one-tick window containing the query tick.
    Use ``active_duration_between`` when a caller needs the full active span of
    a longer interval.
    """

    def is_open(self, time: int) -> bool:
        _tick("time", time)
        return True

    def window_containing(self, time: int) -> GateWindow | None:
        tick = _tick("time", time)
        return GateWindow(start=tick, end=tick + 1)

    def active_duration_between(self, start: int, end: int) -> int:
        start_tick, end_tick = _interval(start, end)
        return end_tick - start_tick


@dataclass(frozen=True, slots=True)
class PeriodicGate:
    """
    Periodic detector gate schedule.

    Parameters
    ----------
    period_ticks : int
        Positive gate period in simulation ticks.
    open_duration_ticks : int
        Positive active duration in each period. It must not exceed
        ``period_ticks``.
    first_open_tick : int, default=0
        First tick at which the periodic schedule opens.

    Notes
    -----
    Windows repeat as ``[first_open_tick + n * period_ticks,
    first_open_tick + n * period_ticks + open_duration_ticks)``. The model is
    deterministic and consumes no RNG. Ticks before ``first_open_tick`` are
    closed; the periodic schedule is not extrapolated backward.
    """

    period_ticks: int
    open_duration_ticks: int
    first_open_tick: int = 0

    def __post_init__(self) -> None:
        validate_positive_int(self.period_ticks, field_name="period_ticks")
        validate_positive_int(
            self.open_duration_ticks,
            field_name="open_duration_ticks",
        )
        if self.open_duration_ticks > self.period_ticks:
            raise ValueError("open_duration_ticks must be <= period_ticks")
        _tick("first_open_tick", self.first_open_tick)

    def is_open(self, time: int) -> bool:
        return self.window_containing(time) is not None

    def window_containing(self, time: int) -> GateWindow | None:
        tick = _tick("time", time)
        if tick < self.first_open_tick:
            return None

        offset = (tick - self.first_open_tick) % self.period_ticks
        if offset >= self.open_duration_ticks:
            return None

        start = tick - offset
        return GateWindow(start=start, end=start + self.open_duration_ticks)

    def active_duration_between(self, start: int, end: int) -> int:
        start_tick, end_tick = _interval(start, end)
        return self._active_until(end_tick) - self._active_until(start_tick)

    def _active_until(self, end: int) -> int:
        if end <= self.first_open_tick:
            return 0

        elapsed = end - self.first_open_tick
        cycles, remainder = divmod(elapsed, self.period_ticks)
        return cycles * self.open_duration_ticks + min(
            remainder,
            self.open_duration_ticks,
        )


@dataclass(frozen=True, slots=True)
class ScheduledGate:
    """
    Explicit detector gate schedule.

    Parameters
    ----------
    windows : tuple[GateWindow, ...], default=()
        Sorted non-overlapping active intervals.

    Notes
    -----
    The schedule is a static list of half-open windows. It does not merge or
    reorder windows; callers must provide already sorted, non-overlapping
    entries. Adjacent windows are allowed because half-open intervals that meet
    at a boundary do not overlap.
    """

    windows: tuple[GateWindow, ...] = ()
    _starts: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _ends: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.windows, tuple):
            raise TypeError("windows must be tuple[GateWindow, ...]")

        previous_end: int | None = None

        for window in self.windows:
            if not isinstance(window, GateWindow):
                raise TypeError("windows must contain GateWindow")

            if previous_end is not None and window.start < previous_end:
                raise ValueError("windows must be sorted and non-overlapping")

            previous_end = window.end

        object.__setattr__(self, "_starts", tuple(w.start for w in self.windows))
        object.__setattr__(self, "_ends", tuple(w.end for w in self.windows))

    def is_open(self, time: int) -> bool:
        return self.window_containing(time) is not None

    def window_containing(self, time: int) -> GateWindow | None:
        tick = _tick("time", time)

        index = bisect_right(self._starts, tick) - 1

        if index < 0:
            return None

        window = self.windows[index]

        if window.contains_tick(tick):
            return window

        return None

    def active_duration_between(self, start: int, end: int) -> int:
        start_tick, end_tick = _interval(start, end)

        active_duration = 0

        index = bisect_right(self._ends, start_tick)

        for window in self.windows[index:]:
            if window.start >= end_tick:
                break

            active_duration += window.overlap_duration(start_tick, end_tick)

        return active_duration


__all__ = [
    "AlwaysOpenGate",
    "GateModel",
    "GateWindow",
    "PeriodicGate",
    "ScheduledGate",
]
