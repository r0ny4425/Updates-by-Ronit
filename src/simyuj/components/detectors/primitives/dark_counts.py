"""Dark-count probability and timing primitives for detector windows.

The helpers here model dark counts as a Poisson process over an active detector
window. They return candidate click ticks to ``SinglePhotonDetector`` and keep
RNG consumption explicit by requiring the caller to pass a deterministic stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1

from simyuj.primitives.units import ticks_to_seconds
from simyuj.primitives.validation import (
    require_bool,
    require_non_negative_int,
    require_non_negative_real,
)


def _sample_tick_offset(rng, width_ticks: int) -> int:
    """Sample an integer offset in [0, width_ticks).

    The simulator RNG may expose either `integers(low, high)` or only
    `random()`. This helper keeps the dark-count policy independent of the
    exact RNG implementation.
    """
    if width_ticks <= 0:
        raise ValueError("width_ticks must be positive")

    if hasattr(rng, "integers"):
        return int(rng.integers(0, width_ticks))

    if hasattr(rng, "randint"):
        # Common fallback: randint(a, b) is usually inclusive.
        return int(rng.randint(0, width_ticks - 1))

    if hasattr(rng, "random"):
        # Assumes random() returns a float in [0.0, 1.0).
        return min(width_ticks - 1, int(float(rng.random()) * width_ticks))

    raise TypeError("rng must provide integers(), randint(), or random()")


@dataclass(frozen=True, slots=True)
class DarkCountProcess:
    """
    Poisson dark-count process for one detector channel.

    Parameters
    ----------
    rate_hz : float
        Non-negative dark-count rate in hertz.

    Notes
    -----
    ``p_at_least_one`` implements
    :math:`P(N \\ge 1) = 1 - e^{-rT}` using ``expm1`` for small-duration
    stability. ``sample_count`` delegates the Poisson draw to the supplied
    deterministic RNG object.

    Detector-window policies pass durations in seconds, so callers that start
    from simulation ticks must convert through the shared simulator tick unit.
    """

    rate_hz: float

    def __post_init__(self) -> None:
        require_non_negative_real(
            self.rate_hz,
            field_name="rate_hz",
            type_name="numeric",
        )

    def p_at_least_one(self, active_duration_s: float) -> float:
        duration_s = require_non_negative_real(
            active_duration_s,
            field_name="active_duration_s",
            type_name="numeric",
        )

        # More numerically stable than 1.0 - exp(-x), especially for small x.
        return -expm1(-float(self.rate_hz) * duration_s)

    def sample_count(self, active_duration_s: float, rng) -> int:
        duration_s = require_non_negative_real(
            active_duration_s,
            field_name="active_duration_s",
            type_name="numeric",
        )

        count = int(rng.poisson(float(self.rate_hz) * duration_s))

        if count < 0:
            raise ValueError("rng.poisson returned a negative count")

        return count


@dataclass(frozen=True, slots=True)
class OnArrivalWindowDarkCounts:
    """
    Lazy dark-count policy for signal-triggered detector windows.

    Parameters
    ----------
    window_duration_ticks : int
        Non-negative detector observation window length in simulation ticks.
    time_resolved : bool, default=False
        Whether sampled dark counts receive tick offsets inside the active
        window.
    return_all_clicks : bool, default=False
        Whether to return every sampled dark click rather than only the first.

    Notes
    -----
    With ``time_resolved=False``, the whole active window is one threshold bin:
    any positive Poisson count becomes a single dark click at the arrival tick.
    With ``time_resolved=True``, count offsets are sampled uniformly within the
    window. Ordinary threshold detectors use the earliest dark click; returning
    all clicks is intended for time-resolved or photon-number-resolving models.
    This uniform placement is a compact timing assumption after the Poisson
    count is known; it is not a full optical/electronic pulse model.
    """

    window_duration_ticks: int
    time_resolved: bool = False
    return_all_clicks: bool = False

    def __post_init__(self) -> None:
        require_non_negative_int(
            self.window_duration_ticks,
            field_name="window_duration_ticks",
        )
        require_bool(self.time_resolved, field_name="time_resolved")
        require_bool(self.return_all_clicks, field_name="return_all_clicks")

    def sample_dark_clicks(
        self,
        *,
        time: int,
        process: DarkCountProcess,
        rng,
    ) -> tuple[int, ...]:
        start_time = require_non_negative_int(time, field_name="time")

        if not isinstance(process, DarkCountProcess):
            raise TypeError("process must be DarkCountProcess")

        if self.window_duration_ticks == 0:
            return ()

        if float(process.rate_hz) == 0.0:
            return ()

        active_duration_s = ticks_to_seconds(self.window_duration_ticks)
        count = process.sample_count(active_duration_s, rng)

        if count <= 0:
            return ()

        # Coarse threshold-window model:
        # at least one dark event inside the active window becomes one click
        # at the signal-arrival tick.
        if not self.time_resolved:
            return (start_time,)

        offsets = sorted(
            _sample_tick_offset(rng, self.window_duration_ticks) for _ in range(count)
        )

        if not self.return_all_clicks:
            return (start_time + offsets[0],)

        return tuple(start_time + offset for offset in offsets)


__all__ = [
    "DarkCountProcess",
    "OnArrivalWindowDarkCounts",
]
