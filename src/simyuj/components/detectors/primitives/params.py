"""Physical parameter records for detector-channel models."""

from __future__ import annotations

from dataclasses import dataclass

from simyuj.primitives.validation import (
    require_bool,
    require_non_negative_int,
    require_non_negative_real,
    require_positive_real,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class SinglePhotonDetectorParams:
    """
    Physical parameters for one single-photon detector channel.

    Parameters
    ----------
    efficiency : float, default=1.0
        Probability that a present signal produces a signal-click candidate.
    dark_count_rate_hz : float, default=0.0
        Poisson dark-count rate in hertz.
    dead_time_ticks : int, default=0
        Recovery interval after an accepted click, in simulation ticks.
    jitter_stddev_ticks : float, default=0.0
        Standard deviation, in ticks, for non-negative detector latency.
    p_afterpulse : float, default=0.0
        Integrated probability of an afterpulse over future time before
        window, gate, and dead-time effects.
    afterpulse_decay_ticks : float, default=100.0
        Exponential afterpulse decay constant in simulation ticks.
    photon_number_resolving : bool, default=False
        Whether one evaluation window may return multiple accepted clicks.

    Notes
    -----
    ``p_afterpulse`` is the total integrated probability of an afterpulse over
    all future time before dead time, gating, and observation-window effects.
    It is not a per-window afterpulse probability. ``afterpulse_decay_ticks``
    controls how that probability mass decays after the previous accepted click.

    ``jitter_stddev_ticks`` is the standard deviation, in ticks, of the
    detector response delay. The sampled jitter is clamped to be non-negative
    and added to the photon arrival time to produce the detector firing time.
    If the resulting firing time falls outside the active detection window, no
    click is reported. This models non-negative detector latency, not symmetric
    timestamp noise.

    ``dead_time_ticks`` starts from an accepted click's reported firing time
    after jitter, not from the original signal-arrival tick.
    """

    efficiency: float = 1.0
    dark_count_rate_hz: float = 0.0
    dead_time_ticks: int = 0
    jitter_stddev_ticks: float = 0.0
    p_afterpulse: float = 0.0
    afterpulse_decay_ticks: float = 100.0
    photon_number_resolving: bool = False

    def __post_init__(self) -> None:
        require_probability(self.efficiency, field_name="efficiency")
        require_non_negative_real(
            self.dark_count_rate_hz,
            field_name="dark_count_rate_hz",
            type_name="numeric",
        )
        require_non_negative_int(self.dead_time_ticks, field_name="dead_time_ticks")
        require_non_negative_real(
            self.jitter_stddev_ticks,
            field_name="jitter_stddev_ticks",
            type_name="numeric",
        )
        require_probability(self.p_afterpulse, field_name="p_afterpulse")
        require_positive_real(
            self.afterpulse_decay_ticks,
            field_name="afterpulse_decay_ticks",
            type_name="numeric",
        )
        require_bool(
            self.photon_number_resolving,
            field_name="photon_number_resolving",
        )


__all__ = [
    "SinglePhotonDetectorParams",
]
