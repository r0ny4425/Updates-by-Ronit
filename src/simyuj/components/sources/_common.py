"""Shared source scheduling, timing, validation, and port helpers.

The helpers in this module support concrete source components. They are not a
standalone source model; they keep common event actions, timing profiles,
RNG-binding rules, scalar validation, and source output-port construction in
one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor, isfinite
from typing import TYPE_CHECKING, Protocol

from simyuj.engine.event import Event
from simyuj.primitives.units import Hertz, seconds_to_ticks
from simyuj.qstate.noise import NoiseModel
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme

from ..ports import Port, PortDirection, PortKind

if TYPE_CHECKING:
    from simyuj.engine.component import Component
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


ACTION_START = "start"
ACTION_EMIT = "emit"


@dataclass(frozen=True, slots=True)
class EmissionAttempt:
    """
    Scheduled source-emission payload.

    Parameters
    ----------
    emission_slot_tick : int
        Nominal slot time, in simulation ticks, before timing jitter is applied.
    emission_delay_ticks : int
        Non-negative delay sampled by the timing profile. The concrete source
        handles the emission at ``emission_slot_tick + emission_delay_ticks``.

    Notes
    -----
    Source components pass this immutable record as the ``payload_ref`` for
    ``ACTION_EMIT`` events. The record separates the nominal slot from the
    actual event time so emitted signals can report both scheduling values in
    their timing metadata.
    """

    emission_slot_tick: int
    emission_delay_ticks: int


class EmissionTimingProfile(Protocol):
    """
    Protocol for deterministic source emission-delay sampling.

    Notes
    -----
    Implementations return a non-negative delay in simulation ticks. Source
    schedulers add that delay to the nominal emission slot when scheduling the
    ``ACTION_EMIT`` event.

    ``EntangledPairSource`` and ``SinglePhotonSource`` currently also treat
    this delay as internal source dwell time when applying source noise. If a
    source wants timing jitter without additional dwell-time noise, it should
    not pass this delay as the noise duration.

    Sources using chained one-event scheduling require a finite
    ``max_emission_delay_ticks`` strictly smaller than the emission period.
    """

    @property
    def max_emission_delay_ticks(self) -> int | None: ...

    def sample_emission_delay_ticks(self, rng: DeterministicRNG) -> int: ...


@dataclass(frozen=True, slots=True)
class DeltaTiming:
    """
    Fixed emission delay profile.

    Parameters
    ----------
    emission_delay_ticks : int, default=0
        Non-negative delay, in simulation ticks, returned for every emission
        attempt.

    Notes
    -----
    This profile does not consume RNG state. It is useful for deterministic
    slot-aligned emission and for tests that need exact event times.
    """

    emission_delay_ticks: int = 0

    @property
    def max_emission_delay_ticks(self) -> int:
        return self.emission_delay_ticks

    def __post_init__(self) -> None:
        if type(self.emission_delay_ticks) is not int:
            raise TypeError("emission_delay_ticks must be int")
        if self.emission_delay_ticks < 0:
            raise ValueError("emission_delay_ticks cannot be negative")

    def sample_emission_delay_ticks(self, rng: DeterministicRNG) -> int:
        """Return the configured fixed delay without consuming ``rng``."""
        return self.emission_delay_ticks


@dataclass(frozen=True, slots=True)
class GaussianTiming:
    """
    Gaussian emission-delay profile rounded to simulation ticks.

    Parameters
    ----------
    mean_emission_delay_ticks : float, default=0.0
        Mean of the Gaussian draw, expressed in simulation ticks.
    emission_delay_stddev_ticks : float, default=0.0
        Standard deviation of the Gaussian draw, expressed in simulation ticks.
    max_emission_delay_ticks : int or None, default=None
        Optional upper bound accepted by rejection sampling.
    max_resample_attempts : int, default=100
        Maximum draws attempted before sampling fails.

    Notes
    -----
    Draws are rounded with ``floor(draw + 0.5)`` and clamped to a minimum of
    zero ticks. When ``emission_delay_stddev_ticks`` is zero, the fixed rounded
    delay is used and ``max_emission_delay_ticks`` is filled in when omitted.

    Chained one-event source scheduling requires a finite maximum delay that is
    strictly smaller than the emission period. Nonzero-stddev Gaussian timing
    must therefore be configured with ``max_emission_delay_ticks`` before it is
    used by the current source components.
    """

    mean_emission_delay_ticks: float = 0.0
    emission_delay_stddev_ticks: float = 0.0
    max_emission_delay_ticks: int | None = None
    max_resample_attempts: int = 100

    def __post_init__(self) -> None:
        if type(self.mean_emission_delay_ticks) is bool or not isinstance(
            self.mean_emission_delay_ticks,
            (int, float),
        ):
            raise TypeError("mean_emission_delay_ticks must be numeric")
        if type(self.emission_delay_stddev_ticks) is bool or not isinstance(
            self.emission_delay_stddev_ticks,
            (int, float),
        ):
            raise TypeError("emission_delay_stddev_ticks must be numeric")
        if not isfinite(float(self.mean_emission_delay_ticks)):
            raise ValueError("mean_emission_delay_ticks must be finite")
        if not isfinite(float(self.emission_delay_stddev_ticks)):
            raise ValueError("emission_delay_stddev_ticks must be finite")
        if self.mean_emission_delay_ticks < 0:
            raise ValueError("mean_emission_delay_ticks cannot be negative")
        if self.emission_delay_stddev_ticks < 0:
            raise ValueError("emission_delay_stddev_ticks cannot be negative")
        if self.max_emission_delay_ticks is not None:
            if type(self.max_emission_delay_ticks) is not int:
                raise TypeError("max_emission_delay_ticks must be int or None")
            if self.max_emission_delay_ticks < 0:
                raise ValueError("max_emission_delay_ticks cannot be negative")
        if type(self.max_resample_attempts) is not int:
            raise TypeError("max_resample_attempts must be int")
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")

        if float(self.emission_delay_stddev_ticks) == 0.0:
            fixed_delay = max(0, floor(float(self.mean_emission_delay_ticks) + 0.5))
            if self.max_emission_delay_ticks is None:
                object.__setattr__(self, "max_emission_delay_ticks", fixed_delay)
            elif fixed_delay > self.max_emission_delay_ticks:
                raise ValueError(
                    "fixed Gaussian emission delay exceeds max_emission_delay_ticks"
                )

    def sample_emission_delay_ticks(self, rng: DeterministicRNG) -> int:
        """
        Sample a bounded non-negative Gaussian delay in simulation ticks.

        Raises
        ------
        RuntimeError
            If no sampled delay falls within ``max_emission_delay_ticks`` before
            ``max_resample_attempts`` is exhausted.
        """
        for _ in range(self.max_resample_attempts):
            draw = rng.normal(
                loc=float(self.mean_emission_delay_ticks),
                scale=float(self.emission_delay_stddev_ticks),
            )
            delay = max(0, floor(draw + 0.5))
            if (
                self.max_emission_delay_ticks is None
                or delay <= self.max_emission_delay_ticks
            ):
                return delay

        raise RuntimeError(
            "failed to sample Gaussian emission delay within "
            "max_emission_delay_ticks; increase max_emission_delay_ticks or "
            "max_resample_attempts"
        )


@dataclass(frozen=True, slots=True)
class ExGaussianTiming:
    """
    Gaussian plus exponential emission-delay profile.

    Parameters
    ----------
    gaussian_mean_emission_delay_ticks : float, default=0.0
        Mean of the Gaussian component, expressed in simulation ticks.
    gaussian_emission_delay_stddev_ticks : float, default=0.0
        Standard deviation of the Gaussian component, expressed in simulation
        ticks.
    exponential_emission_delay_scale_ticks : float, default=1.0
        Scale parameter for the exponential component, in simulation ticks.
    max_emission_delay_ticks : int or None, default=None
        Optional upper bound accepted by rejection sampling.
    max_resample_attempts : int, default=100
        Maximum combined draws attempted before sampling fails.

    Notes
    -----
    The Gaussian and exponential draws are added, rounded with
    ``floor(total + 0.5)``, and clamped to a minimum of zero ticks. For chained
    one-event source scheduling, this timing profile must be bounded with
    ``max_emission_delay_ticks``.
    """

    gaussian_mean_emission_delay_ticks: float = 0.0
    gaussian_emission_delay_stddev_ticks: float = 0.0
    exponential_emission_delay_scale_ticks: float = 1.0
    max_emission_delay_ticks: int | None = None
    max_resample_attempts: int = 100

    def __post_init__(self) -> None:
        for name, value in (
            (
                "gaussian_mean_emission_delay_ticks",
                self.gaussian_mean_emission_delay_ticks,
            ),
            (
                "gaussian_emission_delay_stddev_ticks",
                self.gaussian_emission_delay_stddev_ticks,
            ),
            (
                "exponential_emission_delay_scale_ticks",
                self.exponential_emission_delay_scale_ticks,
            ),
        ):
            if type(value) is bool or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.gaussian_mean_emission_delay_ticks < 0:
            raise ValueError("gaussian_mean_emission_delay_ticks cannot be negative")
        if self.gaussian_emission_delay_stddev_ticks < 0:
            raise ValueError("gaussian_emission_delay_stddev_ticks cannot be negative")
        if self.exponential_emission_delay_scale_ticks <= 0:
            raise ValueError("exponential_emission_delay_scale_ticks must be positive")
        if self.max_emission_delay_ticks is not None:
            if type(self.max_emission_delay_ticks) is not int:
                raise TypeError("max_emission_delay_ticks must be int or None")
            if self.max_emission_delay_ticks < 0:
                raise ValueError("max_emission_delay_ticks cannot be negative")
        if type(self.max_resample_attempts) is not int:
            raise TypeError("max_resample_attempts must be int")
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")

    def sample_emission_delay_ticks(self, rng: DeterministicRNG) -> int:
        """
        Sample a bounded non-negative ex-Gaussian delay in simulation ticks.

        Raises
        ------
        RuntimeError
            If no sampled delay falls within ``max_emission_delay_ticks`` before
            ``max_resample_attempts`` is exhausted.
        """
        for _ in range(self.max_resample_attempts):
            gaussian = rng.normal(
                loc=float(self.gaussian_mean_emission_delay_ticks),
                scale=float(self.gaussian_emission_delay_stddev_ticks),
            )
            exponential = rng.exponential(
                float(self.exponential_emission_delay_scale_ticks)
            )
            delay = max(0, floor(gaussian + exponential + 0.5))
            if (
                self.max_emission_delay_ticks is None
                or delay <= self.max_emission_delay_ticks
            ):
                return delay

        raise RuntimeError(
            "failed to sample ex-Gaussian emission delay within "
            "max_emission_delay_ticks; increase max_emission_delay_ticks or "
            "max_resample_attempts"
        )


def emission_period_ticks(frequency_hz: Hertz) -> int:
    """
    Convert source frequency in hertz to an integer emission period.

    Parameters
    ----------
    frequency_hz : Hertz
        Positive finite emission frequency.

    Returns
    -------
    int
        Emission period in simulation ticks.

    Notes
    -----
    The conversion uses the repository's configured seconds-to-ticks helper.
    With the default unit scale this is picosecond resolution. Frequencies that
    imply a period shorter than one tick are rejected.
    """
    if type(frequency_hz) is bool or not isinstance(frequency_hz, (int, float)):
        raise TypeError("frequency_hz must be numeric")
    if not isfinite(float(frequency_hz)) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")

    period_ticks = int(seconds_to_ticks(1.0 / float(frequency_hz)))
    if period_ticks < 1:
        raise ValueError(
            "frequency_hz is too high for 1 ps simulation ticks; "
            "computed period is less than one tick"
        )
    return period_ticks


def start_time_tick(start_time_s: float) -> int:
    """
    Convert source start time in seconds to a simulation tick.

    Parameters
    ----------
    start_time_s : float
        Non-negative finite start time in seconds.
    """
    if type(start_time_s) is bool or not isinstance(start_time_s, (int, float)):
        raise TypeError("start_time_s must be numeric seconds")
    if not isfinite(float(start_time_s)):
        raise ValueError("start_time_s must be finite")
    if start_time_s < 0:
        raise ValueError("start_time_s cannot be negative")
    return int(seconds_to_ticks(float(start_time_s)))


def duration_ticks(duration_s: float | None) -> int | None:
    """
    Convert an optional source duration in seconds to simulation ticks.

    Parameters
    ----------
    duration_s : float or None
        Positive finite duration in seconds, or ``None`` for an unbounded
        source.

    Returns
    -------
    int or None
        Duration in simulation ticks, or ``None`` when no stop time is
        configured.
    """
    if duration_s is None:
        return None
    if type(duration_s) is bool or not isinstance(duration_s, (int, float)):
        raise TypeError("duration_s must be numeric seconds or None")
    if not isfinite(float(duration_s)):
        raise ValueError("duration_s must be finite")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive when provided")
    ticks = int(seconds_to_ticks(float(duration_s)))
    if ticks < 1:
        raise ValueError(
            "duration_s is too short for 1 ps simulation ticks; "
            "computed duration is less than one tick"
        )
    return ticks


def validate_source_scalars(
    *,
    encoding_scheme: EncodingScheme,
    emission_probability: float,
    wavelength_nm: float,
) -> None:
    """
    Validate scalar constructor fields shared by source components.

    Notes
    -----
    This is public-boundary validation for source construction. It checks the
    encoding scheme type, emission probability range, and positive finite
    wavelength before concrete sources schedule any events.
    """
    if not isinstance(encoding_scheme, EncodingScheme):
        raise TypeError("encoding_scheme must be EncodingScheme")

    if type(emission_probability) is bool or not isinstance(
        emission_probability,
        (int, float),
    ):
        raise TypeError("emission_probability must be numeric")
    if not isfinite(float(emission_probability)):
        raise ValueError("emission_probability must be finite")
    if not 0.0 <= float(emission_probability) <= 1.0:
        raise ValueError("emission_probability must be in [0, 1]")

    if type(wavelength_nm) is bool or not isinstance(wavelength_nm, (int, float)):
        raise TypeError("wavelength_nm must be numeric")
    if not isfinite(float(wavelength_nm)) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and positive")


def validate_timing_profile(timing_profile: object) -> None:
    """
    Validate the minimal timing-profile protocol.

    Notes
    -----
    Only the sampling method is required here. Bounded-delay constraints are
    checked separately because they are specific to the chained source
    scheduler.
    """
    sampler = getattr(timing_profile, "sample_emission_delay_ticks", None)
    if not callable(sampler):
        raise TypeError(
            "timing_profile must implement sample_emission_delay_ticks(rng)"
        )


def validate_timing_profile_for_chained_scheduler(
    *,
    timing_profile: EmissionTimingProfile,
    emission_period_tick_count: int,
) -> None:
    """
    Validate timing-profile bounds for chained one-event source scheduling.

    Notes
    -----
    Current source components schedule one future ``ACTION_EMIT`` at a time.
    The timing profile must therefore expose a finite
    ``max_emission_delay_ticks`` strictly smaller than the emission period, so
    an emission event cannot land after a later nominal slot that has not yet
    been scheduled.
    """
    validate_timing_profile(timing_profile)

    if type(emission_period_tick_count) is not int:
        raise TypeError("emission_period_tick_count must be int")
    if emission_period_tick_count <= 0:
        raise ValueError("emission_period_tick_count must be positive")

    max_delay = getattr(timing_profile, "max_emission_delay_ticks", None)
    if max_delay is None:
        raise ValueError(
            "timing_profile has unbounded emission delay. The chained one-event "
            "source scheduler requires finite max_emission_delay_ticks that are "
            "strictly smaller than emission_period_ticks."
        )
    if type(max_delay) is not int:
        raise TypeError("timing_profile.max_emission_delay_ticks must be int or None")
    if max_delay < 0:
        raise ValueError("timing_profile.max_emission_delay_ticks cannot be negative")
    if max_delay >= emission_period_tick_count:
        raise ValueError(
            "timing_profile.max_emission_delay_ticks must be strictly smaller "
            "than emission_period_ticks for the chained one-event source scheduler"
        )


def validate_sampled_emission_delay_ticks(
    emission_delay_ticks: object,
    *,
    emission_period_tick_count: int,
) -> int:
    """
    Validate a sampled emission delay for the chained scheduler.

    Returns
    -------
    int
        The validated delay in simulation ticks.

    Notes
    -----
    This check is applied after timing RNG sampling and before scheduling the
    ``ACTION_EMIT`` event. It protects deterministic event ordering by
    rejecting delays that would reach or overrun the next nominal slot.
    """
    if type(emission_period_tick_count) is not int:
        raise TypeError("emission_period_tick_count must be int")
    if emission_period_tick_count <= 0:
        raise ValueError("emission_period_tick_count must be positive")
    if type(emission_delay_ticks) is not int:
        raise TypeError("timing_profile must return int delay ticks")
    if emission_delay_ticks < 0:
        raise ValueError("timing_profile returned negative delay ticks")
    if emission_delay_ticks >= emission_period_tick_count:
        raise ValueError(
            "timing_profile produced emission_delay_ticks greater than or equal "
            "to emission_period_ticks. This is invalid for the chained one-event "
            "source scheduler."
        )
    return emission_delay_ticks


def normalize_noise_models(
    noise_models: Sequence[NoiseModel],
    *,
    field_name: str,
) -> tuple[NoiseModel, ...]:
    """
    Normalize a source noise-model sequence to an immutable tuple.

    Notes
    -----
    The contained noise-model objects are not copied or validated here. Order
    is preserved because concrete sources apply configured noise models in
    tuple order, and the qstate layer resolves each model when noise is
    applied.
    """
    if isinstance(noise_models, (str, bytes)) or not isinstance(noise_models, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(noise_models)


def quantum_output_port(*, owner: Component, owner_id: str, name: str) -> Port:
    """
    Build a quantum output port owned by a source component.

    Parameters
    ----------
    owner : Component
        Component instance that owns the port.
    owner_id : str
        Public device identifier recorded on the port.
    name : str
        Port name, such as ``"out"``, ``"left"``, or ``"right"``.
    """
    return Port(
        name=name,
        owner=owner,
        owner_id=owner_id,
        port_kind=PortKind.QUANTUM,
        direction=PortDirection.EGRESS,
    )


def stop_time(*, start_tick: int, duration_tick_count: int | None) -> int | None:
    """
    Return the exclusive stop tick for a source schedule.

    Notes
    -----
    A ``None`` duration means the source has no configured stop time.
    """
    if duration_tick_count is None:
        return None
    return start_tick + duration_tick_count


def is_before_stop(
    time: int,
    *,
    start_tick: int,
    duration_tick_count: int | None,
) -> bool:
    """Return whether ``time`` is inside the source's configured time window."""
    limit = stop_time(start_tick=start_tick, duration_tick_count=duration_tick_count)
    return limit is None or time < limit


def bind_source_rngs(
    *,
    source: object,
    context: BindingContext,
    device_id: str,
    component_key: str,
    bound_timeline_id: int | None,
) -> tuple[int | None, DeterministicRNG, DeterministicRNG, DeterministicRNG]:
    """
    Bind deterministic source RNG streams to a timeline.

    Returns
    -------
    tuple
        ``(timeline_id, emission_rng, state_rng, timing_rng)`` for the source.

    Notes
    -----
    The streams are declared through ``Timeline.rng(device_id, component_key,
    stream_name)`` using ``"emission"``, ``"state"``, and ``"timing"`` stream
    names. Rebinding to the same timeline returns the same stream handles;
    binding to a different timeline is rejected.
    """
    del source

    if not isinstance(context, BindingContext):
        raise TypeError("context must be BindingContext")

    timeline = context.timeline
    timeline_id = id(timeline)
    if bound_timeline_id is not None:
        if bound_timeline_id != timeline_id:
            raise RuntimeError("source is already bound to another timeline")
        return (
            bound_timeline_id,
            timeline.rng(device_id, component_key, "emission"),
            timeline.rng(device_id, component_key, "state"),
            timeline.rng(device_id, component_key, "timing"),
        )

    return (
        timeline_id,
        timeline.rng(device_id, component_key, "emission"),
        timeline.rng(device_id, component_key, "state"),
        timeline.rng(device_id, component_key, "timing"),
    )


def schedule_start_event(
    *,
    source: Component,
    timeline: Timeline,
    start_tick: int,
    device_id: str,
) -> Event:
    """
    Schedule the source ``ACTION_START`` event.

    Notes
    -----
    The event targets the source component at ``start_tick`` and carries the
    device identifier in event metadata. Concrete source classes use this as
    the public activation event before any emission attempts are scheduled.
    """
    return timeline.schedule(
        Event(
            time=start_tick,
            target_ref=source,
            action=ACTION_START,
            payload_ref=None,
            source=source,
            subsystem_id="components",
            meta={"device_id": device_id},
        )
    )


def schedule_next_emission_event(
    *,
    source: Component,
    timeline: Timeline,
    device_id: str,
    emission_slot_tick: int,
    emission_period_tick_count: int,
    start_tick: int,
    duration_tick_count: int | None,
    timing_profile: EmissionTimingProfile,
    timing_rng: DeterministicRNG,
) -> None:
    """
    Schedule the next source ``ACTION_EMIT`` event if it is inside the window.

    Notes
    -----
    The timing profile samples its delay with the source timing RNG at
    scheduling time. Both the nominal slot and the delayed event time must be
    before the exclusive stop tick when a duration is configured.

    Near the end of a finite source duration, this means the final nominal slot
    may be ignored if its delayed emission would land outside the active
    window.

    The scheduled event carries an :class:`EmissionAttempt` payload and records
    ``device_id``, ``emission_slot_tick``, and ``emission_delay_ticks`` in event
    metadata. If the sampled event time would be earlier than the timeline's
    current time, scheduling fails instead of silently changing event order.
    """
    if not is_before_stop(
        emission_slot_tick,
        start_tick=start_tick,
        duration_tick_count=duration_tick_count,
    ):
        return

    emission_delay_ticks = validate_sampled_emission_delay_ticks(
        timing_profile.sample_emission_delay_ticks(timing_rng),
        emission_period_tick_count=emission_period_tick_count,
    )
    emission_event_tick = emission_slot_tick + emission_delay_ticks

    if not is_before_stop(
        emission_event_tick,
        start_tick=start_tick,
        duration_tick_count=duration_tick_count,
    ):
        return

    if emission_event_tick < timeline.current_time:
        raise RuntimeError("timing profile produced an event in the past")

    timeline.schedule(
        Event(
            time=emission_event_tick,
            target_ref=source,
            action=ACTION_EMIT,
            payload_ref=EmissionAttempt(
                emission_slot_tick=emission_slot_tick,
                emission_delay_ticks=emission_delay_ticks,
            ),
            source=source,
            subsystem_id="components",
            meta={
                "device_id": device_id,
                "emission_slot_tick": emission_slot_tick,
                "emission_delay_ticks": emission_delay_ticks,
            },
        )
    )


__all__ = [
    "ACTION_EMIT",
    "ACTION_START",
    "DeltaTiming",
    "EmissionAttempt",
    "EmissionTimingProfile",
    "ExGaussianTiming",
    "GaussianTiming",
    "bind_source_rngs",
    "duration_ticks",
    "emission_period_ticks",
    "is_before_stop",
    "normalize_noise_models",
    "quantum_output_port",
    "schedule_next_emission_event",
    "schedule_start_event",
    "start_time_tick",
    "stop_time",
    "validate_sampled_emission_delay_ticks",
    "validate_source_scalars",
    "validate_timing_profile",
    "validate_timing_profile_for_chained_scheduler",
]
