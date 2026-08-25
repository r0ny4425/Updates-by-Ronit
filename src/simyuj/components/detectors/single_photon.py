"""Stateful single-photon detector channel model.

This module evaluates one detector channel over an already-open observation
window. It samples signal clicks, dark counts, jitter, and afterpulses from
caller-provided deterministic RNG streams, then returns low-level ``RawClick``
records for event-facing components to resolve into reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import validate_meta
from simyuj.primitives.validation import require_non_negative_int, validate_bool

from .primitives.dark_counts import DarkCountProcess, OnArrivalWindowDarkCounts
from .primitives.params import SinglePhotonDetectorParams
from .primitives.reports import (
    FLAG_AFTERPULSE,
    FLAG_DARK_COUNT,
    FLAG_SIGNAL_CLICK,
    RawClick,
)
from .primitives.rng import DetectorRNGStreams

# Same-time candidates are resolved deterministically by physical trigger role.
_TRIGGER_ORDER = {
    "signal": 0,
    "dark": 1,
    "afterpulse": 2,
}


@dataclass(frozen=True, slots=True)
class _ClickCandidate:
    time: int
    trigger: str
    flags: tuple[str, ...]


@dataclass(slots=True)
class SinglePhotonDetector:
    """
    Stateful single-photon detector channel model.

    Parameters
    ----------
    detector_id : str
        Non-empty channel identifier used in raw clicks and detector arrays.
    params : SinglePhotonDetectorParams, optional
        Physical and stochastic detector parameters.

    Attributes
    ----------
    dead_until : int
        Earliest tick at which the channel can click again.
    last_click_time : int or None
        Most recent accepted click tick, used by the afterpulse model.

    Notes
    -----
    This class models one detector channel, not an event-target component. It
    owns local recovery state but no ports, qstate references, or timeline. The
    owning component supplies deterministic RNG streams and a detection window
    when evaluating a signal arrival.

    ``dead_until`` and ``last_click_time`` persist across ``evaluate_window``
    calls. Reusing the same detector instance is therefore part of the physical
    model: dead time and afterpulsing depend on previous accepted clicks.
    """

    detector_id: str
    params: SinglePhotonDetectorParams = field(
        default_factory=SinglePhotonDetectorParams
    )

    dead_until: int = field(init=False, default=0)
    last_click_time: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.detector_id, field_name="detector_id")
        if not isinstance(self.params, SinglePhotonDetectorParams):
            raise TypeError("params must be SinglePhotonDetectorParams")

    def evaluate_window(
        self,
        *,
        time: int,
        signal_present: bool,
        window_duration_ticks: int,
        rngs: DetectorRNGStreams,
        signal_click_probability: float | None = None,
        outcome_label: str | None = None,
        dark_count_policy: OnArrivalWindowDarkCounts | None = None,
        meta: tuple[tuple[str, object], ...] = (),
    ) -> tuple[RawClick, ...]:
        """
        Evaluate signal, dark-count, jitter, and afterpulse candidates.

        Parameters
        ----------
        time : int
            Start tick of the detector observation window.
        signal_present : bool
            Whether this detector is exposed to a signal click candidate.
        window_duration_ticks : int
            Non-negative active observation duration in simulation ticks.
        rngs : DetectorRNGStreams
            Deterministic RNG streams for efficiency, dark counts, jitter, and
            afterpulsing.
        signal_click_probability : float or None, default=None
            Probability of a signal click in this window, **replacing**
            ``params.efficiency``. ``None`` uses that efficiency.
        outcome_label : str or None, default=None
            Logical label copied to emitted raw clicks.
        dark_count_policy : OnArrivalWindowDarkCounts or None, default=None
            Optional dark-count timing policy. When omitted, a policy matching
            ``window_duration_ticks`` is created.
        meta : tuple[tuple[str, object], ...], default=()
            Metadata copied to emitted raw clicks.

        Returns
        -------
        tuple[RawClick, ...]
            Accepted raw clicks, possibly empty.

        Notes
        -----
        If ``time`` is before ``dead_until``, the whole window is blocked. The
        stage-one model does not recover midway through a window. Accepted
        candidates are sorted by time and trigger priority
        ``signal < dark < afterpulse``. A threshold detector returns only the
        first accepted click; a photon-number-resolving detector may return
        multiple clicks that are not blocked by updated dead time.

        ``signal_click_probability`` **overrides** ``params.efficiency``; it
        does not multiply it. It exists for a payload whose detection
        probability is not a bare efficiency -- a coherent pulse, where
        ``coherent_optics.click_probability`` returns ``1 - exp(-eta_d * mu)``
        with this detector's own ``eta_d`` already inside the exponent.
        Multiplying would apply ``eta_d`` twice and lower every click rate by
        that factor with nothing raising, because a uniformly low rate is
        indistinguishable from a lossier link. It governs the signal click
        alone: dark counts and afterpulses read ``params`` on every path, which
        is why this overrides one term rather than replacing the parameters.

        Efficiency ``0.0`` and ``1.0`` avoid consuming the efficiency RNG, and
        so do those two values supplied as ``signal_click_probability`` -- the
        branch structure is identical either way, so threading a probability
        through cannot shift a stream position that an efficiency would not
        have shifted. Jitter is non-negative detector latency sampled from a
        normal distribution and clamped at zero.

        Detector state is updated only when a click is accepted. Dark counts
        and afterpulses are still sampled when ``signal_present=False``; only
        signal-efficiency sampling is skipped. Afterpulse sampling is
        conditional on ``last_click_time`` and requires an afterpulse RNG only
        when afterpulsing can occur.
        """

        tick = require_non_negative_int(time, field_name="time")
        window_duration_ticks = require_non_negative_int(
            window_duration_ticks,
            field_name="window_duration_ticks",
        )

        validate_bool(signal_present, field_name="signal_present")

        if not isinstance(rngs, DetectorRNGStreams):
            raise TypeError("rngs must be DetectorRNGStreams")

        if outcome_label is not None and not isinstance(outcome_label, str):
            raise TypeError("outcome_label must be str or None")

        validate_meta(meta, field_name="meta", require_hashable=False)

        policy = self._resolve_dark_count_policy(
            window_duration_ticks=window_duration_ticks,
            dark_count_policy=dark_count_policy,
        )

        # Stage-1 model choice:
        # if the detector is already dead at the start of this arrival window,
        # the whole arrival window is blocked.
        #
        # This keeps the first implementation simple and deterministic.
        # Later, if you want detector recovery inside the same time-resolved
        # window, this check can move into the candidate loop.
        if tick < self.dead_until:
            return ()

        candidates = self._build_click_candidates(
            time=tick,
            signal_present=signal_present,
            signal_click_probability=signal_click_probability,
            window_duration_ticks=window_duration_ticks,
            rngs=rngs,
            dark_count_policy=policy,
        )

        if not candidates:
            return ()

        window_end_exclusive = tick + window_duration_ticks

        candidates = [
            candidate
            for candidate in candidates
            if tick <= candidate.time < window_end_exclusive
        ]

        if not candidates:
            return ()

        candidates.sort(
            key=lambda candidate: (
                candidate.time,
                _TRIGGER_ORDER[candidate.trigger],
            )
        )

        raw_clicks: list[RawClick] = []

        for candidate in candidates:
            if candidate.time < self.dead_until:
                continue

            raw_clicks.append(
                RawClick(
                    detector_id=self.detector_id,
                    time=candidate.time,
                    trigger=candidate.trigger,
                    outcome_label=outcome_label,
                    flags=candidate.flags,
                    meta=meta,
                )
            )

            self.dead_until = candidate.time + int(self.params.dead_time_ticks)
            self.last_click_time = candidate.time

            if not self.params.photon_number_resolving:
                break

        return tuple(raw_clicks)

    def _resolve_dark_count_policy(
        self,
        *,
        window_duration_ticks: int,
        dark_count_policy: OnArrivalWindowDarkCounts | None,
    ) -> OnArrivalWindowDarkCounts:
        if dark_count_policy is None:
            return OnArrivalWindowDarkCounts(
                window_duration_ticks=window_duration_ticks,
            )

        if not isinstance(dark_count_policy, OnArrivalWindowDarkCounts):
            raise TypeError(
                "dark_count_policy must be OnArrivalWindowDarkCounts or None"
            )

        if dark_count_policy.window_duration_ticks != window_duration_ticks:
            raise ValueError(
                "dark_count_policy.window_duration_ticks must match "
                "window_duration_ticks"
            )

        return dark_count_policy

    def _build_click_candidates(
        self,
        *,
        time: int,
        signal_present: bool,
        signal_click_probability: float | None,
        window_duration_ticks: int,
        rngs: DetectorRNGStreams,
        dark_count_policy: OnArrivalWindowDarkCounts,
    ) -> list[_ClickCandidate]:
        candidates: list[_ClickCandidate] = []

        if self._sample_signal_click(
            signal_present=signal_present,
            signal_click_probability=signal_click_probability,
            rng=rngs.efficiency,
        ):
            candidates.append(
                _ClickCandidate(
                    time=self._apply_jitter(time, rng=rngs.jitter),
                    trigger="signal",
                    flags=(FLAG_SIGNAL_CLICK,),
                )
            )

        dark_times = self._sample_dark_clicks(
            time=time,
            window_duration_ticks=window_duration_ticks,
            rng=rngs.dark,
            dark_count_policy=dark_count_policy,
        )

        for dark_time in dark_times:
            candidates.append(
                _ClickCandidate(
                    time=self._apply_jitter(dark_time, rng=rngs.jitter),
                    trigger="dark",
                    flags=(FLAG_DARK_COUNT,),
                )
            )

        afterpulse_time = self._sample_afterpulse_time(
            time=time,
            window_duration_ticks=window_duration_ticks,
            rngs=rngs,
        )

        if afterpulse_time is not None:
            candidates.append(
                _ClickCandidate(
                    time=self._apply_jitter(afterpulse_time, rng=rngs.jitter),
                    trigger="afterpulse",
                    flags=(FLAG_AFTERPULSE,),
                )
            )

        return candidates

    def _sample_signal_click(
        self,
        *,
        signal_present: bool,
        signal_click_probability: float | None,
        rng,
    ) -> bool:
        if not signal_present:
            return False

        # Override, never multiply. `params.efficiency` is already inside a
        # supplied probability, so applying it again would divide every click
        # rate by it -- silently, and plausibly. See `evaluate_window`.
        probability = (
            float(self.params.efficiency)
            if signal_click_probability is None
            else float(signal_click_probability)
        )

        if probability <= 0.0:
            return False

        if probability >= 1.0:
            return True

        return rng.random() < probability

    def _sample_dark_clicks(
        self,
        *,
        time: int,
        window_duration_ticks: int,
        rng,
        dark_count_policy: OnArrivalWindowDarkCounts,
    ) -> tuple[int, ...]:
        require_non_negative_int(time, field_name="time")
        require_non_negative_int(
            window_duration_ticks,
            field_name="window_duration_ticks",
        )

        return dark_count_policy.sample_dark_clicks(
            time=time,
            process=DarkCountProcess(
                rate_hz=float(self.params.dark_count_rate_hz),
            ),
            rng=rng,
        )

    def _sample_afterpulse_time(
        self,
        *,
        time: int,
        window_duration_ticks: int,
        rngs: DetectorRNGStreams,
    ) -> int | None:
        time = require_non_negative_int(time, field_name="time")
        window_duration_ticks = require_non_negative_int(
            window_duration_ticks,
            field_name="window_duration_ticks",
        )

        if self.last_click_time is None:
            return None

        if window_duration_ticks <= 0:
            return None

        elapsed_ticks = time - self.last_click_time

        if elapsed_ticks < 0:
            return None

        afterpulse_probability = float(self.params.p_afterpulse)

        if not math.isfinite(afterpulse_probability):
            raise ValueError("p_afterpulse must be finite")

        if afterpulse_probability <= 0.0:
            return None

        if afterpulse_probability > 1.0:
            raise ValueError("p_afterpulse must be between 0 and 1")

        decay_ticks = float(self.params.afterpulse_decay_ticks)

        if not math.isfinite(decay_ticks) or decay_ticks <= 0.0:
            raise ValueError("afterpulse_decay_ticks must be finite and positive")

        afterpulse_rng = rngs.afterpulse

        if afterpulse_rng is None:
            raise ValueError("afterpulse RNG stream is required when p_afterpulse > 0")
        afterpulse_rng = cast(Any, afterpulse_rng)

        start_survival = math.exp(-elapsed_ticks / decay_ticks)

        interval_mass = start_survival * (
            -math.expm1(-window_duration_ticks / decay_ticks)
        )

        probability = afterpulse_probability * interval_mass
        probability = min(max(probability, 0.0), 1.0)

        if afterpulse_rng.random() >= probability:
            return None

        end_survival = math.exp(-(elapsed_ticks + window_duration_ticks) / decay_ticks)

        u = afterpulse_rng.random()

        survival = start_survival - u * (start_survival - end_survival)

        survival = min(max(survival, end_survival), start_survival)

        delay_ticks = int(round(-decay_ticks * math.log(survival)))

        sampled_time = self.last_click_time + delay_ticks
        window_end_exclusive = time + window_duration_ticks
        return min(sampled_time, window_end_exclusive - 1)

    def _apply_jitter(self, time: int, *, rng) -> int:
        stddev = float(self.params.jitter_stddev_ticks)

        if stddev == 0.0:
            return time

        jitter_ticks = max(0, int(round(rng.normal(loc=0.0, scale=stddev))))
        return time + jitter_ticks


__all__ = [
    "SinglePhotonDetector",
]
