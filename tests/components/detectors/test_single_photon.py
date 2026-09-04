from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from simyuj.components.detectors.primitives.dark_counts import OnArrivalWindowDarkCounts
from simyuj.components.detectors.primitives.params import SinglePhotonDetectorParams
from simyuj.components.detectors.primitives.reports import (
    FLAG_DARK_COUNT,
    FLAG_SIGNAL_CLICK,
)
from simyuj.components.detectors.primitives.rng import DetectorRNGStreams
from simyuj.components.detectors.single_photon import SinglePhotonDetector
from simyuj.engine.timeline import Timeline


@dataclass(slots=True)
class ScriptedRNG:
    random_values: list[float] = field(default_factory=list)
    normal_values: list[float] = field(default_factory=list)
    poisson_values: list[int] = field(default_factory=list)
    poisson_lams: list[float] = field(default_factory=list)

    def random(self) -> float:
        if self.random_values:
            return self.random_values.pop(0)
        return 0.0

    def normal(self, *, loc: float, scale: float) -> float:
        if self.normal_values:
            return self.normal_values.pop(0)
        return 0.0

    def poisson(self, lam: float) -> int:
        self.poisson_lams.append(lam)
        if self.poisson_values:
            return self.poisson_values.pop(0)
        return 0


class FakeRNG:
    def __init__(
        self,
        *,
        random_values: tuple[float, ...] = (),
        poisson_values: tuple[int, ...] = (),
        integer_values: tuple[int, ...] = (),
        normal_values: tuple[float, ...] = (),
    ) -> None:
        self._random_values = list(random_values)
        self._poisson_values = list(poisson_values)
        self._integer_values = list(integer_values)
        self._normal_values = list(normal_values)

    def random(self) -> float:
        if not self._random_values:
            raise AssertionError("unexpected random call")
        return self._random_values.pop(0)

    def poisson(self, lam: float) -> int:
        if not self._poisson_values:
            raise AssertionError("unexpected poisson call")
        return self._poisson_values.pop(0)

    def integers(self, low: int, high: int) -> int:
        if not self._integer_values:
            raise AssertionError("unexpected integers call")

        value = self._integer_values.pop(0)

        assert low <= value < high

        return value

    def normal(self, *, loc: float, scale: float) -> float:
        if not self._normal_values:
            raise AssertionError("unexpected normal call")
        return self._normal_values.pop(0)


def _streams(
    *,
    efficiency: ScriptedRNG | object | None = None,
    dark: ScriptedRNG | object | None = None,
    jitter: ScriptedRNG | object | None = None,
    afterpulse: ScriptedRNG | object | None = None,
) -> DetectorRNGStreams:
    return DetectorRNGStreams(
        efficiency=ScriptedRNG() if efficiency is None else efficiency,
        dark=ScriptedRNG() if dark is None else dark,
        jitter=ScriptedRNG() if jitter is None else jitter,
        afterpulse=afterpulse,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"efficiency": -0.1},
        {"efficiency": 1.1},
        {"dark_count_rate_hz": -1.0},
        {"dead_time_ticks": -1},
        {"p_afterpulse": 2.0},
        {"afterpulse_decay_ticks": 0.0},
    ],
)
def test_single_photon_detector_params_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SinglePhotonDetectorParams(**kwargs)


def test_efficiency_one_signal_present_produces_signal_click() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0),
    )

    clicks = detector.evaluate_window(
        time=10,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(),
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "signal"
    assert clicks[0].time == 10
    assert clicks[0].flags == (FLAG_SIGNAL_CLICK,)


def test_efficiency_zero_signal_present_produces_no_signal_click() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=0.0),
    )

    assert (
        detector.evaluate_window(
            time=10,
            signal_present=True,
            window_duration_ticks=1,
            rngs=_streams(),
        )
        == ()
    )


def test_signal_absent_never_produces_signal_click() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0),
    )

    assert (
        detector.evaluate_window(
            time=10,
            signal_present=False,
            window_duration_ticks=1,
            rngs=_streams(),
        )
        == ()
    )


def test_dead_time_blocks_until_dead_until_boundary() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0, dead_time_ticks=5),
    )

    assert detector.evaluate_window(
        time=10,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(),
    )
    for time in (11, 12, 13, 14):
        assert (
            detector.evaluate_window(
                time=time,
                signal_present=True,
                window_duration_ticks=1,
                rngs=_streams(),
            )
            == ()
        )

    assert detector.evaluate_window(
        time=15,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(),
    )


def test_earliest_candidate_sets_dead_time_and_blocks_later_candidates() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=1.0,
            dark_count_rate_hz=1.0e12,
            dead_time_ticks=5,
            jitter_stddev_ticks=1.0,
        ),
    )
    clicks = detector.evaluate_window(
        time=10,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(
            dark=ScriptedRNG(poisson_values=[1]),
            jitter=ScriptedRNG(normal_values=[3.0, 0.0]),
        ),
    )

    assert clicks[0].trigger == "dark"
    assert clicks[0].time == 10
    assert clicks[0].flags == (FLAG_DARK_COUNT,)
    assert detector.dead_until == 15


def test_same_tick_candidates_use_signal_dark_afterpulse_tie_order() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=1.0,
            dark_count_rate_hz=1.0e12,
            p_afterpulse=1.0,
            afterpulse_decay_ticks=100.0,
        ),
    )
    detector.last_click_time = 10

    clicks = detector.evaluate_window(
        time=10,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(
            dark=ScriptedRNG(poisson_values=[1]),
            afterpulse=ScriptedRNG(random_values=[0.0, 0.0]),
        ),
    )

    assert clicks[0].trigger == "signal"


def test_afterpulse_probability_uses_elapsed_window() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            p_afterpulse=1.0,
            afterpulse_decay_ticks=10.0,
        ),
    )
    detector.last_click_time = 100

    clicks = detector.evaluate_window(
        time=100,
        signal_present=False,
        window_duration_ticks=10,
        rngs=_streams(afterpulse=ScriptedRNG(random_values=[0.5, 0.0])),
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "afterpulse"
    assert clicks[0].time == 100


def test_afterpulse_time_is_sampled_inside_window() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            p_afterpulse=1.0,
            afterpulse_decay_ticks=100.0,
        ),
    )
    detector.last_click_time = 1000

    clicks = detector.evaluate_window(
        time=1030,
        signal_present=False,
        window_duration_ticks=20,
        rngs=_streams(afterpulse=ScriptedRNG(random_values=[0.0, 0.5])),
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "afterpulse"
    assert 1030 <= clicks[0].time < 1050


def test_afterpulse_probability_decays_with_elapsed_time() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            p_afterpulse=1.0,
            afterpulse_decay_ticks=10.0,
        ),
    )
    detector.last_click_time = 100

    clicks = detector.evaluate_window(
        time=200,
        signal_present=False,
        window_duration_ticks=10,
        rngs=_streams(afterpulse=ScriptedRNG(random_values=[0.5])),
    )

    assert clicks == ()


def test_afterpulse_requires_rng_stream_when_possible() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            p_afterpulse=0.5,
            afterpulse_decay_ticks=10.0,
        ),
    )
    detector.last_click_time = 100

    with pytest.raises(ValueError, match="afterpulse RNG"):
        detector.evaluate_window(
            time=100,
            signal_present=False,
            window_duration_ticks=10,
            rngs=_streams(),
        )


def test_zero_jitter_leaves_click_time_unchanged() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0, jitter_stddev_ticks=0.0),
    )

    clicks = detector.evaluate_window(
        time=10,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(jitter=ScriptedRNG(normal_values=[99.0])),
    )

    assert clicks[0].time == 10


def test_nonzero_jitter_never_creates_negative_click_time() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0, jitter_stddev_ticks=5.0),
    )

    clicks = detector.evaluate_window(
        time=0,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(jitter=ScriptedRNG(normal_values=[-100.0])),
    )

    assert clicks[0].time == 0


def test_signal_jitter_outside_window_discards_click() -> None:
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=1.0, jitter_stddev_ticks=1.0),
    )

    clicks = detector.evaluate_window(
        time=105,
        signal_present=True,
        window_duration_ticks=5,
        rngs=_streams(jitter=ScriptedRNG(normal_values=[7.0])),
    )

    assert clicks == ()


def test_same_named_rng_streams_reproduce_click_sequence() -> None:
    def run_once() -> tuple:
        timeline = Timeline(master_seed=1234)
        detector = SinglePhotonDetector(
            detector_id="det-0",
            params=SinglePhotonDetectorParams(
                efficiency=0.5,
                dark_count_rate_hz=1.0e12,
                jitter_stddev_ticks=1.0,
                p_afterpulse=0.25,
            ),
        )
        rngs = DetectorRNGStreams(
            efficiency=timeline.rng("det", "det-0", "efficiency"),
            dark=timeline.rng("det", "det-0", "dark"),
            jitter=timeline.rng("det", "det-0", "jitter"),
            afterpulse=timeline.rng("det", "det-0", "afterpulse"),
        )
        return (
            detector.evaluate_window(
                time=10,
                signal_present=True,
                window_duration_ticks=2,
                rngs=rngs,
            ),
            detector.evaluate_window(
                time=20,
                signal_present=True,
                window_duration_ticks=2,
                rngs=rngs,
            ),
        )

    assert run_once() == run_once()


def test_changing_efficiency_stream_does_not_change_dark_count_stream() -> None:
    def run_once(efficiency_stream: str) -> tuple:
        timeline = Timeline(master_seed=99)
        detector = SinglePhotonDetector(
            detector_id="det-0",
            params=SinglePhotonDetectorParams(
                efficiency=0.0,
                dark_count_rate_hz=1.0e18,
            ),
        )
        return detector.evaluate_window(
            time=10,
            signal_present=True,
            window_duration_ticks=1,
            rngs=DetectorRNGStreams(
                efficiency=timeline.rng("det", "det-0", efficiency_stream),
                dark=timeline.rng("det", "det-0", "dark"),
                jitter=timeline.rng("det", "det-0", "jitter"),
            ),
        )

    assert run_once("efficiency-a") == run_once("efficiency-b")


def test_single_photon_detector_default_dark_count_is_coarse() -> None:
    detector = SinglePhotonDetector(
        detector_id="d0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            dark_count_rate_hz=1e12,
            jitter_stddev_ticks=0.0,
        ),
    )

    rngs = DetectorRNGStreams(
        efficiency=FakeRNG(),
        dark=FakeRNG(poisson_values=(1,)),
        jitter=FakeRNG(),
        afterpulse=None,
    )

    clicks = detector.evaluate_window(
        time=100,
        signal_present=False,
        window_duration_ticks=50,
        rngs=rngs,
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "dark"
    assert clicks[0].time == 100


def test_single_photon_detector_preserves_time_resolved_dark_count() -> None:
    detector = SinglePhotonDetector(
        detector_id="d0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            dark_count_rate_hz=1e12,
            jitter_stddev_ticks=0.0,
        ),
    )

    rngs = DetectorRNGStreams(
        efficiency=FakeRNG(),
        dark=FakeRNG(
            poisson_values=(1,),
            integer_values=(17,),
        ),
        jitter=FakeRNG(),
        afterpulse=None,
    )

    clicks = detector.evaluate_window(
        time=100,
        signal_present=False,
        window_duration_ticks=50,
        rngs=rngs,
        dark_count_policy=OnArrivalWindowDarkCounts(
            window_duration_ticks=50,
            time_resolved=True,
        ),
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "dark"
    assert clicks[0].time == 117


def test_single_photon_detector_threshold_mode_returns_first_dark_click_only() -> None:
    detector = SinglePhotonDetector(
        detector_id="d0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            dark_count_rate_hz=1e12,
            jitter_stddev_ticks=0.0,
            photon_number_resolving=False,
        ),
    )

    rngs = DetectorRNGStreams(
        efficiency=FakeRNG(),
        dark=FakeRNG(
            poisson_values=(3,),
            integer_values=(40, 7, 25),
        ),
        jitter=FakeRNG(),
        afterpulse=None,
    )

    clicks = detector.evaluate_window(
        time=100,
        signal_present=False,
        window_duration_ticks=50,
        rngs=rngs,
        dark_count_policy=OnArrivalWindowDarkCounts(
            window_duration_ticks=50,
            time_resolved=True,
            return_all_clicks=True,
        ),
    )

    assert len(clicks) == 1
    assert clicks[0].trigger == "dark"
    assert clicks[0].time == 107


def test_single_photon_detector_pnr_mode_can_return_multiple_dark_clicks() -> None:
    detector = SinglePhotonDetector(
        detector_id="d0",
        params=SinglePhotonDetectorParams(
            efficiency=0.0,
            dark_count_rate_hz=1e12,
            dead_time_ticks=0,
            jitter_stddev_ticks=0.0,
            photon_number_resolving=True,
        ),
    )

    rngs = DetectorRNGStreams(
        efficiency=FakeRNG(),
        dark=FakeRNG(
            poisson_values=(3,),
            integer_values=(40, 7, 25),
        ),
        jitter=FakeRNG(),
        afterpulse=None,
    )

    clicks = detector.evaluate_window(
        time=100,
        signal_present=False,
        window_duration_ticks=50,
        rngs=rngs,
        dark_count_policy=OnArrivalWindowDarkCounts(
            window_duration_ticks=50,
            time_resolved=True,
            return_all_clicks=True,
        ),
    )

    assert tuple(click.time for click in clicks) == (107, 125, 140)
    assert tuple(click.trigger for click in clicks) == ("dark", "dark", "dark")


def test_single_photon_detector_rejects_mismatched_dark_policy_window() -> None:
    detector = SinglePhotonDetector(
        detector_id="d0",
        params=SinglePhotonDetectorParams(),
    )

    rngs = DetectorRNGStreams(
        efficiency=FakeRNG(),
        dark=FakeRNG(),
        jitter=FakeRNG(),
        afterpulse=None,
    )

    try:
        detector.evaluate_window(
            time=100,
            signal_present=False,
            window_duration_ticks=50,
            rngs=rngs,
            dark_count_policy=OnArrivalWindowDarkCounts(
                window_duration_ticks=20,
                time_resolved=True,
            ),
        )
    except ValueError as exc:
        assert "window_duration_ticks" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------
# signal_click_probability: override, never multiply
# --------------------------------------------------------------------------


def _click_count(
    *,
    efficiency: float,
    signal_click_probability: float | None,
    trials: int,
    seed: int = 4242,
) -> int:
    """Run `trials` independent windows and count the signal clicks.

    A fresh detector per trial, because dead time and afterpulsing are state and
    this is a question about one Bernoulli draw. Draws come from a real
    `Timeline` stream, so what is exercised is the production RNG rather than a
    scripted one that could be made to say anything.
    """
    timeline = Timeline(master_seed=seed)
    rng = timeline.rng("probe", "single_photon", "det-0", "efficiency")
    params = SinglePhotonDetectorParams(
        efficiency=efficiency,
        dark_count_rate_hz=0.0,
    )

    clicked = 0
    for trial in range(trials):
        detector = SinglePhotonDetector(detector_id="det-0", params=params)
        clicks = detector.evaluate_window(
            time=trial,
            signal_present=True,
            window_duration_ticks=1,
            rngs=_streams(efficiency=rng),
            signal_click_probability=signal_click_probability,
        )
        clicked += len(clicks)

    return clicked


def test_signal_click_probability_overrides_efficiency_and_never_multiplies() -> None:
    # The S8 trap, and the only assertion in the suite that catches it.
    #
    # `click_probability` returns 1 - exp(-eta_d * mu) with the detector's own
    # eta_d ALREADY inside the exponent. A `_sample_signal_click` that multiplies
    # by params.efficiency instead of replacing it applies eta_d twice: at
    # eta_d = 0.2 a run yields a fifth of the clicks it should, nothing raises,
    # and every rate is uniformly low -- indistinguishable from a lossier link,
    # which is the quantity a QKD run is trying to measure.
    #
    # efficiency 0.5 with probability 1.0 must click on EVERY trial. Under the
    # multiply bug it clicks about half the time.
    trials = 400

    assert (
        _click_count(
            efficiency=0.5,
            signal_click_probability=1.0,
            trials=trials,
        )
        == trials
    )

    # The converse, so neither value can be the one being read by accident: if
    # the code read params.efficiency and ignored the argument, this would click
    # on every trial instead of about half.
    half = _click_count(
        efficiency=1.0,
        signal_click_probability=0.5,
        trials=trials,
    )
    assert 0 < half < trials
    assert half == pytest.approx(trials * 0.5, rel=0.15)


def test_signal_click_probability_zero_beats_a_perfect_efficiency() -> None:
    # The other direction of the same override, at the branch that short-circuits
    # before touching the RNG.
    assert _click_count(efficiency=1.0, signal_click_probability=0.0, trials=50) == 0


def test_signal_click_probability_of_none_keeps_params_efficiency() -> None:
    # Regression guard on the default path: every existing caller passes nothing,
    # and must keep sampling against params.efficiency exactly as before.
    assert _click_count(efficiency=1.0, signal_click_probability=None, trials=50) == 50
    assert _click_count(efficiency=0.0, signal_click_probability=None, trials=50) == 0

    partial = _click_count(efficiency=0.5, signal_click_probability=None, trials=400)
    assert partial == pytest.approx(400 * 0.5, rel=0.15)


def test_a_supplied_probability_consumes_the_stream_like_an_efficiency() -> None:
    # Draw-count parity is what makes S8 safe to land: the branch structure is
    # identical either way, so exactly one random() is consumed when
    # 0 < p < 1 and none at the two endpoints. A supplied probability therefore
    # cannot shift a stream position that an efficiency would not have shifted.
    detector = SinglePhotonDetector(
        detector_id="det-0",
        params=SinglePhotonDetectorParams(efficiency=0.5, dark_count_rate_hz=0.0),
    )

    # 0 < p < 1: one draw, and the scripted value is the one compared.
    rng = ScriptedRNG(random_values=[0.9])
    detector.evaluate_window(
        time=0,
        signal_present=True,
        window_duration_ticks=1,
        rngs=_streams(efficiency=rng),
        signal_click_probability=0.25,
    )
    assert rng.random_values == []

    # p == 1.0 and p == 0.0 short-circuit: FakeRNG raises on an unexpected call.
    for probability in (0.0, 1.0):
        detector = SinglePhotonDetector(
            detector_id="det-0",
            params=SinglePhotonDetectorParams(efficiency=0.5, dark_count_rate_hz=0.0),
        )
        detector.evaluate_window(
            time=0,
            signal_present=True,
            window_duration_ticks=1,
            rngs=_streams(efficiency=FakeRNG()),
            signal_click_probability=probability,
        )
