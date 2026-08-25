"""Detector-window evaluation: exposure contract, gating, and click metadata.

``primitives/window.py`` had no test file of its own. Its helpers are reached
only through ``DetectorArray`` and ``BellStateAnalyzer``, so a change here
surfaced indirectly or not at all -- and the file sits on the path of every
detector click in the simulator.

What is tested is the helper's contract rather than detector physics:
``single_photon.py`` owns efficiency, dark counts, dead time, jitter and
afterpulsing, and ``test_single_photon.py`` covers them. This file covers the
layer that decides which detectors are evaluated, over which ticks, and what
each resulting click is labelled with.
"""

from __future__ import annotations

import pytest

from simyuj.components.detectors.primitives.gate import (
    AlwaysOpenGate,
    GateWindow,
    ScheduledGate,
)
from simyuj.components.detectors.primitives.params import SinglePhotonDetectorParams
from simyuj.components.detectors.primitives.readout import DetectorExposure
from simyuj.components.detectors.primitives.reports import (
    FLAG_DARK_COUNT,
    FLAG_SIGNAL_CLICK,
)
from simyuj.components.detectors.primitives.rng import DetectorRNGStreams
from simyuj.components.detectors.primitives.window import evaluate_detector_windows
from simyuj.components.detectors.single_photon import SinglePhotonDetector

DEVICE_ID = "rx"


class _ConstantRNG:
    """Returns a fixed value from every method a detector stream may call."""

    def __init__(self, *, random: float = 0.0, poisson: int = 0) -> None:
        self._random = random
        self._poisson = poisson

    def random(self) -> float:
        return self._random

    def normal(self, *, loc: float, scale: float) -> float:
        return 0.0

    def poisson(self, lam: float) -> int:
        return self._poisson

    def integers(self, low: int, high: int) -> int:
        return low


def _detectors(count: int = 2, **params) -> tuple[SinglePhotonDetector, ...]:
    resolved = SinglePhotonDetectorParams(**params)
    return tuple(
        SinglePhotonDetector(detector_id=f"d{index}", params=resolved)
        for index in range(count)
    )


def _rngs(
    detectors: tuple[SinglePhotonDetector, ...],
    *,
    poisson: int = 0,
) -> dict[str, DetectorRNGStreams]:
    return {
        detector.detector_id: DetectorRNGStreams(
            efficiency=_ConstantRNG(),
            dark=_ConstantRNG(poisson=poisson),
            jitter=_ConstantRNG(),
            afterpulse=_ConstantRNG(),
        )
        for detector in detectors
    }


def _evaluate(
    detectors: tuple[SinglePhotonDetector, ...],
    exposures: tuple[DetectorExposure, ...],
    *,
    time: int = 100,
    detection_window_ticks: int = 10,
    gate_model=None,
    measurement_label: str | None = None,
    detector_rngs: dict[str, DetectorRNGStreams] | None = None,
    click_meta: tuple[tuple[str, object], ...] = (),
):
    return evaluate_detector_windows(
        device_id=DEVICE_ID,
        time=time,
        detectors=detectors,
        exposures=exposures,
        detector_rngs=_rngs(detectors) if detector_rngs is None else detector_rngs,
        detection_window_ticks=detection_window_ticks,
        gate_model=AlwaysOpenGate() if gate_model is None else gate_model,
        measurement_label=measurement_label,
        fallback_complete_time=time,
        click_meta=click_meta,
    )


# --------------------------------------------------------------------------
# the exposure contract
# --------------------------------------------------------------------------


def test_exposures_must_match_the_detector_list_in_length_and_order() -> None:
    # Exposure order is the readout contract: window evaluation zips the two
    # lists, and a mismatch would apply one detector's exposure to another's
    # physics with nothing downstream able to recover from it.
    detectors = _detectors(2, efficiency=1.0)

    with pytest.raises(RuntimeError, match="exposure length must match"):
        _evaluate(detectors, (DetectorExposure(detector_id="d0"),))

    swapped = (
        DetectorExposure(detector_id="d1"),
        DetectorExposure(detector_id="d0"),
    )
    with pytest.raises(RuntimeError, match="exposure detector_id must match"):
        _evaluate(detectors, swapped)


def test_an_unexposed_detector_is_still_evaluated_for_dark_counts() -> None:
    # signal_present=False silences the signal candidate, not the detector. A
    # detector nobody pointed light at can still fire, and a receiver that
    # skipped it would report a dark-count rate of zero.
    detectors = _detectors(2, efficiency=1.0, dark_count_rate_hz=1e12)
    exposures = (
        DetectorExposure(detector_id="d0", signal_present=False, outcome_label="a"),
        DetectorExposure(detector_id="d1", signal_present=False, outcome_label="b"),
    )

    clicks, _complete = _evaluate(
        detectors,
        exposures,
        detector_rngs=_rngs(detectors, poisson=1),
    )

    assert {click.detector_id for click in clicks} == {"d0", "d1"}
    assert all(click.trigger == "dark" for click in clicks)
    assert all(click.flags == (FLAG_DARK_COUNT,) for click in clicks)
    # The logical label survives, so a dark count reports which outcome it faked.
    assert {click.outcome_label for click in clicks} == {"a", "b"}


def test_time_offset_shifts_one_exposure_without_moving_the_others() -> None:
    detectors = _detectors(2, efficiency=1.0)
    exposures = (
        DetectorExposure(detector_id="d0"),
        DetectorExposure(detector_id="d1", time_offset_ticks=7),
    )

    clicks, complete = _evaluate(detectors, exposures, time=100)

    by_id = {click.detector_id: click for click in clicks}
    assert by_id["d0"].time == 100
    assert by_id["d1"].time == 107
    # Completion follows the latest window end, which the offset extended.
    assert complete == 107 + 10


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------


def test_a_detector_outside_its_gate_is_skipped_and_leaves_completion_alone() -> None:
    detectors = _detectors(2, efficiency=1.0)
    exposures = (
        DetectorExposure(detector_id="d0"),
        DetectorExposure(detector_id="d1", time_offset_ticks=500),
    )
    gate = ScheduledGate(windows=(GateWindow(start=100, end=120),))

    clicks, complete = _evaluate(
        detectors,
        exposures,
        time=100,
        detection_window_ticks=10,
        gate_model=gate,
    )

    assert [click.detector_id for click in clicks] == ["d0"]
    assert complete == 110


def test_a_gate_closing_early_shortens_the_window_it_reports() -> None:
    detectors = _detectors(1, efficiency=1.0)
    exposures = (DetectorExposure(detector_id="d0"),)
    gate = ScheduledGate(windows=(GateWindow(start=100, end=104),))

    clicks, complete = _evaluate(
        detectors,
        exposures,
        time=100,
        detection_window_ticks=10,
        gate_model=gate,
    )

    assert complete == 104
    assert dict(clicks[0].meta)["window_duration_ticks"] == 4
    # The configured length is kept beside the clipped one, so a log reader can
    # see that the gate -- not the configuration -- decided this.
    assert dict(clicks[0].meta)["configured_detection_window_ticks"] == 10


def test_completion_falls_back_when_no_exposure_is_active() -> None:
    detectors = _detectors(1, efficiency=1.0)
    exposures = (DetectorExposure(detector_id="d0"),)
    gate = ScheduledGate(windows=(GateWindow(start=500, end=600),))

    clicks, complete = _evaluate(detectors, exposures, time=100, gate_model=gate)

    assert clicks == ()
    assert complete == 100


# --------------------------------------------------------------------------
# click metadata
# --------------------------------------------------------------------------


def test_click_meta_carries_the_readout_and_window_fields() -> None:
    detectors = _detectors(2, efficiency=1.0)
    exposures = (
        DetectorExposure(
            detector_id="d0",
            outcome_label="zero",
            time_offset_ticks=3,
            meta=(("exposure_key", "exposure_value"),),
        ),
        DetectorExposure(detector_id="d1", signal_present=False),
    )

    clicks, _complete = _evaluate(
        detectors,
        exposures,
        time=100,
        detection_window_ticks=10,
        measurement_label="Z",
        click_meta=(("component_key", "component_value"),),
    )

    meta = dict(clicks[0].meta)
    assert meta["device_id"] == DEVICE_ID
    assert meta["detector_index"] == 0
    assert meta["measurement_label"] == "Z"
    assert meta["readout_signal_present"] is True
    assert meta["readout_time_offset_ticks"] == 3
    assert meta["readout_outcome_label"] == "zero"
    assert meta["window_duration_ticks"] == 10
    assert meta["configured_detection_window_ticks"] == 10
    # Component metadata precedes exposure metadata, and both survive.
    assert meta["component_key"] == "component_value"
    assert meta["exposure_key"] == "exposure_value"
    assert clicks[0].flags == (FLAG_SIGNAL_CLICK,)


# --------------------------------------------------------------------------
# signal_click_probability (S7)
# --------------------------------------------------------------------------


def test_the_exposure_probability_reaches_the_detector_and_decides_the_click() -> None:
    # S7 is the whole of this helper's involvement: carry the number, do not
    # compute it, do not combine it with params.efficiency. A detector with
    # efficiency 0.0 that clicks because its exposure said 1.0 proves the value
    # arrived and replaced rather than multiplied.
    detectors = _detectors(2, efficiency=0.0)
    exposures = (
        DetectorExposure(detector_id="d0", signal_click_probability=1.0),
        DetectorExposure(detector_id="d1"),
    )

    clicks, _complete = _evaluate(detectors, exposures)

    assert [click.detector_id for click in clicks] == ["d0"]
    assert clicks[0].flags == (FLAG_SIGNAL_CLICK,)


def test_the_exposure_probability_is_recorded_in_the_click_meta() -> None:
    detectors = _detectors(2, efficiency=1.0)
    exposures = (
        DetectorExposure(detector_id="d0", signal_click_probability=1.0),
        DetectorExposure(detector_id="d1"),
    )

    clicks, _complete = _evaluate(detectors, exposures)

    by_id = {click.detector_id: dict(click.meta) for click in clicks}
    assert by_id["d0"]["readout_signal_click_probability"] == 1.0
    # None is recorded rather than omitted, so a trace distinguishes "the
    # detector used its own efficiency" from "this key predates the field".
    assert by_id["d1"]["readout_signal_click_probability"] is None
