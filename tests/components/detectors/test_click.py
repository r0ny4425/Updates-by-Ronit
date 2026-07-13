from __future__ import annotations

from simyuj.components.detectors.primitives.click import (
    POVMLabelClickResolver,
    ThresholdClickResolver,
)
from simyuj.components.detectors.primitives.measurement import (
    Measure,
    MeasurementContext,
)
from simyuj.components.detectors.primitives.reports import (
    FLAG_DOUBLE_CLICK,
    FLAG_NO_CLICK,
    RawClick,
)
from simyuj.qstate import SubsystemId
from simyuj.signal import EncodingScheme, Signal, SignalKind


class FakeRNG:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class FakeResult:
    label = "zero"


def _signal(signal_id: str) -> Signal:
    return Signal(
        id=signal_id,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="src",
    )


def test_threshold_no_click_fails() -> None:
    resolver = ThresholdClickResolver()
    report = resolver.resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=None,
        raw_clicks=(),
    )

    assert report.success is False
    assert report.outcome is None
    assert FLAG_NO_CLICK in report.flags
    assert report.raw_clicks == ()


def test_threshold_single_click_succeeds() -> None:
    resolver = ThresholdClickResolver()
    click = RawClick(
        detector_id="d0",
        time=10,
        trigger="signal",
        outcome_label="0",
    )

    report = resolver.resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=None,
        raw_clicks=(click,),
    )

    assert report.success is True
    assert report.outcome == "0"
    assert report.raw_clicks == (click,)


def test_threshold_double_click_fail_policy() -> None:
    resolver = ThresholdClickResolver(double_click_policy="fail")
    clicks = (
        RawClick(detector_id="d0", time=10, trigger="signal", outcome_label="0"),
        RawClick(detector_id="d1", time=10, trigger="signal", outcome_label="1"),
    )

    report = resolver.resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=None,
        raw_clicks=clicks,
    )

    assert report.success is False
    assert report.outcome is None
    assert FLAG_DOUBLE_CLICK in report.flags


def test_threshold_double_click_first_policy_uses_earliest_click() -> None:
    resolver = ThresholdClickResolver(double_click_policy="first")
    early = RawClick(detector_id="d0", time=10, trigger="signal", outcome_label="0")
    late = RawClick(detector_id="d1", time=11, trigger="signal", outcome_label="1")

    report = resolver.resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=None,
        raw_clicks=(late, early),
    )

    assert report.success is True
    assert report.outcome == "0"
    assert report.meta == (("selected_detector_id", "d0"),)


def test_threshold_double_click_random_policy() -> None:
    resolver = ThresholdClickResolver(double_click_policy="random")
    clicks = (
        RawClick(detector_id="d0", time=10, trigger="signal", outcome_label="0"),
        RawClick(detector_id="d1", time=10, trigger="signal", outcome_label="1"),
    )

    report = resolver.resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=None,
        raw_clicks=clicks,
        rng=FakeRNG(0.75),
    )

    assert report.success is True
    assert report.outcome == "1"
    assert FLAG_DOUBLE_CLICK in report.flags


def test_report_includes_measurement_metadata() -> None:
    q0 = SubsystemId("q0")
    context = MeasurementContext(
        device_id="det",
        time=10,
        signal=None,
        signal_targets=(q0,),
    )
    call = Measure.random({"z": 0.5, "x": 0.5}).choose(
        context,
        rng=FakeRNG(0.1),
    )

    click = RawClick(
        detector_id="d0",
        time=10,
        trigger="signal",
        outcome_label="0",
    )

    report = ThresholdClickResolver().resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=None,
        measurement_call=call,
        raw_clicks=(click,),
    )

    assert report.measurement_method == "projective"
    assert report.measurement_label == "z"
    assert report.selection_index == 0
    assert report.selection_probability == 0.5
    assert report.selection_label == "z"


def test_povm_label_resolver_uses_result_label() -> None:
    click = RawClick(
        detector_id="d0",
        time=10,
        trigger="signal",
        outcome_label="hardware-0",
    )

    report = POVMLabelClickResolver().resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=FakeResult(),
        measurement_call=None,
        raw_clicks=(click,),
    )

    assert report.success is True
    assert report.outcome == "zero"


def test_povm_label_resolver_requires_click_by_default() -> None:
    report = POVMLabelClickResolver().resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=FakeResult(),
        measurement_call=None,
        raw_clicks=(),
    )

    assert report.success is False
    assert FLAG_NO_CLICK in report.flags


def test_povm_label_resolver_can_use_result_without_click() -> None:
    report = POVMLabelClickResolver(require_click=False).resolve(
        device_id="det",
        time=10,
        signal=None,
        qstate_result=FakeResult(),
        measurement_call=None,
        raw_clicks=(),
    )

    assert report.success is True
    assert report.outcome == "zero"
    assert report.raw_clicks == ()


def test_report_id_includes_signal_id_to_avoid_same_tick_collisions() -> None:
    resolver = ThresholdClickResolver()

    first = resolver.resolve(
        device_id="det",
        time=10,
        signal=_signal("sig-a"),
        qstate_result=None,
        measurement_call=None,
        raw_clicks=(),
    )
    second = resolver.resolve(
        device_id="det",
        time=10,
        signal=_signal("sig-b"),
        qstate_result=None,
        measurement_call=None,
        raw_clicks=(),
    )

    assert first.report_id == "det:report:10:sig-a:no-click"
    assert second.report_id == "det:report:10:sig-b:no-click"
    assert first.report_id != second.report_id
