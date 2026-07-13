from __future__ import annotations

import pytest

from simyuj.components.connections import PortDelivery, connect_ports
from simyuj.components.detectors import (
    ACTION_COINCIDENCE_TIMEOUT,
    ACTION_RUN_BELL_ANALYSIS,
    FLAG_DARK_COUNT,
    FLAG_NO_CLICK,
    FLAG_NO_OUTCOME,
    FLAG_SIGNAL_CLICK,
    FLAG_TIMEOUT,
    BellStateAnalyzer,
    BSMModel,
    ClickPattern,
    DetectionReport,
    Measure,
    SinglePhotonDetector,
    SinglePhotonDetectorParams,
)
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import StateNotFoundError, SubsystemId
from simyuj.qstate.measure import BellResult
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.binding import binding_context

ACTION_RECEIVE_REPORT = "receive_report"


class DummyComponent(Component):
    def handle_event(self, event, timeline) -> None:
        raise NotImplementedError


class ReportSink(Component):
    def __init__(self, device_id: str = "sink") -> None:
        self.device_id = device_id
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )
        self.received: list[DetectionReport] = []
        self.received_times: list[int] = []

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        assert event.action == ACTION_RECEIVE_REPORT
        assert isinstance(event.payload_ref, PortDelivery)

        payload = event.payload_ref.payload
        assert isinstance(payload, DetectionReport)

        self.received.append(payload)
        self.received_times.append(timeline.current_time)


def _source_port(owner_id: str = "src") -> Port:
    source = DummyComponent()
    return Port(
        name="out",
        owner=source,
        owner_id=owner_id,
        port_kind=PortKind.QUANTUM,
        direction=PortDirection.EGRESS,
    )


def _signal_for_target(
    *,
    signal_id: str,
    state_ref: int,
    subsystem: SubsystemId,
    pair_key: str | None = None,
) -> Signal:
    protocol_params: tuple[tuple[str, object], ...] = ()
    if pair_key is not None:
        protocol_params = (("bsa_pair_id", pair_key),)

    return Signal(
        id=signal_id,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="src",
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=str(subsystem),
                kind="qubit",
                index=0,
                metadata=(("qstate_subsystem", str(subsystem)),),
            ),
        ),
        protocol_params=protocol_params,
    )


def _delivery(
    *,
    analyzer: BellStateAnalyzer,
    side: str,
    signal: Signal,
    source_owner_id: str = "src",
) -> PortDelivery:
    target_port = (
        analyzer.left_input_port if side == "left" else analyzer.right_input_port
    )

    return PortDelivery(
        payload=signal,
        source_port=_source_port(source_owner_id),
        target_port=target_port,
        connection_id=f"{source_owner_id}.out->{analyzer.device_id}.{side}",
    )


def _event(
    *,
    analyzer: BellStateAnalyzer,
    side: str,
    signal: Signal,
    time: int,
    priority: int = 0,
) -> Event:
    return Event(
        time=time,
        priority=priority,
        target_ref=analyzer,
        action=ACTION_RUN_BELL_ANALYSIS,
        payload_ref=_delivery(
            analyzer=analyzer,
            side=side,
            signal=signal,
        ),
        source=None,
        subsystem_id="components",
    )


def _bell_signals(
    timeline: Timeline,
    *,
    state: str,
    left_id: str = "left",
    right_id: str = "right",
    pair_key: str | None = None,
) -> tuple[Signal, Signal]:
    q0 = SubsystemId(f"{left_id}-q")
    q1 = SubsystemId(f"{right_id}-q")
    state_ref = timeline.qstate.prepare(state, subsystems=(q0, q1))

    return (
        _signal_for_target(
            signal_id=left_id,
            state_ref=state_ref,
            subsystem=q0,
            pair_key=pair_key,
        ),
        _signal_for_target(
            signal_id=right_id,
            state_ref=state_ref,
            subsystem=q1,
            pair_key=pair_key,
        ),
    )


def _run_pair(
    *,
    timeline: Timeline,
    analyzer: BellStateAnalyzer,
    left_signal: Signal,
    right_signal: Signal,
    time: int = 10,
) -> None:
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=time,
            priority=0,
        )
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_signal,
            time=time,
            priority=1,
        )
    )


def _linear_bsm_detectors(
    *,
    params: SinglePhotonDetectorParams | None = None,
) -> tuple[SinglePhotonDetector, ...]:
    detector_params = (
        SinglePhotonDetectorParams(dark_count_rate_hz=0.0) if params is None else params
    )
    return tuple(
        SinglePhotonDetector(detector_id=detector_id, params=detector_params)
        for detector_id in ("d0", "d1", "d2", "d3")
    )


def _linear_bsm_patterns() -> tuple[ClickPattern, ...]:
    return (
        ClickPattern(outcome="psi+", detector_ids=("d0", "d1")),
        ClickPattern(outcome="psi-", detector_ids=("d2", "d3")),
    )


def _linear_bsm_analyzer(
    *,
    bsm_model: object = "linear_optical",
    detectors: tuple[SinglePhotonDetector, ...] | None = None,
    **kwargs,
) -> BellStateAnalyzer:
    return BellStateAnalyzer(
        device_id="bsa",
        bsm_model=bsm_model,
        detectors=_linear_bsm_detectors() if detectors is None else detectors,
        click_patterns=_linear_bsm_patterns(),
        coincidence_window_ticks=0,
        **kwargs,
    )


def _meta_value(report: DetectionReport, key: str) -> object:
    for meta_key, value in report.meta:
        if meta_key == key:
            return value
    raise AssertionError(f"missing report meta key: {key}")


def test_bell_state_analyzer_rejects_negative_window() -> None:
    with pytest.raises(ValueError, match="coincidence_window_ticks"):
        BellStateAnalyzer(
            device_id="bsa",
            coincidence_window_ticks=-1,
        )


def test_bell_state_analyzer_logs_ready_on_bind() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.INFO, sinks=[log_sink]))
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        bsm_model=BSMModel(kind="ideal", heralding_efficiency=0.75),
        coincidence_window_ticks=5,
        pairing_key="pair_id",
        output_latency_ticks=3,
        output_priority=4,
    )

    analyzer.bind(binding_context(timeline))

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.ready"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "device_id": "bsa",
        "measurement_label": "bell",
        "bsm_model": "ideal",
        "heralding_efficiency": 0.75,
        "detector_count": 0,
        "click_pattern_count": 0,
        "coincidence_window_ticks": 5,
        "detection_window_ticks": 1,
        "pairing_key": "pair_id",
        "output_latency_ticks": 3,
        "output_priority": 4,
    }


def test_bsm_model_accepts_string_spec() -> None:
    analyzer = _linear_bsm_analyzer(bsm_model="linear_optical")

    assert isinstance(analyzer.bsm_model, BSMModel)
    assert analyzer.bsm_model.kind == "linear_optical"


def test_linear_optical_bsm_requires_detectors() -> None:
    with pytest.raises(ValueError, match="requires detectors"):
        BellStateAnalyzer(
            device_id="bsa",
            bsm_model="linear_optical",
        )


def test_linear_optical_bsm_requires_explicit_patterns() -> None:
    with pytest.raises(ValueError, match="requires click_patterns"):
        BellStateAnalyzer(
            device_id="bsa",
            bsm_model="linear_optical",
            detectors=_linear_bsm_detectors(),
        )


def test_linear_optical_bsm_rejects_unknown_pattern_detector() -> None:
    with pytest.raises(ValueError, match="unknown click pattern detector_id"):
        BellStateAnalyzer(
            device_id="bsa",
            bsm_model="linear_optical",
            detectors=_linear_bsm_detectors(),
            click_patterns=(
                ClickPattern(outcome="psi+", detector_ids=("d0", "missing")),
                ClickPattern(outcome="psi-", detector_ids=("d2", "d3")),
            ),
        )


def test_linear_optical_bsm_rejects_ambiguous_click_patterns() -> None:
    with pytest.raises(ValueError, match="same detector pair"):
        BellStateAnalyzer(
            device_id="bsa",
            bsm_model="linear_optical",
            detectors=_linear_bsm_detectors(),
            click_patterns=(
                ClickPattern(outcome="psi+", detector_ids=("d0", "d1")),
                ClickPattern(outcome="psi-", detector_ids=("d1", "d0")),
            ),
        )


def test_linear_optical_bsm_rejects_duplicate_click_patterns() -> None:
    with pytest.raises(ValueError, match="duplicate outcome/detector pair"):
        BellStateAnalyzer(
            device_id="bsa",
            bsm_model="linear_optical",
            detectors=_linear_bsm_detectors(),
            click_patterns=(
                ClickPattern(outcome="psi+", detector_ids=("d0", "d1")),
                ClickPattern(outcome="psi+", detector_ids=("d1", "d0")),
                ClickPattern(outcome="psi-", detector_ids=("d2", "d3")),
            ),
        )


def test_bsm_model_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        BSMModel(kind="bad")  # type: ignore[arg-type]


def test_bsm_model_rejects_invalid_efficiency() -> None:
    with pytest.raises(ValueError, match="heralding_efficiency"):
        BSMModel(heralding_efficiency=1.1)


def test_ideal_model_reports_phi_plus_by_default() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(device_id="bsa", coincidence_window_ticks=0)
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="phi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.measurement_method == "bell"
    assert report.measurement_label == "bell"
    assert report.success is True
    assert report.raw_clicks == ()
    assert report.flags == ()
    assert isinstance(report.qstate_result, BellResult)
    assert report.outcome == "phi+"
    assert report.qstate_result.label == "phi+"
    assert report.qstate_result.outcome == (0, 0)
    assert _meta_value(report, "bsm_model") == "ideal"
    assert _meta_value(report, "readout_model") == "label"
    assert _meta_value(report, "reported_bell_label") == "phi+"


def test_bell_state_analyzer_logs_successful_analysis_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]))
    analyzer = BellStateAnalyzer(device_id="bsa", coincidence_window_ticks=2)
    analyzer.bind(binding_context(timeline))
    left_signal, right_signal = _bell_signals(
        timeline,
        state="psi-",
        pair_key="pair-1",
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=10,
        )
    )
    scheduled = timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_signal,
            time=11,
        )
    )

    timeline.run_until_empty()

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.analysis"
    )
    report = analyzer.reports[0]

    assert record.level is LogLevel.DEBUG
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_RUN_BELL_ANALYSIS
    assert dict(record.meta) == {
        "device_id": "bsa",
        "report_id": report.report_id,
        "measurement_label": "bell",
        "left_signal_id": "left",
        "right_signal_id": "right",
        "pair_key": "pair-1",
        "bsm_model": "ideal",
        "readout_model": "label",
        "success": True,
        "outcome": report.outcome,
        "true_bell_label": "psi-",
        "arrival_delta_ticks": 1,
        "click_count": 0,
        "flags": (),
    }


def test_bell_state_analyzer_requires_collapsing_bell_measurement() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        measurement=Measure.bell(
            targets=("left", "right"),
            collapse=False,
        ),
        coincidence_window_ticks=0,
    )
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="phi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    with pytest.raises(ValueError, match="requires collapse=True"):
        timeline.run_until_empty()


@pytest.mark.parametrize("label", ["phi+", "phi-", "psi+", "psi-"])
def test_ideal_bsm_reports_all_bell_labels(label: str) -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        bsm_model=BSMModel(kind="ideal", heralding_efficiency=1.0),
        coincidence_window_ticks=0,
    )
    analyzer.bind(binding_context(timeline))

    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = timeline.qstate.prepare(label, subsystems=(q0, q1))

    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
    )
    right_signal = _signal_for_target(
        signal_id="right",
        state_ref=state_ref,
        subsystem=q1,
    )

    timeline.schedule(
        _event(analyzer=analyzer, side="left", signal=left_signal, time=10)
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_signal,
            time=10,
            priority=1,
        )
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.success is True
    assert isinstance(report.qstate_result, BellResult)
    assert report.outcome == label
    assert report.qstate_result.label == label
    assert ("true_bell_label", label) in report.meta
    assert ("reported_bell_label", label) in report.meta


def test_linear_optical_model_does_not_report_phi_plus() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = _linear_bsm_analyzer()
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="phi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert report.flags == (FLAG_NO_CLICK, FLAG_NO_OUTCOME)
    assert isinstance(report.qstate_result, BellResult)
    assert report.qstate_result.label == "phi+"
    assert _meta_value(report, "bsm_model") == "linear_optical"
    assert _meta_value(report, "readout_model") == "linear_optical_click_patterns"
    assert _meta_value(report, "bsm_failure_reason") == "undetectable_bell_label"


@pytest.mark.parametrize(
    ("label", "expected_success"),
    [
        ("phi+", False),
        ("phi-", False),
        ("psi+", True),
        ("psi-", True),
    ],
)
def test_linear_optical_bsm_reports_only_psi_labels(
    label: str,
    expected_success: bool,
) -> None:
    timeline = Timeline(master_seed=123)
    analyzer = _linear_bsm_analyzer(
        bsm_model=BSMModel(
            kind="linear_optical",
            heralding_efficiency=1.0,
        ),
    )
    analyzer.bind(binding_context(timeline))

    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    state_ref = timeline.qstate.prepare(label, subsystems=(q0, q1))

    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
    )
    right_signal = _signal_for_target(
        signal_id="right",
        state_ref=state_ref,
        subsystem=q1,
    )

    timeline.schedule(
        _event(analyzer=analyzer, side="left", signal=left_signal, time=10)
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_signal,
            time=10,
            priority=1,
        )
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.success is expected_success
    assert isinstance(report.qstate_result, BellResult)
    assert report.outcome == (label if expected_success else None)
    assert report.qstate_result.label == label
    assert ("true_bell_label", label) in report.meta
    if expected_success:
        assert tuple(click.trigger for click in report.raw_clicks) == (
            "signal",
            "signal",
        )
    else:
        assert report.raw_clicks == ()


def test_linear_optical_model_reports_psi_plus() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = _linear_bsm_analyzer()
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="psi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    success_reports = [report for report in analyzer.reports if report.success]

    assert len(success_reports) == 1

    report = success_reports[0]
    assert tuple(click.detector_id for click in report.raw_clicks) == ("d0", "d1")
    assert tuple(click.trigger for click in report.raw_clicks) == ("signal", "signal")
    assert all(FLAG_SIGNAL_CLICK in click.flags for click in report.raw_clicks)
    assert isinstance(report.qstate_result, BellResult)
    assert report.outcome == "psi+"
    assert report.qstate_result.label == "psi+"


def _run_linear_optical_pattern_choice(seed: int) -> DetectionReport:
    timeline = Timeline(master_seed=seed)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        bsm_model=BSMModel(
            kind="linear_optical",
            heralding_efficiency=0.5,
        ),
        detectors=_linear_bsm_detectors(),
        click_patterns=(
            ClickPattern(outcome="psi+", detector_ids=("d0", "d1")),
            ClickPattern(outcome="psi+", detector_ids=("d0", "d2")),
            ClickPattern(outcome="psi-", detector_ids=("d2", "d3")),
        ),
        coincidence_window_ticks=0,
    )
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="psi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    return analyzer.reports[0]


def test_linear_optical_pattern_choice_replays_from_timeline_seed() -> None:
    first = _run_linear_optical_pattern_choice(seed=1)
    replay = _run_linear_optical_pattern_choice(seed=1)
    alternate_seed = _run_linear_optical_pattern_choice(seed=2)

    assert first.success is True
    assert replay.success is True
    assert alternate_seed.success is True
    assert tuple(click.detector_id for click in first.raw_clicks) == ("d0", "d2")
    assert tuple(click.detector_id for click in replay.raw_clicks) == ("d0", "d2")
    assert tuple(click.detector_id for click in alternate_seed.raw_clicks) == (
        "d0",
        "d1",
    )


def test_linear_optical_efficiency_zero_fails_but_keeps_qstate_result() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = _linear_bsm_analyzer(
        bsm_model=BSMModel(
            kind="linear_optical",
            heralding_efficiency=0.0,
        ),
    )
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="psi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert isinstance(report.qstate_result, BellResult)
    assert report.qstate_result.label == "psi+"
    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert report.flags == (FLAG_NO_CLICK, FLAG_NO_OUTCOME)
    assert _meta_value(report, "bsm_failure_reason") == "heralding_efficiency_miss"


def test_linear_optical_dark_counts_can_false_herald_psi_plus() -> None:
    timeline = Timeline(master_seed=12)
    params = SinglePhotonDetectorParams(
        efficiency=0.0,
        dark_count_rate_hz=1.0e12,
    )
    analyzer = _linear_bsm_analyzer(detectors=_linear_bsm_detectors(params=params))
    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="phi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    report = analyzer.reports[0]

    assert report.success is True
    assert report.outcome == "psi+"
    assert isinstance(report.qstate_result, BellResult)
    assert report.qstate_result.label == "phi+"
    assert tuple(click.detector_id for click in report.raw_clicks) == ("d0", "d1")
    assert tuple(click.trigger for click in report.raw_clicks) == ("dark", "dark")
    assert all(FLAG_DARK_COUNT in click.flags for click in report.raw_clicks)
    assert FLAG_DARK_COUNT in report.flags


def test_unmatched_input_times_out_unchanged() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        coincidence_window_ticks=5,
    )
    analyzer.bind(binding_context(timeline))

    q0 = SubsystemId("q0")
    state_ref = timeline.qstate.prepare("|0>", subsystems=(q0,))

    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
    )

    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=10,
        )
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert report.qstate_result is None
    assert report.flags == (FLAG_TIMEOUT,)
    assert report.signal_id == "left"
    assert _meta_value(report, "bsm_model") == "ideal"
    assert _meta_value(report, "readout_model") == "label"


def test_bell_state_analyzer_logs_buffered_input_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        coincidence_window_ticks=5,
    )
    analyzer.bind(binding_context(timeline))
    q0 = SubsystemId("q0")
    state_ref = timeline.qstate.prepare("|0>", subsystems=(q0,))
    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
        pair_key="pair-1",
    )

    scheduled = timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=10,
        )
    )
    timeline.run_until(10)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.buffer"
    )

    assert record.level is LogLevel.TRACE
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_RUN_BELL_ANALYSIS
    assert dict(record.meta) == {
        "device_id": "bsa",
        "side": "left",
        "signal_id": "left",
        "buffer_id": analyzer._buffers["left"][0].buffer_id,
        "pair_key": "pair-1",
        "coincidence_window_ticks": 5,
        "timeout_time": 16,
    }


def test_bell_state_analyzer_logs_timeout_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        coincidence_window_ticks=5,
    )
    analyzer.bind(binding_context(timeline))
    q0 = SubsystemId("q0")
    state_ref = timeline.qstate.prepare("|0>", subsystems=(q0,))
    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
        pair_key="pair-1",
    )

    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=10,
        )
    )
    timeline.run_until_empty()

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.timeout"
    )
    buffer_record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.buffer"
    )
    report = analyzer.reports[0]

    assert record.level is LogLevel.TRACE
    assert record.action == ACTION_COINCIDENCE_TIMEOUT
    assert dict(record.meta) == {
        "device_id": "bsa",
        "side": "left",
        "signal_id": "left",
        "buffer_id": dict(buffer_record.meta)["buffer_id"],
        "report_id": report.report_id,
        "measurement_label": "bell",
        "success": False,
        "pair_key": "pair-1",
        "arrival_time": 10,
        "timeout_time": 16,
        "coincidence_window_ticks": 5,
        "flags": (FLAG_TIMEOUT,),
    }


def test_bell_state_analyzer_timeout_uses_configured_measurement_label() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        measurement=Measure.bell(
            targets=("left", "right"),
            label="custom_bell",
        ),
        coincidence_window_ticks=5,
    )
    analyzer.bind(binding_context(timeline))
    q0 = SubsystemId("q0")
    state_ref = timeline.qstate.prepare("|0>", subsystems=(q0,))
    left_signal = _signal_for_target(
        signal_id="left",
        state_ref=state_ref,
        subsystem=q0,
    )

    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_signal,
            time=10,
        )
    )
    timeline.run_until_empty()

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.bell_state_analyzer.timeout"
    )
    report = analyzer.reports[0]

    assert report.measurement_label == "custom_bell"
    assert dict(record.meta)["measurement_label"] == "custom_bell"


def test_pair_key_matching_unchanged() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        coincidence_window_ticks=10,
        pairing_key="bsa_pair_id",
    )
    analyzer.bind(binding_context(timeline))

    a0 = SubsystemId("a0")
    a1 = SubsystemId("a1")
    b1 = SubsystemId("b1")

    state_ref_a = timeline.qstate.prepare("psi+", subsystems=(a0, a1))
    state_ref_b = timeline.qstate.prepare("|0>", subsystems=(b1,))

    left_a = _signal_for_target(
        signal_id="left-a",
        state_ref=state_ref_a,
        subsystem=a0,
        pair_key="round-a",
    )
    right_b = _signal_for_target(
        signal_id="right-b",
        state_ref=state_ref_b,
        subsystem=b1,
        pair_key="round-b",
    )
    right_a = _signal_for_target(
        signal_id="right-a",
        state_ref=state_ref_a,
        subsystem=a1,
        pair_key="round-a",
    )

    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="left",
            signal=left_a,
            time=10,
            priority=0,
        )
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_b,
            time=11,
            priority=0,
        )
    )
    timeline.schedule(
        _event(
            analyzer=analyzer,
            side="right",
            signal=right_a,
            time=12,
            priority=0,
        )
    )

    timeline.run_until_empty()

    success_reports = [report for report in analyzer.reports if report.success]

    assert len(success_reports) == 1
    assert success_reports[0].signal_id == ("left-a", "right-a")


def test_bell_state_analyzer_swaps_entanglement_across_state_records() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(device_id="bsa", coincidence_window_ticks=0)
    analyzer.bind(binding_context(timeline))

    a = SubsystemId("A")
    b = SubsystemId("B")
    c = SubsystemId("C")
    d = SubsystemId("D")

    state_ref_ab = timeline.qstate.prepare("phi+", subsystems=(a, b))
    state_ref_cd = timeline.qstate.prepare("phi+", subsystems=(c, d))

    assert timeline.qstate.state_of(a) == state_ref_ab
    assert timeline.qstate.state_of(b) == state_ref_ab
    assert timeline.qstate.state_of(c) == state_ref_cd
    assert timeline.qstate.state_of(d) == state_ref_cd

    left_signal = _signal_for_target(
        signal_id="B",
        state_ref=state_ref_ab,
        subsystem=b,
    )
    right_signal = _signal_for_target(
        signal_id="C",
        state_ref=state_ref_cd,
        subsystem=c,
    )

    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    assert len(analyzer.reports) == 1

    report = analyzer.reports[0]
    assert report.success is True
    assert isinstance(report.qstate_result, BellResult)
    assert _meta_value(report, "reported_bell_label") == report.qstate_result.label

    with pytest.raises(StateNotFoundError):
        timeline.qstate.state_of(b)
    with pytest.raises(StateNotFoundError):
        timeline.qstate.state_of(c)

    assert timeline.qstate.state_of(a) == timeline.qstate.state_of(d)

    swapped_result = timeline.qstate.measure_bell(
        targets=(a, d),
        collapse=False,
    )

    assert swapped_result.label == report.qstate_result.label
    assert swapped_result.outcome == report.qstate_result.outcome
    assert swapped_result.probability == 1.0


def test_output_latency_uses_measurement_time() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = BellStateAnalyzer(
        device_id="bsa",
        coincidence_window_ticks=0,
        output_latency_ticks=7,
    )
    sink = ReportSink()

    connect_ports(
        analyzer.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="psi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    success_reports = [report for report in analyzer.reports if report.success]

    assert len(success_reports) == 1
    assert sink.received == success_reports
    assert sink.received_times == [17]


def test_linear_optical_output_latency_uses_latest_raw_click_time() -> None:
    timeline = Timeline(master_seed=123)
    analyzer = _linear_bsm_analyzer(
        detection_window_ticks=5,
        output_latency_ticks=7,
    )
    sink = ReportSink()

    connect_ports(
        analyzer.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    analyzer.bind(binding_context(timeline))

    left_signal, right_signal = _bell_signals(timeline, state="psi+")
    _run_pair(
        timeline=timeline,
        analyzer=analyzer,
        left_signal=left_signal,
        right_signal=right_signal,
    )

    timeline.run_until_empty()

    success_reports = [report for report in analyzer.reports if report.success]

    assert len(success_reports) == 1
    assert tuple(click.time for click in success_reports[0].raw_clicks) == (10, 10)
    assert sink.received == success_reports
    assert sink.received_times == [17]
