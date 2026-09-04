"""Optical detection inside the delay interferometer.

The optics themselves are tested in ``test_delay_interferometer.py``, at
``detectors=None``, where exact amplitudes and exact ticks can be asserted.
This file covers only what the two internal detectors add: turning each output
port's mean photon number into an independent click, and resolving the pair into
one slot decision.

The split matters. Behind a detector every claim in the optics file becomes a
statistical one needing a seed sweep, which is a strictly weaker test of the same
physics -- so the optics keep their exact assertions and detection is tested
against them, not instead of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isclose, pi

import pytest

from simyuj.components.connections import connect_ports
from simyuj.components.detectors import (
    FLAG_DOUBLE_CLICK,
    FLAG_NO_CLICK,
    DetectionReport,
    SinglePhotonDetector,
    SinglePhotonDetectorParams,
)
from simyuj.components.detectors.primitives.click import ThresholdClickResolver
from simyuj.components.detectors.primitives.gate import GateWindow, ScheduledGate
from simyuj.components.interferometers import (
    ACTION_INTERFERE,
    PORT_OUT_0,
    PORT_OUT_1,
    DelayInterferometer,
)
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.components.sources import (
    FixedCarrierPhase,
    FixedIntensity,
    PhaseSequence,
    WeakCoherentPulseSource,
)
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, Signal, SignalKind

from ..support.mock_components import ACTION_RECEIVE_SIGNAL, SignalSink

MU = 0.2
TAU_TICKS = 1000
SIGMA_S = 3e-11
CLOCK_HZ = 1e9
ETA = 0.6

# A mu this large saturates 1 - exp(-eta*mu) to exactly 1.0 in double
# precision, which is how a test asks for a certain click without reaching
# past the physics. eta = 1.0 does NOT mean "always clicks": at mu = 0.2 a
# perfect detector still sees nothing 82% of the time, because most pulses
# contain no photon at all. That is the model working, not a knob to bypass.
BRIGHT_MU = 50.0

ACTION_RECEIVE_REPORT = "receive_report"

_DEFAULT = object()


@dataclass(slots=True)
class _PulseDriver(Component):
    """Emits coherent pulses on demand at caller-chosen ticks."""

    device_id: str = "driver"

    output_port: Port = field(init=False)
    _count: int = field(default=0)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:  # pragma: no cover
        raise AssertionError("driver is not an event target")

    def emit(
        self,
        timeline: Timeline,
        *,
        at_tick: int,
        phase_rad: float = 0.0,
        mean_photon_number: float = MU,
    ) -> Signal:
        self._count += 1
        signal = Signal(
            id=f"{self.device_id}:pulse:{self._count}",
            signal_kind=SignalKind.PULSE,
            encoding_scheme=EncodingScheme.PHASE,
            emission_time=at_tick,
            origin=self.device_id,
            coherent_state=CoherentState.from_mean_photon_number(
                mean_photon_number,
                phase_rad=phase_rad,
            ),
            temporal_mode_sigma_s=SIGMA_S,
            meta=(("pulse_index", self._count),),
        )
        assert self.output_port.connection is not None
        self.output_port.connection.transmit(
            signal,
            timeline,
            time=at_tick,
            priority=0,
            source=self,
        )
        return signal


@dataclass(slots=True)
class _ReportSink(Component):
    """Terminating consumer for the detection port."""

    device_id: str = "report_sink"

    input_port: Port = field(init=False)
    received: list[tuple[int, DetectionReport]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        assert event.action == ACTION_RECEIVE_REPORT
        payload = event.payload_ref.payload
        assert isinstance(payload, DetectionReport)
        self.received.append((timeline.current_time, payload))


def _detectors(
    *,
    efficiency: float = ETA,
    count: int = 2,
    dark_count_rate_hz: float = 0.0,
    **params,
) -> tuple[SinglePhotonDetector, ...]:
    resolved = SinglePhotonDetectorParams(
        efficiency=efficiency,
        dark_count_rate_hz=dark_count_rate_hz,
        **params,
    )
    return tuple(
        SinglePhotonDetector(detector_id=f"bob_d{index}", params=resolved)
        for index in range(count)
    )


def _build(
    *,
    detectors=_DEFAULT,
    seed: int = 1,
    connect_outputs: bool = True,
    connect_reports: bool = False,
    **kwargs,
):
    """Wire driver -> interferometer -> optical sinks (and optionally reports)."""
    timeline = Timeline(master_seed=seed)
    driver = _PulseDriver()
    device = DelayInterferometer(
        device_id="bob_di",
        detectors=_detectors() if detectors is _DEFAULT else detectors,
        delay_ticks=kwargs.pop("delay_ticks", TAU_TICKS),
        **kwargs,
    )
    sink_0 = SignalSink(device_id="out0_sink")
    sink_1 = SignalSink(device_id="out1_sink")
    reports = _ReportSink()

    connect_ports(
        driver.output_port,
        device.input_port,
        target_action=ACTION_INTERFERE,
    )
    if connect_outputs:
        connect_ports(
            device.output_port_0,
            sink_0.input_port,
            target_action=ACTION_RECEIVE_SIGNAL,
        )
        connect_ports(
            device.output_port_1,
            sink_1.input_port,
            target_action=ACTION_RECEIVE_SIGNAL,
        )
    if connect_reports:
        connect_ports(
            device.detection_port,
            reports.input_port,
            target_action=ACTION_RECEIVE_REPORT,
        )

    device.bind(BindingContext(timeline=timeline, logger=timeline.logger))
    return timeline, driver, device, reports


def _run_train(
    phases,
    *,
    num_slots: int = 5,
    seed: int = 1,
    detectors=_DEFAULT,
    **kwargs,
):
    """Run a real source through a detecting interferometer."""
    timeline = Timeline(master_seed=seed)
    source = WeakCoherentPulseSource(
        device_id="alice",
        frequency_hz=CLOCK_HZ,
        intensity=FixedIntensity(MU),
        duration_s=num_slots / CLOCK_HZ,
        carrier_phase=FixedCarrierPhase(0.0),
        encoding_phase=PhaseSequence(phases, repeat=True),
        temporal_mode_sigma_s=SIGMA_S,
    )
    device = DelayInterferometer(
        device_id="bob_di",
        detectors=_detectors() if detectors is _DEFAULT else detectors,
        delay_ticks=TAU_TICKS,
        **kwargs,
    )
    sink_0 = SignalSink(device_id="out0_sink")
    sink_1 = SignalSink(device_id="out1_sink")

    connect_ports(source.output_port, device.input_port, target_action=ACTION_INTERFERE)
    connect_ports(
        device.output_port_0,
        sink_0.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
    )
    connect_ports(
        device.output_port_1,
        sink_1.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
    )

    device.bind(BindingContext(timeline=timeline, logger=timeline.logger))
    source.schedule_start(timeline)
    timeline.run_until_empty()
    return timeline, source, device, sink_0, sink_1


def _emit_train(
    timeline,
    driver,
    device,
    *,
    count: int,
    phases=(0.0,),
    mean_photon_number: float = MU,
):
    for index in range(count):
        driver.emit(
            timeline,
            at_tick=index * TAU_TICKS,
            phase_rad=phases[index % len(phases)],
            mean_photon_number=mean_photon_number,
        )
    timeline.run_until_empty()
    return device.detection_reports


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_detection_needs_exactly_one_detector_per_output_port() -> None:
    # BS2 has two ports and both always carry light -- the destructive one
    # carrying nearly nothing is a result, not an absence. One detector could
    # only ever read half the slot.
    for count in (1, 3):
        with pytest.raises(ValueError, match="exactly two detectors"):
            DelayInterferometer(
                device_id="bob_di",
                detectors=_detectors(count=count),
                delay_ticks=TAU_TICKS,
            )

    with pytest.raises(ValueError, match="detector_id values must be unique"):
        DelayInterferometer(
            device_id="bob_di",
            detectors=(
                SinglePhotonDetector(detector_id="same"),
                SinglePhotonDetector(detector_id="same"),
            ),
            delay_ticks=TAU_TICKS,
        )


# --------------------------------------------------------------------------
# the physics: two independent ports
# --------------------------------------------------------------------------


def test_each_port_clicks_at_its_own_one_minus_exp_minus_eta_mu() -> None:
    # The claim the whole component rests on: port k's rate is set by port k's
    # own mu, not by the slot's total. Swept over seeds because a single run of
    # a Bernoulli trial asserts nothing.
    #
    # Equal phases put the light on port 1, so mu_1 = 0.2 and mu_0 ~ 0.
    port_0 = 0
    port_1 = 0
    slots = 0

    # 30 seeds x 100 slots ~= 3000 trials, where one relative standard
    # deviation on p = 0.113 is about 5%. A tolerance tighter than the sampling
    # noise would be a coin flip dressed as an assertion.
    for seed in range(30):
        _timeline, _source, device, _s0, _s1 = _run_train(
            (0.0,),
            num_slots=100,
            seed=seed,
        )
        for report in device.detection_reports:
            meta = dict(report.meta)
            if meta["short_pulse_index"] is None or meta["long_pulse_index"] is None:
                continue  # a vacuum slot: both ports equal, no bit
            slots += 1
            clicked = {click.detector_id for click in report.raw_clicks}
            port_0 += "bob_d0" in clicked
            port_1 += "bob_d1" in clicked

    expected_1 = 1.0 - exp(-ETA * MU)
    assert port_1 / slots == pytest.approx(expected_1, rel=0.15)
    # Port 0 is the dark port at gamma = 1: mu ~ 1e-33, so it never fires.
    assert port_0 == 0


def test_a_double_click_comes_from_two_signal_clicks() -> None:
    # The difference from DetectorArray, stated as a test. There a readout maps
    # one measured outcome to one exposed detector, so two *signal* clicks are
    # structurally impossible and the double-click rate is identically zero at
    # every mu. Here both ports are exposed every slot.
    #
    # Driven to certainty rather than sampled: efficiency 1.0 and a pi step put
    # real light on both ports, so both must fire in the same slot.
    timeline, driver, device, _reports = _build()
    driver.emit(timeline, at_tick=0, phase_rad=0.0, mean_photon_number=BRIGHT_MU)
    driver.emit(
        timeline,
        at_tick=TAU_TICKS,
        phase_rad=pi / 2,
        mean_photon_number=BRIGHT_MU,
    )
    timeline.run_until_empty()

    paired = [
        report
        for report in device.detection_reports
        if dict(report.meta)["long_pulse_index"] is not None
        and dict(report.meta)["short_pulse_index"] is not None
    ]
    assert len(paired) == 1

    report = paired[0]
    meta = dict(report.meta)
    # A quadrature step splits the light evenly, so neither port is dark and
    # both probabilities saturate: the double click is certain, not sampled.
    assert meta["mean_photon_number_0"] == pytest.approx(BRIGHT_MU / 2, rel=1e-9)
    assert meta["mean_photon_number_1"] == pytest.approx(BRIGHT_MU / 2, rel=1e-9)
    assert {click.detector_id for click in report.raw_clicks} == {"bob_d0", "bob_d1"}
    assert all(click.trigger == "signal" for click in report.raw_clicks)
    assert FLAG_DOUBLE_CLICK in report.flags


def test_the_dark_port_produces_no_click_of_its_own() -> None:
    # A perfect dark port delivers mu ~ 1e-33 -- the exp(1j*pi) residue, squared
    # -- not 0.0. The closed form has to put that at the floor without a special
    # case, or an ideal interferometer would click on both ports every slot.
    timeline, driver, device, _reports = _build()
    _emit_train(timeline, driver, device, count=4, mean_photon_number=BRIGHT_MU)

    for report in device.detection_reports:
        meta = dict(report.meta)
        if meta["short_pulse_index"] is None or meta["long_pulse_index"] is None:
            continue
        # The bright port saturates, so a click there is certain; anything on
        # port 0 could only have come from the residue.
        assert meta["mean_photon_number_1"] == pytest.approx(BRIGHT_MU, rel=1e-9)
        assert meta["mean_photon_number_0"] < 1e-28
        assert {click.detector_id for click in report.raw_clicks} == {"bob_d1"}


# --------------------------------------------------------------------------
# resolving two clicks into one slot
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy, success",
    [("fail", False), ("first", True), ("random", True)],
)
def test_the_double_click_policy_decides_what_the_slot_reports(policy, success) -> None:
    # The rate is physics and belongs to the ports; the response is protocol and
    # belongs to the resolver. This is the seam between them.
    timeline, driver, device, _reports = _build(
        click_resolver=ThresholdClickResolver(double_click_policy=policy),
    )
    driver.emit(timeline, at_tick=0, phase_rad=0.0, mean_photon_number=BRIGHT_MU)
    driver.emit(
        timeline,
        at_tick=TAU_TICKS,
        phase_rad=pi / 2,
        mean_photon_number=BRIGHT_MU,
    )
    timeline.run_until_empty()

    report = next(
        report for report in device.detection_reports if len(report.raw_clicks) == 2
    )
    assert report.success is success
    assert FLAG_DOUBLE_CLICK in report.flags
    if success:
        assert report.outcome in (PORT_OUT_0, PORT_OUT_1)
    else:
        assert report.outcome is None


def test_a_slot_with_no_click_still_reports_and_still_carries_its_optics() -> None:
    # A blind detector is a real detector, and a slot that produced no click is
    # the observation "nothing arrived here" -- which a receiver needs in order
    # to count sifted slots at all. The optical metadata is unchanged by it.
    timeline, driver, device, _reports = _build(detectors=_detectors(efficiency=0.0))
    reports = _emit_train(timeline, driver, device, count=4)

    assert len(reports) == 5  # n pulses -> n + 1 combinations
    for report in reports:
        assert report.raw_clicks == ()
        assert report.success is False
        assert report.outcome is None
        assert report.flags == (FLAG_NO_CLICK,)
        assert dict(report.meta)["temporal_overlap"] == pytest.approx(1.0)


def test_a_report_carries_no_measurement_because_none_was_run() -> None:
    # A threshold detector reads intensity. There is no basis, no qstate result
    # and no single measured signal, and the report says so rather than
    # inventing a label -- DPS sifts on detection time, not on a basis match.
    timeline, driver, device, _reports = _build(detectors=_detectors(efficiency=1.0))
    reports = _emit_train(timeline, driver, device, count=3)

    for report in reports:
        assert report.measurement_label is None
        assert report.measurement_method is None
        assert report.selection_index is None
        assert report.selection_probability is None
        assert report.selection_label is None
        assert report.qstate_result is None
        assert report.signal_id is None


def test_a_report_carries_the_optics_of_the_slot_it_decided() -> None:
    timeline, driver, device, _reports = _build(detectors=_detectors(efficiency=1.0))
    reports = _emit_train(timeline, driver, device, count=3)

    interference = {report.interference_index: report for report in device.reports}

    for report in reports:
        meta = dict(report.meta)
        # interference_index is the join key back to the optics record, which is
        # what replaces a signal id for a two-port measurement.
        optics = interference[meta["interference_index"]]
        assert meta["short_pulse_index"] == optics.short_pulse_index
        assert meta["long_pulse_index"] == optics.long_pulse_index
        assert meta["temporal_overlap"] == optics.temporal_overlap
        assert meta["mean_photon_number_0"] == optics.mean_photon_number_0
        assert meta["mean_photon_number_1"] == optics.mean_photon_number_1
        assert meta["output_signal_ids"] == optics.output_signal_ids
        assert isinstance(meta["phase_rad_0"], float)
        assert isinstance(meta["phase_rad_1"], float)

    # And the click that decided the slot names its port, not a basis outcome.
    clicked = [report for report in reports if report.success]
    assert clicked
    assert {report.outcome for report in clicked} <= {PORT_OUT_0, PORT_OUT_1}


def test_the_vacuum_slots_report_like_any_other() -> None:
    # The first pulse's short arm and the flushed long arm each meet vacuum and
    # split mu/2 both ways, so both ports carry equal probability and the slot
    # holds no bit. It is still a slot, still gets a report, and is dropped by
    # the agent on the None pulse index -- not special-cased here.
    timeline, driver, device, _reports = _build(detectors=_detectors(efficiency=1.0))
    reports = _emit_train(timeline, driver, device, count=4)

    edges = [
        report
        for report in reports
        if dict(report.meta)["short_pulse_index"] is None
        or dict(report.meta)["long_pulse_index"] is None
    ]
    assert len(edges) == 2

    # BS1 halves the pulse into two arms of mu/2; the surviving arm then meets
    # vacuum at BS2 and is halved again, so each port carries mu/4 and the slot
    # total is the one real arm.
    for report in edges:
        meta = dict(report.meta)
        assert meta["mean_photon_number_0"] == pytest.approx(MU / 4, rel=1e-9)
        assert meta["mean_photon_number_1"] == pytest.approx(MU / 4, rel=1e-9)
        # Equal mu means equal probability: neither port is preferred, which is
        # exactly why the slot carries no information.
        assert isclose(
            meta["mean_photon_number_0"],
            meta["mean_photon_number_1"],
            rel_tol=1e-12,
        )
    # One is the first pulse (no predecessor), one is the flush (no successor).
    assert [dict(report.meta)["long_pulse_index"] for report in edges][0] is None
    assert [dict(report.meta)["short_pulse_index"] for report in edges][1] is None


# --------------------------------------------------------------------------
# ports and emission
# --------------------------------------------------------------------------


def test_reports_are_stored_always_and_emitted_when_the_port_is_wired() -> None:
    timeline, driver, device, reports = _build(
        detectors=_detectors(efficiency=1.0),
        connect_reports=True,
    )
    _emit_train(timeline, driver, device, count=4)

    assert len(device.detection_reports) == 5
    assert [report for _tick, report in reports.received] == device.detection_reports

    # A no-click report is ready when the windows close; a click report when the
    # last contributing channel fired. Never before the resolve tick.
    for tick, report in reports.received:
        assert tick >= report.time


def test_an_unwired_detection_port_still_stores_every_slot() -> None:
    timeline, driver, device, reports = _build(
        detectors=_detectors(efficiency=1.0),
        connect_reports=False,
    )
    _emit_train(timeline, driver, device, count=4)

    assert len(device.detection_reports) == 5
    assert reports.received == []


def test_detection_leaves_the_optical_ports_byte_identical() -> None:
    # The optics file is the authority on the physics, and stays the authority
    # only if fitting detectors changes nothing it asserts. Same seed, same
    # train, detectors on and off.
    _t1, _s1, with_det, det_0, det_1 = _run_train((0.0, pi), seed=1)
    _t2, _s2, without_det, off_0, off_1 = _run_train((0.0, pi), seed=1, detectors=None)

    assert with_det.reports == without_det.reports
    assert with_det.interference_count == without_det.interference_count

    for detected, plain in ((det_0, off_0), (det_1, off_1)):
        assert len(detected.signals) == len(plain.signals)
        for emitted, unwatched in zip(detected.signals, plain.signals):
            assert emitted.id == unwatched.id
            assert emitted.coherent_state == unwatched.coherent_state
            assert emitted.temporal_mode_sigma_s == unwatched.temporal_mode_sigma_s
            assert emitted.emission_time == unwatched.emission_time
            assert emitted.meta == unwatched.meta

    assert without_det.detection_reports == []
    assert with_det.detection_reports != []


# --------------------------------------------------------------------------
# arm delays
# --------------------------------------------------------------------------


def test_a_common_transit_shifts_both_arms_and_changes_no_interference() -> None:
    # Only the difference between the arms is physical. Making the common
    # transit explicit must move every combination downstream by exactly that
    # amount and leave the overlap, the intensities and the pairing untouched.
    short_delay = 137

    timeline, driver, delayed, _r0 = _build(
        detectors=_detectors(efficiency=1.0),
        seed=5,
        short_delay_ticks=short_delay,
    )
    _emit_train(timeline, driver, delayed, count=4)

    baseline, driver_b, plain, _r1 = _build(
        detectors=_detectors(efficiency=1.0),
        seed=5,
    )
    _emit_train(baseline, driver_b, plain, count=4)

    assert len(delayed.reports) == len(plain.reports)
    for shifted, base in zip(delayed.reports, plain.reports):
        assert shifted.temporal_overlap == base.temporal_overlap
        assert shifted.delta_ticks == base.delta_ticks
        assert shifted.mean_photon_number_0 == base.mean_photon_number_0
        assert shifted.mean_photon_number_1 == base.mean_photon_number_1
        assert shifted.short_pulse_index == base.short_pulse_index
        assert shifted.long_pulse_index == base.long_pulse_index

    # Every combination a pulse opened is shifted by the common transit. The
    # flush is a deadline on an *arrival*, expressed in arrival ticks, so it
    # does not move -- a uniform transit cannot change which pulses can pair.
    for shifted, base in zip(delayed.reports, plain.reports):
        if base.short_pulse_index is not None:
            assert shifted.time == base.time + short_delay


# --------------------------------------------------------------------------
# detector state and dark counts
# --------------------------------------------------------------------------


def test_dead_time_carries_from_one_slot_into_the_next() -> None:
    # The two detectors are owned by the component, not built per slot, so
    # recovery is a property of the channel across the whole train. A dead time
    # longer than the slot period must cost the following slot.
    # Saturating mu, so every slot would click and only recovery can stop one.
    # A sampled version of this would pass on luck alone.
    def bright_port_slots(dead_time_ticks: int) -> int:
        detectors = _detectors(dead_time_ticks=dead_time_ticks)
        timeline, driver, device, _reports = _build(detectors=detectors)
        _emit_train(
            timeline,
            driver,
            device,
            count=6,
            mean_photon_number=BRIGHT_MU,
        )
        return sum(
            any(click.detector_id == "bob_d1" for click in report.raw_clicks)
            for report in device.detection_reports
        )

    # 6 pulses give 7 combinations, and with no dead time the bright port fires
    # on every one of them.
    assert bright_port_slots(0) == 7
    # A dead time longer than the slot period costs roughly every other slot.
    blocked = bright_port_slots(TAU_TICKS + 10)
    assert 0 < blocked < 7


def test_dark_counts_reach_a_slot_that_carries_no_light() -> None:
    # A detector fires on its own, and a receiver that only evaluated exposed
    # ports would report a dark-count rate of zero. mu = 0 everywhere, so every
    # click here is thermal.
    detectors = _detectors(efficiency=1.0, dark_count_rate_hz=5e10)
    timeline, driver, device, _reports = _build(
        detectors=detectors,
        detection_window_ticks=200,
    )
    for index in range(6):
        driver.emit(timeline, at_tick=index * TAU_TICKS, mean_photon_number=0.0)
    timeline.run_until_empty()

    clicks = [
        click for report in device.detection_reports for click in report.raw_clicks
    ]
    assert clicks
    assert all(click.trigger == "dark" for click in clicks)


def test_a_closed_gate_costs_the_slot_it_closes_over() -> None:
    # Gating is real receiver hardware, and the final flush lands two slot
    # periods after the last arrival -- a gate has to still be open then, or
    # accept losing that slot. This asserts the losing.
    gate = ScheduledGate(windows=(GateWindow(start=0, end=2 * TAU_TICKS),))
    timeline, driver, device, _reports = _build(
        detectors=_detectors(efficiency=1.0),
        gate_model=gate,
    )
    _emit_train(timeline, driver, device, count=4)

    late = [
        report for report in device.detection_reports if report.time >= 2 * TAU_TICKS
    ]
    assert late
    assert all(report.raw_clicks == () for report in late)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_optics_only_still_declares_no_rng_stream() -> None:
    # detectors=None must leave the component exactly as ideal as it was: a
    # declared-but-never-consumed stream would be a lie in the binding log.
    _timeline, _source, device, _s0, _s1 = _run_train((0.0, pi), detectors=None)
    assert device.detection_reports == []

    # Streams freeze on the first executed event, so the run has to be real.
    timeline, driver, bare, _reports = _build(detectors=None)
    _emit_train(timeline, driver, bare, count=3)

    assert bare.detection_reports == []
    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.rng("bob_di", "delay_interferometer", "any")


def test_detector_streams_use_the_four_segment_path() -> None:
    # Four segments, not three: the detector id is the segment that stops the
    # two channels of one receiver sharing a stream and clicking together.
    timeline, driver, device, _reports = _build()

    # Declared at bind, so asking for the same path returns the same stream and
    # does not create one.
    streams = {
        (detector_id, role): timeline.rng(
            "bob_di", "delay_interferometer", detector_id, role
        )
        for detector_id in ("bob_d0", "bob_d1")
        for role in ("efficiency", "dark", "jitter", "afterpulse")
    }
    assert len(set(map(id, streams.values()))) == 8
    assert timeline.rng("bob_di", "delay_interferometer", "resolver") is not None

    _emit_train(timeline, driver, device, count=3)

    # Every stream the component can ever draw from was declared before the run,
    # which is what `Timeline.rng` refusing a late one enforces.
    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.rng("bob_di", "delay_interferometer", "bob_d0", "unheard_of")


def test_a_run_replays_at_a_fixed_seed_and_moves_with_the_seed() -> None:
    first = _run_train((0.0, pi), num_slots=30, seed=11)[2].detection_reports
    again = _run_train((0.0, pi), num_slots=30, seed=11)[2].detection_reports
    other = _run_train((0.0, pi), num_slots=30, seed=999_983)[2].detection_reports

    assert first == again
    assert [report.outcome for report in first] != [report.outcome for report in other]
