"""The delay interferometer: train structure, timing boundaries, and rejections.

The physics inside ``_resolve`` is three lines -- a tick delta, one
``gaussian_temporal_overlap`` call, one ``interfere`` call -- and those three are
covered in ``test_coherent_optics.py``. What is tested here is everything around
them: how many combinations a train produces, which arm meets vacuum, exactly
which arrival ticks still pair, and that an expired deadline cannot destroy a
live pairing.

Pulses are driven by ``_PulseDriver`` rather than by a real source, because half
these tests need an arrival on one specific tick. It transmits at
``priority=0``, which is what ``QuantumChannel.delivery_priority`` defaults to,
so the flush-ordering tests exercise the real priority relationship. It is kept
in this file rather than added to ``tests/support/mock_components/`` while it has
one consumer; the sink there is the shared one and is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi

import pytest

from simyuj.components.connections import connect_ports
from simyuj.components.interferometers import (
    ACTION_FLUSH_DELAY_ARM,
    ACTION_INTERFERE,
    ACTION_RESOLVE_BS2,
    DelayArmFlush,
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

ATOL = 1e-12

MU = 0.2
TAU_TICKS = 1000
SIGMA_S = 3e-11
# 1000-tick slot period, so tau equals the pulse period: the DPS design point.
CLOCK_HZ = 1e9


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
        sigma_s: float | None = SIGMA_S,
        state_ref: int | None = None,
        coherent: bool = True,
    ) -> Signal:
        self._count += 1
        signal = Signal(
            id=f"{self.device_id}:pulse:{self._count}",
            signal_kind=SignalKind.PULSE,
            encoding_scheme=EncodingScheme.PHASE,
            emission_time=at_tick,
            origin=self.device_id,
            state_ref=state_ref,
            coherent_state=(
                CoherentState.from_mean_photon_number(
                    mean_photon_number,
                    phase_rad=phase_rad,
                )
                if coherent
                else None
            ),
            temporal_mode_sigma_s=sigma_s,
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


def _build(*, delay_ticks: int = TAU_TICKS, connect_outputs: bool = True, **kwargs):
    """Return ``(timeline, driver, interferometer, sink_0, sink_1)`` wired up."""
    timeline = Timeline(master_seed=1)
    driver = _PulseDriver()
    device = DelayInterferometer(
        device_id="bob_di",
        # Optics only, said out loud. Every assertion in this file is about
        # exact amplitudes, energy to fifteen decimals, and exact ticks; behind
        # a detector each becomes a statistical claim needing a seed sweep,
        # which is a strictly weaker test of the same physics. Detection has its
        # own file.
        detectors=kwargs.pop("detectors", None),
        delay_ticks=delay_ticks,
        **kwargs,
    )
    sink_0 = SignalSink(device_id="out0_sink")
    sink_1 = SignalSink(device_id="out1_sink")

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

    device.bind(BindingContext(timeline=timeline, logger=timeline.logger))
    return timeline, driver, device, sink_0, sink_1


def _run_train(phases, *, num_slots: int = 5, seed: int = 1):
    """Run a real source through the interferometer and return the pieces."""
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
        detectors=None,
        delay_ticks=TAU_TICKS,
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


# --------------------------------------------------------------------------
# train structure
# --------------------------------------------------------------------------


def test_an_n_pulse_train_gives_n_plus_one_combinations_and_conserves_energy() -> None:
    num_slots = 5
    timeline, source, device, sink_0, sink_1 = _run_train((0.0,), num_slots=num_slots)

    assert source.pulse_count == num_slots
    # The first pulse's short arm and the last pulse's long arm each meet
    # vacuum, so the device produces one more slot than it received.
    assert device.interference_count == num_slots + 1
    assert len(device.reports) == num_slots + 1
    assert len(sink_0.received) == num_slots + 1
    assert len(sink_1.received) == num_slots + 1

    # Energy ledger: 1/2 mu + (N-1) mu + 1/2 mu == N mu. Nothing is lost in an
    # ideal interferometer, and the two half-slots at the ends are where the
    # count and the energy reconcile.
    total_in = sum(report.mean_photon_number_in for report in device.reports)
    assert total_in == pytest.approx(num_slots * MU, abs=ATOL)
    for report in device.reports:
        assert (
            report.mean_photon_number_0 + report.mean_photon_number_1
        ) == pytest.approx(report.mean_photon_number_in, abs=ATOL)

    # Nothing is left waiting, and no quantum state was ever created.
    assert device.held_arm_count == 0
    assert timeline.qstate.size() == 0


def test_equal_phases_leave_port_zero_dark_and_a_pi_step_swaps_them() -> None:
    # This is the DPS signal itself: the bit lives in the differential phase,
    # and the device turns it into which port is bright.
    _timeline, _source, constant, _s0, _s1 = _run_train((0.0,))
    for report in constant.reports[1:-1]:
        assert report.mean_photon_number_0 == pytest.approx(0.0, abs=ATOL)
        assert report.mean_photon_number_1 == pytest.approx(MU, abs=ATOL)

    _timeline, _source, alternating, _s0, _s1 = _run_train((0.0, pi))
    for report in alternating.reports[1:-1]:
        assert report.mean_photon_number_0 == pytest.approx(MU, abs=ATOL)
        assert report.mean_photon_number_1 == pytest.approx(0.0, abs=ATOL)


def test_the_first_and_last_combinations_meet_vacuum_and_carry_no_bit() -> None:
    _timeline, _source, device, _s0, _s1 = _run_train((0.0,), num_slots=4)

    first, last = device.reports[0], device.reports[-1]

    # None here is a defined absence -- that arm was the coherent vacuum -- and
    # is what tells a decoder these two slots carry no differential bit.
    assert first.short_pulse_index == 1
    assert first.long_pulse_index is None
    assert last.short_pulse_index is None
    assert last.long_pulse_index == 4

    assert first.signal_ids == ("alice:pulse:1",)
    assert last.signal_ids == ("alice:pulse:4",)

    # Half a pulse arrives and splits evenly: mu/4 on each port.
    for report in (first, last):
        assert report.mean_photon_number_0 == pytest.approx(MU / 4.0, abs=ATOL)
        assert report.mean_photon_number_1 == pytest.approx(MU / 4.0, abs=ATOL)


# --------------------------------------------------------------------------
# the pairing deadline and its one tick of ordering margin
# --------------------------------------------------------------------------


def _pairing_report(device: DelayInterferometer):
    """Return the one combination with two real arms, or ``None``."""
    paired = [
        report
        for report in device.reports
        if report.short_pulse_index is not None and report.long_pulse_index is not None
    ]
    assert len(paired) <= 1
    return paired[0] if paired else None


@pytest.mark.parametrize(
    ("offset", "pairs"),
    [
        # The physical deadline. The flush is not due yet.
        (2 * TAU_TICKS, True),
        # The flush event's own tick. The delivery is priority 0 and the flush
        # is priority 10000, both already queued, so the arrival is dispatched
        # first within the batch and still finds the arm.
        (2 * TAU_TICKS + 1, True),
        # One tick past it. The arm has already been flushed against vacuum.
        (2 * TAU_TICKS + 2, False),
    ],
    ids=["deadline", "flush_tick", "after_flush"],
)
def test_the_last_tick_on_which_a_pulse_still_pairs(offset, pairs) -> None:
    timeline, driver, device, _s0, _s1 = _build()
    driver.emit(timeline, at_tick=0)
    driver.emit(timeline, at_tick=offset)
    timeline.run_until_empty()

    report = _pairing_report(device)
    if not pairs:
        assert report is None
        # Both arms still leave, each against vacuum, so nothing is lost.
        assert device.interference_count == 4
        total = sum(r.mean_photon_number_in for r in device.reports)
        assert total == pytest.approx(2 * MU, abs=ATOL)
        return

    assert report is not None
    assert report.short_pulse_index == 2
    assert report.long_pulse_index == 1
    # The short arm arrives at `offset`; the long arm reached BS2 at tau.
    assert report.delta_ticks == offset - TAU_TICKS
    assert device.interference_count == 3


def test_the_overlap_discarded_at_the_deadline_is_not_negligible() -> None:
    # The deadline is the nearest-neighbour assumption, not a decay estimate.
    # At sigma = tau a pulse one tick too late is discarded while it would still
    # have interfered at gamma ~ 0.78. Asserted so the assumption stays visible
    # in the suite rather than living only in a docstring.
    sigma_at_tau = TAU_TICKS * 1e-12
    gamma = exp(-((TAU_TICKS * 1e-12) ** 2) / (4.0 * sigma_at_tau**2))

    assert gamma == pytest.approx(0.7788007830714, abs=1e-9)
    assert gamma > 0.75


# --------------------------------------------------------------------------
# the stale-flush guard
# --------------------------------------------------------------------------


def test_a_stale_flush_cannot_destroy_a_later_pulses_held_arm() -> None:
    # The failure this guard prevents is pulse 1's expired deadline flushing
    # pulse 2's arm. In that situation the holder is PRESENT AND DIFFERENT, so a
    # guard written as `if self._held is None: return` would pass an
    # empty-holder test and still ship the bug. Both flushes below are delivered
    # to a full holder; only the generation tells them apart.
    timeline, driver, device, _s0, _s1 = _build()
    driver.emit(timeline, at_tick=0)
    driver.emit(timeline, at_tick=TAU_TICKS)
    timeline.run_until(TAU_TICKS)

    # Pulse 1 paired with pulse 2 already; the holder now carries pulse 2's long
    # arm, which is hold generation 2. Hold ids increment once per arrival.
    assert device.held_arm_count == 1
    assert device.interference_count == 2

    stale = Event(
        time=timeline.current_time,
        priority=0,
        target_ref=device,
        action=ACTION_FLUSH_DELAY_ARM,
        payload_ref=DelayArmFlush(hold_id=1),
        source=device,
    )
    device.handle_event(stale, timeline)

    # Untouched: no combination resolved, and pulse 2's arm is still waiting.
    assert device.interference_count == 2
    assert device.held_arm_count == 1

    # Positive control in the same test, so the assertions above cannot pass by
    # the guard rejecting everything.
    live = Event(
        time=timeline.current_time,
        priority=0,
        target_ref=device,
        action=ACTION_FLUSH_DELAY_ARM,
        payload_ref=DelayArmFlush(hold_id=2),
        source=device,
    )
    device.handle_event(live, timeline)

    assert device.interference_count == 3
    assert device.held_arm_count == 0
    assert device.reports[-1].short_pulse_index is None
    assert device.reports[-1].long_pulse_index == 2


# --------------------------------------------------------------------------
# causality: arriving before the held arm has reached BS2
# --------------------------------------------------------------------------


def test_a_pulse_arriving_before_the_held_arm_reaches_bs2_is_deferred() -> None:
    # Reachable whenever the source uses a stochastic timing profile, which
    # shortens the spacing below tau. Resolving on arrival would emit light that
    # has not arrived yet; rejecting the pulse would turn a legitimate timing
    # profile into a seed-dependent mid-run abort.
    early = TAU_TICKS // 2
    timeline, driver, device, _s0, _s1 = _build()
    driver.emit(timeline, at_tick=0)
    driver.emit(timeline, at_tick=early)

    # Nothing resolves the pair at the arrival tick.
    timeline.run_until(early)
    assert device.interference_count == 1

    timeline.run_until_empty()

    report = _pairing_report(device)
    assert report is not None
    # It resolved when the long arm actually reached BS2, not when the pulse
    # landed, and the overlap reflects the real separation.
    assert report.time == TAU_TICKS
    assert report.delta_ticks == TAU_TICKS - early
    assert report.temporal_overlap < 1.0

    assert device.interference_count == 3
    total = sum(r.mean_photon_number_in for r in device.reports)
    assert total == pytest.approx(2 * MU, abs=ATOL)


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"coherent": False}, "coherent_state=None"),
        ({"state_ref": 7}, "state_ref=7"),
        ({"sigma_s": None}, "temporal_mode_sigma_s=None"),
    ],
    ids=["no_amplitude", "qstate_backed", "no_temporal_mode"],
)
def test_the_device_rejects_what_it_cannot_interfere(kwargs, expected) -> None:
    # A transform component's whole purpose is the transformation, so passing an
    # uninterferable signal through would make it a silent no-op for exactly the
    # wiring mistake it should catch.
    timeline, driver, _device, _s0, _s1 = _build()
    driver.emit(timeline, at_tick=0, **kwargs)

    with pytest.raises(ValueError, match=expected):
        timeline.run_until_empty()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"delay_s": 1e-9, "delay_ticks": 1000},
        # Rounds to 0 ticks at 1 tick == 1 ps: a balanced interferometer, in
        # which every pulse would interfere only with itself.
        {"delay_s": 1e-13},
    ],
    ids=["neither", "both", "sub_tick"],
)
def test_construction_requires_exactly_one_delay_of_at_least_one_tick(kwargs) -> None:
    with pytest.raises(ValueError):
        DelayInterferometer(device_id="bob_di", detectors=None, **kwargs)


def test_both_output_ports_must_be_connected() -> None:
    # An ideal interferometer always puts light on both ports; the destructive
    # one carrying nearly nothing is a result, not an absence. A half-wired
    # device would otherwise emit into nothing and look like it worked.
    timeline, driver, _device, _s0, _s1 = _build(connect_outputs=False)
    driver.emit(timeline, at_tick=0)

    with pytest.raises(RuntimeError, match="bob_di.out_0' is not connected"):
        timeline.run_until_empty()


def test_execution_is_rejected_before_binding_and_for_bad_events() -> None:
    device = DelayInterferometer(
        device_id="bob_di",
        detectors=None,
        delay_ticks=TAU_TICKS,
    )
    timeline = Timeline(master_seed=1)

    unbound = Event(
        time=0,
        priority=0,
        target_ref=device,
        action=ACTION_INTERFERE,
        payload_ref=None,
        source=device,
    )
    with pytest.raises(RuntimeError, match="must be bound"):
        device.handle_event(unbound, timeline)

    device.bind(BindingContext(timeline=timeline, logger=timeline.logger))

    with pytest.raises(TypeError, match="must be PortDelivery"):
        device.handle_event(unbound, timeline)

    for action, message in (
        (ACTION_RESOLVE_BS2, "must be PendingCombination"),
        (ACTION_FLUSH_DELAY_ARM, "must be DelayArmFlush"),
    ):
        event = Event(
            time=0,
            priority=0,
            target_ref=device,
            action=action,
            payload_ref="not a payload",
            source=device,
        )
        with pytest.raises(TypeError, match=message):
            device.handle_event(event, timeline)

    unknown = Event(
        time=0,
        priority=0,
        target_ref=device,
        action="nonsense",
        payload_ref=None,
        source=device,
    )
    with pytest.raises(ValueError, match="unsupported event action"):
        device.handle_event(unknown, timeline)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_the_device_declares_no_rng_stream_and_replays_at_any_seed() -> None:
    # Ideal by specification, so there is nothing to sample. A
    # declared-but-never-consumed stream would be a lie in the binding log.
    timeline, _source, first, _s0, _s1 = _run_train((0.0, pi), seed=1)

    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.rng("bob_di", "delay_interferometer", "any")

    _timeline, _source, second, _s0, _s1 = _run_train((0.0, pi), seed=999_983)
    assert first.reports == second.reports
