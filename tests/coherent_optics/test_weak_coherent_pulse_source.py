"""Tests for ``WeakCoherentPulseSource``.

Structured after ``tests/components/test_components_single_photon_source.py``:
the same ``ReportAgent`` / ``_attach_context`` scaffolding for the report-port
path, and the shared ``SignalSink`` for the quantum path.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field
from math import pi

import pytest

from simyuj.components.connections import connect_ports
from simyuj.components.sources import (
    DPS_PHASES,
    CoherentPulsePreparationReport,
    FixedCarrierPhase,
    FixedIntensity,
    FixedPhase,
    PerPulseRandomCarrierPhase,
    PhaseSequence,
    RandomPhaseChoice,
    WeakCoherentPulseSource,
)
from simyuj.components.sources._common import DeltaTiming, GaussianTiming
from simyuj.control import AGENT_REPORT, Agent, AgentContext
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.network import Network
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.primitives.coherent_state import CoherentState
from simyuj.runtime.binding import BindingContext
from simyuj.signal import EncodingScheme, SignalKind

from ..support.mock_components import ACTION_RECEIVE_SIGNAL, SignalSink

ATOL = 1e-12

# 1 THz slot clock: one pulse per picosecond tick, so slot arithmetic in these
# tests is readable directly in ticks.
FREQUENCY_HZ = 1e12


@dataclass(slots=True)
class ReportAgent(Agent):
    reports_seen: list[object] = field(default_factory=list)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        del ctx
        self.reports_seen.append(report)


def _attach_context(agent: Agent, timeline: Timeline) -> Network:
    network = Network()
    topology = NetworkTopology(network)
    planner = RoutePlanner(topology)

    def provider(event: Event, current_timeline: Timeline) -> AgentContext:
        return AgentContext(
            agent_id=agent.agent_id,
            node_id=None,
            session_id="session-1",
            timeline=current_timeline,
            event=event,
            network=network,
            topology=topology,
            route_planner=planner,
        )

    agent.attach_context_provider(provider)
    return network


def _make_source(**overrides) -> WeakCoherentPulseSource:
    kwargs = {
        "device_id": "alice_laser",
        "frequency_hz": FREQUENCY_HZ,
        "intensity": FixedIntensity(0.2),
        "duration_s": 5e-12,
    }
    kwargs.update(overrides)
    return WeakCoherentPulseSource(**kwargs)


def _run(source: WeakCoherentPulseSource, *, seed: int = 1, until: int = 20):
    timeline = Timeline(master_seed=seed)
    sink = SignalSink(device_id=f"{source.device_id}_sink")
    connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
    )
    source.schedule_start(timeline)
    timeline.run_until(until)
    return timeline, sink


def _wrapped_delta(a: float, b: float) -> float:
    """Signed difference between two phases, wrapped into ``(-pi, pi]``."""
    return (a - b + pi) % (2.0 * pi) - pi


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------


def test_every_active_slot_emits_exactly_one_pulse() -> None:
    # No emission Bernoulli: pulse_count equals the number of active slots.
    source = _make_source()
    timeline, sink = _run(source)

    assert source.pulse_count == 5
    assert len(sink.received) == 5
    assert [time for time, _ in sink.received] == [0, 1, 2, 3, 4]
    assert len(source.reports) == 5
    del timeline


def test_emitted_signal_is_a_coherent_pulse_with_no_qstate() -> None:
    source = _make_source(temporal_mode_sigma_s=2e-13)
    timeline, sink = _run(source)

    signal = sink.signals[0]
    assert signal.signal_kind is SignalKind.PULSE
    assert signal.encoding_scheme is EncodingScheme.PHASE
    assert signal.state_ref is None
    assert signal.state_targets == ()
    assert isinstance(signal.coherent_state, CoherentState)
    assert signal.temporal_mode_sigma_s == 2e-13
    assert signal.polarization is None
    assert signal.id == "alice_laser:pulse:1"

    # The single sharpest statement that no photon number was ever sampled:
    # the source never touches the quantum state manager at all.
    assert timeline.qstate.size() == 0


def test_each_pulse_carries_its_own_coherent_state_object() -> None:
    # The reference implementation built one CoherentState in __post_init__ and
    # attached that exact object to every pulse, asserting identity with `is`.
    # That is what forecloses per-pulse decoy intensities and polarization, so
    # this test asserts the inverse.
    source = _make_source(encoding_phase=RandomPhaseChoice(DPS_PHASES))
    _timeline, sink = _run(source)

    states = [signal.coherent_state for signal in sink.signals]
    assert len({id(state) for state in states}) == len(states)


def test_amplitude_is_sqrt_mu_at_the_summed_phase() -> None:
    source = _make_source(
        intensity=FixedIntensity(0.36),
        carrier_phase=FixedCarrierPhase(0.4),
        encoding_phase=FixedPhase(pi),
    )
    _timeline, sink = _run(source)

    for signal in sink.signals:
        state = signal.coherent_state
        assert state.mean_photon_number == pytest.approx(0.36, abs=ATOL)
        # phase_rad is the total *wrapped* phase, so compare modulo 2*pi.
        assert _wrapped_delta(state.phase_rad, 0.4 + pi) == pytest.approx(0.0, abs=ATOL)


def test_zero_mean_photon_number_emits_a_real_vacuum_pulse() -> None:
    # mu = 0 is coherent vacuum, not a skipped slot: it still produces a
    # signal, a delivery, and a report, and it still counts.
    source = _make_source(intensity=FixedIntensity(0.0))
    _timeline, sink = _run(source)

    assert source.pulse_count == 5
    assert len(sink.received) == 5
    assert len(source.reports) == 5
    assert sink.signals[0].coherent_state.alpha == 0j
    assert sink.signals[0].coherent_state.phase_rad == 0.0


def test_signal_metadata_carries_identity_but_not_the_preparation_choices() -> None:
    # Protocol knowledge is earned from a message. A downstream device must not
    # be able to read the sender's phase choice off a signal in flight.
    source = _make_source(encoding_phase=RandomPhaseChoice(DPS_PHASES))
    _timeline, sink = _run(source)

    meta = dict(sink.signals[0].meta)
    assert meta == {"source_device_id": "alice_laser", "pulse_index": 1}

    timing_meta = dict(sink.signals[0].timing_meta)
    assert timing_meta["emission_slot_tick"] == 0
    assert timing_meta["emission_period_ticks"] == 1


def test_emission_requires_a_connected_output_port() -> None:
    timeline = Timeline(master_seed=1)
    source = _make_source()
    source.schedule_start(timeline)

    with pytest.raises(RuntimeError, match="is not connected"):
        timeline.run_until(2)


def test_unbound_source_rejects_events() -> None:
    source = _make_source()
    timeline = Timeline(master_seed=1)

    with pytest.raises(RuntimeError, match="must be bound"):
        source.handle_event(
            Event(time=0, target_ref=source, action="start", payload_ref=None),
            timeline,
        )


# --------------------------------------------------------------------------
# the preparation report
# --------------------------------------------------------------------------


def test_report_records_both_phases_separately_and_their_alphabet_indices() -> None:
    source = _make_source(
        intensity=FixedIntensity(0.15),
        carrier_phase=FixedCarrierPhase(0.25),
        encoding_phase=PhaseSequence((0.0, pi, 0.0, pi, 0.0)),
    )
    _timeline, sink = _run(source)

    assert [report.encoding_phase_index for report in source.reports] == [0, 1, 2, 3, 4]
    assert [report.encoding_phase_rad for report in source.reports] == [
        0.0,
        pi,
        0.0,
        pi,
        0.0,
    ]

    for index, report in enumerate(source.reports, start=1):
        assert isinstance(report, CoherentPulsePreparationReport)
        assert report.pulse_index == index
        assert report.report_id == f"alice_laser:prep:{index}"
        assert report.device_id == "alice_laser"
        assert report.signal_ids == (f"alice_laser:pulse:{index}",)
        assert report.mean_photon_number == pytest.approx(0.15, abs=ATOL)
        assert report.intensity_index == 0
        assert report.carrier_phase_rad == pytest.approx(0.25, abs=ATOL)
        assert report.polarization is None
        assert report.polarization_index is None
        # Kept apart, not pre-summed: their sum is already on the state, and
        # separating them is what lets a later analysis attribute a visibility
        # loss to carrier drift rather than encoding.
        summed = report.carrier_phase_rad + report.encoding_phase_rad
        assert _wrapped_delta(
            report.coherent_state.phase_rad,
            summed,
        ) == pytest.approx(0.0, abs=ATOL)

    assert [report.coherent_state for report in source.reports] == [
        signal.coherent_state for signal in sink.signals
    ]


def test_report_carries_no_qstate_or_sampler_fields() -> None:
    source = _make_source()
    _run(source)

    report = source.reports[0]
    for absent in ("state_ref", "state_targets", "sampler_index", "sampler_label"):
        assert not hasattr(report, absent)


def test_the_encoding_phase_is_not_recoverable_from_the_amplitude() -> None:
    # cmath.phase returns the total wrapped phase. With a nonzero carrier phase
    # it already differs from the encoding phase, which is why protocol code
    # must read encoding_phase_index from the report and never arg(alpha).
    source = _make_source(
        carrier_phase=FixedCarrierPhase(1.0),
        encoding_phase=FixedPhase(pi),
    )
    _timeline, sink = _run(source)

    state = sink.signals[0].coherent_state
    assert cmath.phase(state.alpha) != pytest.approx(pi, abs=1e-6)
    assert source.reports[0].encoding_phase_rad == pi
    assert source.reports[0].encoding_phase_index == 0


def test_report_reaches_an_agent_through_the_report_port() -> None:
    timeline = Timeline(master_seed=1)
    sink = SignalSink()
    agent = ReportAgent(agent_id="alice_agent")
    network = _attach_context(agent, timeline)

    source = _make_source(duration_s=1e-12)
    connect_ports(
        source.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
    )
    network.wire_ports(
        "alice_laser_reports",
        source.report_port,
        agent.reports.port("source"),
        target_action=AGENT_REPORT,
    )

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert agent.reports_seen == [source.reports[0]]
    assert isinstance(agent.reports_seen[0], CoherentPulsePreparationReport)


# --------------------------------------------------------------------------
# RNG streams and determinism
# --------------------------------------------------------------------------


def test_four_rng_streams_are_declared_and_no_fifth_can_be_added() -> None:
    source = _make_source()
    timeline, _sink = _run(source)

    # Already-declared paths still resolve after the timeline freezes.
    for stream_name in ("timing", "intensity", "carrier", "encoding"):
        timeline.rng("alice_laser", "weak_coherent_pulse_source", stream_name)

    # An undeclared one does not, which is what proves the four above were
    # declared during bind() rather than lazily on first use.
    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.rng("alice_laser", "weak_coherent_pulse_source", "emission")


def test_deterministic_configuration_is_seed_independent() -> None:
    def trace(seed: int):
        source = _make_source(
            carrier_phase=FixedCarrierPhase(0.3),
            encoding_phase=PhaseSequence((0.0, pi), repeat=True),
            timing_profile=DeltaTiming(),
        )
        _timeline, sink = _run(source, seed=seed)
        return (
            [time for time, _ in sink.received],
            [str(signal.id) for signal in sink.signals],
            [signal.coherent_state.alpha for signal in sink.signals],
            list(source.reports),
        )

    assert trace(1) == trace(999_983)


def test_random_selectors_replay_from_the_seed_but_differ_across_seeds() -> None:
    def phases(seed: int) -> list[int]:
        source = _make_source(encoding_phase=RandomPhaseChoice(DPS_PHASES))
        _run(source, seed=seed)
        return [report.encoding_phase_index for report in source.reports]

    assert phases(7) == phases(7)
    assert phases(7) != phases(8)


def test_fixed_carrier_phase_gives_exactly_zero_differential_phase() -> None:
    # This is the property differential-phase encoding rests on, asserted where
    # it is exact rather than inferred from a visibility measurement.
    source = _make_source(carrier_phase=FixedCarrierPhase(0.7))
    _run(source)

    carriers = [report.carrier_phase_rad for report in source.reports]
    assert all(b - a == 0.0 for a, b in zip(carriers, carriers[1:]))


def test_per_pulse_random_carrier_phase_gives_a_different_phase_each_pulse() -> None:
    source = _make_source(carrier_phase=PerPulseRandomCarrierPhase())
    _run(source, seed=11)

    carriers = [report.carrier_phase_rad for report in source.reports]
    assert len(set(carriers)) == len(carriers)
    assert all(-pi <= phase < pi for phase in carriers)
    assert all(b - a != 0.0 for a, b in zip(carriers, carriers[1:]))


def test_intensity_and_encoding_draw_from_independent_streams() -> None:
    # Changing the intensity policy must not shift the encoding-phase sequence,
    # or adding decoy levels later would break replay of an existing run.
    def encoding_indices(intensity) -> list[int]:
        source = _make_source(
            intensity=intensity,
            encoding_phase=RandomPhaseChoice(DPS_PHASES),
        )
        _run(source, seed=5)
        return [report.encoding_phase_index for report in source.reports]

    assert encoding_indices(FixedIntensity(0.2)) == encoding_indices(
        FixedIntensity(0.9),
    )


def test_timing_profile_jitter_uses_the_timing_stream_only() -> None:
    source = _make_source(
        timing_profile=GaussianTiming(
            mean_emission_delay_ticks=0,
            emission_delay_stddev_ticks=0.0,
            max_emission_delay_ticks=0,
        ),
    )
    _timeline, sink = _run(source)

    assert [time for time, _ in sink.received] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# construction-time validation
# --------------------------------------------------------------------------


def test_binding_to_a_second_timeline_is_rejected() -> None:
    source = _make_source()
    first = Timeline(master_seed=1)
    source.bind(BindingContext(timeline=first, logger=first.logger))

    # Rebinding to the same timeline is idempotent.
    source.bind(BindingContext(timeline=first, logger=first.logger))

    second = Timeline(master_seed=1)
    with pytest.raises(RuntimeError, match="already bound to another timeline"):
        source.bind(BindingContext(timeline=second, logger=second.logger))


def test_bind_rejects_a_non_binding_context() -> None:
    source = _make_source()
    with pytest.raises(TypeError, match="context must be BindingContext"):
        source.bind(object())


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"device_id": ""}, ValueError),
        ({"frequency_hz": 0.0}, ValueError),
        ({"frequency_hz": -1e12}, ValueError),
        ({"frequency_hz": True}, TypeError),
        ({"encoding_scheme": "phase"}, TypeError),
        ({"wavelength_nm": 0.0}, ValueError),
        ({"wavelength_nm": True}, TypeError),
        ({"start_time_s": -1e-12}, ValueError),
        ({"duration_s": 0.0}, ValueError),
        ({"temporal_mode_sigma_s": 0.0}, ValueError),
        ({"temporal_mode_sigma_s": -1e-13}, ValueError),
        ({"temporal_mode_sigma_s": True}, TypeError),
        ({"intensity": object()}, TypeError),
        ({"carrier_phase": object()}, TypeError),
        ({"encoding_phase": object()}, TypeError),
    ],
)
def test_construction_rejects_invalid_configuration(overrides, expected) -> None:
    with pytest.raises(expected):
        _make_source(**overrides)


def test_temporal_mode_sigma_is_optional_and_not_quantized_to_ticks() -> None:
    # seconds_to_ticks rounds to integer picoseconds; a width far below one tick
    # must survive as a float, or no partial-overlap model is writable later.
    assert _make_source().temporal_mode_sigma_s is None
    assert _make_source(temporal_mode_sigma_s=1e-15).temporal_mode_sigma_s == 1e-15


def test_exhausted_phase_sequence_aborts_the_run_with_a_clear_reason() -> None:
    source = _make_source(encoding_phase=PhaseSequence((0.0, pi)))

    with pytest.raises(RuntimeError, match="phase sequence exhausted"):
        _run(source)
