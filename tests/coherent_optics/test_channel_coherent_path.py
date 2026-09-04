"""Coherent-amplitude transport through ``QuantumChannel``.

The qstate carrier path is covered by ``test_components_quantum_channel.py`` and
is not re-tested here; ``qstate_payload_role`` by ``test_quantum_targets.py``;
``attenuated`` / ``phase_shifted`` by ``test_coherent_optics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import pytest

from simyuj.components import (
    ACTION_TRANSMIT_QUANTUM,
    Port,
    PortDirection,
    PortKind,
    QuantumChannel,
    connect_ports,
)
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.qstate.noise import depolarizing
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink

from ..support.binding import binding_context
from ..support.mock_components import ACTION_RECEIVE_SIGNAL, SignalSink

ATOL = 1e-12
DELAY_TICKS = 5


@dataclass(slots=True)
class _Sender(Component):
    device_id: str = "sender"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:  # pragma: no cover
        raise AssertionError("sender receives nothing")


def _pulse(
    *,
    mean_photon_number: float = 0.2,
    phase_rad: float = 0.0,
    signal_id: str = "pulse-1",
    state_ref: int | None = None,
    state_targets: tuple[SubsystemHandle, ...] = (),
) -> Signal:
    return Signal(
        id=signal_id,
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.PHASE,
        emission_time=0,
        origin="alice_laser",
        coherent_state=CoherentState.from_mean_photon_number(
            mean_photon_number,
            phase_rad=phase_rad,
        ),
        temporal_mode_sigma_s=3e-11,
        state_ref=state_ref,
        state_targets=state_targets,
    )


def _chain(**channel_kwargs):
    """Sender -> channel -> sink, wired and ready to bind."""
    channel_kwargs.setdefault("delay_ticks", DELAY_TICKS)
    sender = _Sender()
    channel = QuantumChannel(channel_id="qch", **channel_kwargs)
    sink = SignalSink(device_id="pulse_sink")

    inbound = connect_ports(
        sender.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    connect_ports(
        channel.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
    )
    return sender, channel, sink, inbound


def _send(signal, *, seed: int = 1, logger=None, **channel_kwargs):
    """Run one pulse end to end and return ``(channel, sink, timeline)``."""
    timeline = (
        Timeline(master_seed=seed, logger=logger)
        if logger
        else Timeline(master_seed=seed)
    )
    sender, channel, sink, inbound = _chain(**channel_kwargs)
    channel.bind(binding_context(timeline))
    inbound.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(50)
    return channel, sink, timeline


# --------------------------------------------------------------------------
# the amplitude path
# --------------------------------------------------------------------------


def test_pulse_is_attenuated_by_eta_and_arrives_on_the_nominal_tick() -> None:
    channel, sink, _tl = _send(
        _pulse(mean_photon_number=0.4, phase_rad=0.7),
        fixed_insertion_loss_db=3.0,
    )
    eta = channel.survival_probability

    arrival, delivered = sink.received[0]
    assert arrival == DELAY_TICKS
    assert delivered.coherent_state.mean_photon_number == pytest.approx(
        eta * 0.4, abs=ATOL
    )
    # phase survives attenuation, so adjacent pulses stay comparable
    assert delivered.coherent_state.phase_rad == pytest.approx(0.7, abs=ATOL)
    assert delivered.temporal_mode_sigma_s == 3e-11


def test_nothing_is_discarded_however_lossy_the_fibre_is() -> None:
    # lost_count == 0 here means "nothing was discarded", not "lossless".
    channel, sink, _tl = _send(_pulse(), fixed_insertion_loss_db=30.0)

    assert channel.lost_count == 0
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.attenuated_count == 1
    assert len(sink.received) == 1


def test_extreme_attenuation_delivers_a_vanishing_pulse_on_time() -> None:
    # eta -> 0 is a real optical state arriving in its slot, not a drop.
    # Deciding no photon was seen is the detector's job.
    #
    # Attenuation is continuous and asymptotic: 400 dB gives mu = 1e-40, not
    # zero, and no finite dB loss ever reaches exactly zero. That is correct
    # and must not be special-cased -- a detector's 1 - exp(-eta*mu) returns
    # "no click" from its own statistics on a tiny-but-nonzero mu with no
    # branch. Exact vacuum is reachable, but only from a source that prepares
    # mu = 0 upstream.
    _channel, sink, _tl = _send(_pulse(), fixed_insertion_loss_db=400.0)

    arrival, delivered = sink.received[0]
    assert arrival == DELAY_TICKS
    assert 0.0 < delivered.coherent_state.mean_photon_number < 1e-30

    _channel, vacuum_sink, _tl = _send(_pulse(mean_photon_number=0.0))
    assert vacuum_sink.signals[0].coherent_state.mean_photon_number == 0.0


def test_the_amplitude_path_consumes_no_randomness() -> None:
    # The strongest available form of "_is_lost never runs": a Bernoulli draw
    # here would reintroduce the per-slot photon-number sampling the coherent
    # source exists to avoid, and would make the output seed-dependent.
    def trace(seed: int):
        _channel, sink, _tl = _send(
            _pulse(mean_photon_number=0.3),
            seed=seed,
            fixed_insertion_loss_db=6.0,
        )
        return [(t, s.id, s.coherent_state.alpha) for t, s in sink.received]

    assert trace(1) == trace(999_983)


def test_phase_noise_shifts_the_phase_and_preserves_mu() -> None:
    quiet = _send(_pulse(), fixed_insertion_loss_db=0.0)[1].signals[0]
    noisy = _send(
        _pulse(), seed=11, fixed_insertion_loss_db=0.0, phase_noise_stddev_rad=0.4
    )[1].signals[0]

    assert quiet.coherent_state.phase_rad == pytest.approx(0.0, abs=ATOL)
    assert noisy.coherent_state.phase_rad != pytest.approx(0.0, abs=1e-6)
    assert noisy.coherent_state.mean_photon_number == pytest.approx(
        quiet.coherent_state.mean_photon_number, abs=ATOL
    )


def test_pulse_forwarded_is_a_separate_topic_that_never_claims_a_loss_trial() -> None:
    log_sink = MemorySink()
    logger = SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink])
    _channel, sink, _tl = _send(
        _pulse(mean_photon_number=0.5), logger=logger, fixed_insertion_loss_db=3.0
    )

    topics = [r.category for r in log_sink.records]
    assert "components.channels.quantum.pulse_forwarded" in topics
    # The qstate record's meta is asserted by exact equality elsewhere; reusing
    # its topic with a different key set would break a passing test.
    assert "components.channels.quantum.signal_forwarded" not in topics

    record = next(
        r
        for r in log_sink.records
        if r.category == "components.channels.quantum.pulse_forwarded"
    )
    meta = dict(record.meta)
    assert "survival_probability" not in meta
    assert meta["channel_power_transmission"] == pytest.approx(
        _channel_eta := 10 ** (-3.0 / 10), abs=ATOL
    )
    assert meta["mean_photon_number_in"] == pytest.approx(0.5, abs=ATOL)
    assert meta["mean_photon_number_out"] == pytest.approx(_channel_eta * 0.5, abs=ATOL)
    assert meta["qstate_payload_role"] is None

    delivered_meta = dict(sink.signals[0].meta)
    assert "survival_probability" not in delivered_meta
    assert delivered_meta["channel_power_transmission"] == pytest.approx(
        _channel_eta, abs=ATOL
    )


# --------------------------------------------------------------------------
# rejections, all at event time
# --------------------------------------------------------------------------


def test_jitter_is_rejected_when_a_pulse_arrives_not_at_construction() -> None:
    # A channel cannot know at construction what it will be asked to carry.
    channel = QuantumChannel(
        channel_id="qch", delay_ticks=DELAY_TICKS, timing_jitter_stddev_ticks=3.0
    )
    assert channel.timing_jitter_stddev_ticks == 3.0

    with pytest.raises(ValueError, match="not supported for coherent-amplitude"):
        _send(_pulse(), timing_jitter_stddev_ticks=3.0)


def test_noise_models_are_rejected_for_a_bare_amplitude() -> None:
    # A fibre configured for single photons cannot be reused unchanged for
    # pulses; the message has to say what to set instead.
    with pytest.raises(ValueError) as excinfo:
        _send(_pulse(), noise_models=(depolarizing(0.02),))

    message = str(excinfo.value)
    assert "Kraus operators" in message
    assert "phase_noise_stddev_rad" in message
    assert "kind='mode'" in message


def test_a_qubit_carrier_beside_an_amplitude_is_rejected() -> None:
    signal = _pulse(
        state_ref=3,
        state_targets=(SubsystemHandle(label="q0", kind="qubit"),),
    )

    with pytest.raises(ValueError, match="carrier must be one or the other"):
        _send(signal)


def test_a_mode_record_with_no_amplitude_is_rejected() -> None:
    timeline = Timeline(master_seed=1)
    subsystem = SubsystemId("alice:pol:0")
    state_ref = timeline.qstate.prepare("|0>", rep="ket", subsystems=(subsystem,))
    signal = Signal(
        id="orphan",
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="alice",
        state_ref=state_ref,
        state_targets=(SubsystemHandle(label=str(subsystem), kind="mode"),),
    )

    sender, channel, _sink, inbound = _chain()
    channel.bind(binding_context(timeline))
    inbound.transmit(signal, timeline, time=0, source=sender)

    with pytest.raises(ValueError, match="no amplitude occupying it"):
        timeline.run_until(50)


# --------------------------------------------------------------------------
# step 7 reachability
# --------------------------------------------------------------------------


def test_a_mode_record_beside_an_amplitude_survives_loss_and_takes_kraus_noise() -> (
    None
):
    # The decoy-BB84 signal, built by hand because polarization is not
    # implemented. It must reach the amplitude path without rewriting this
    # branch at step 7: alpha attenuates deterministically while the record
    # takes the same Kraus noise a qubit carrier would, and is NOT discarded.
    timeline = Timeline(master_seed=5)
    subsystem = SubsystemId("alice:pol:1")
    state_ref = timeline.qstate.prepare("|+>", rep="ket", subsystems=(subsystem,))
    before = timeline.qstate.size()

    signal = Signal(
        id="polarized-pulse",
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="alice_laser",
        coherent_state=CoherentState.from_mean_photon_number(0.2, phase_rad=pi / 3),
        state_ref=state_ref,
        state_targets=(SubsystemHandle(label=str(subsystem), kind="mode"),),
    )

    sender, channel, sink, inbound = _chain(
        fixed_insertion_loss_db=6.0,
        noise_models=(depolarizing(0.05),),
    )
    channel.bind(binding_context(timeline))
    inbound.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(50)

    arrival, delivered = sink.received[0]
    assert arrival == DELAY_TICKS

    # the amplitude attenuated
    assert delivered.coherent_state.mean_photon_number == pytest.approx(
        channel.survival_probability * 0.2, abs=ATOL
    )
    # the record was not discarded by the loss path
    assert channel.lost_count == 0
    assert channel.attenuated_count == 1
    assert timeline.qstate.size() == before
    assert delivered.state_ref is not None

    # and it took the noise: |+> depolarized is no longer a pure ket
    record = timeline.qstate.record(delivered.state_ref)
    assert record.rep == "density"


def test_binding_declares_a_phase_stream_unconditionally() -> None:
    # Declared even at the 0.0 default, because Timeline.rng refuses a new
    # stream once execution begins. No polarization stream: that arrives at
    # step 7 with the branch that draws from it.
    _channel, _sink, timeline = _send(_pulse())

    timeline.rng("qch", "quantum_channel", "phase")
    with pytest.raises(RuntimeError, match="after freeze"):
        timeline.rng("qch", "quantum_channel", "polarization")


def test_event_time_rejections_leave_construction_alone() -> None:
    # Every rejection above is a runtime error about a payload, so a channel
    # carrying only qstate signals is unaffected by any of this configuration.
    channel = QuantumChannel(
        channel_id="qch",
        delay_ticks=DELAY_TICKS,
        timing_jitter_stddev_ticks=3.0,
        noise_models=(depolarizing(0.02),),
        phase_noise_stddev_rad=0.4,
    )

    assert channel.attenuated_count == 0
    assert channel.phase_noise_stddev_rad == 0.4


@pytest.mark.parametrize("bad", [-0.1, float("nan"), True])
def test_phase_noise_stddev_is_validated_at_construction(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        QuantumChannel(channel_id="qch", phase_noise_stddev_rad=bad)
