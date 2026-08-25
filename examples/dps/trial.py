"""Trial runner for the DPS-QKD transmitter and receiver optics.

It wires Alice's weak coherent pulse source straight into Bob's delay
interferometer, terminates both interferometer output ports, runs the timeline,
and returns the preparation record, the interference record, and the counters
worth checking before anything else is built on top.

**The receiver optics now close the loop on the encoding.** Alice's differential
bits are decoded from ``encoding_phase_index`` on the control plane; Bob's are
decoded from which output port is bright. The two are computed by entirely
different routes and the trial reports whether they agree, which is the first
point in this build where the protocol says something rather than merely runs.

The chain is the real one: source -> channel -> interferometer -> two taps. The
channel is lossless and phase-noise free by default, so its only effect is a
uniform delay -- every pulse shifts by the same ticks, spacing at BS2 is
unchanged, and the bright-port pattern is identical. ``demo.py`` turns the
physics on: ``--channel-attenuation-db-per-km`` scales both arms equally and the
bits survive, while ``--channel-phase-noise-rad`` puts an independent phase on
every pulse and the bits do not.

Still missing, and each is a later step: no optical detector, so both output
ports end at a stand-in rather than at a click, and Bob's bits are read from the
interferometer's reported intensities; and no agents, so nothing here is a
protocol in the ``control/`` sense -- the decode is a function call in this
module, not knowledge an agent earned from a message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simyuj.components.channels import ACTION_TRANSMIT_QUANTUM, QuantumChannel
from simyuj.components.connections import PortDelivery, connect_ports
from simyuj.components.interferometers import ACTION_INTERFERE, DelayInterferometer
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.components.sources import (
    FixedCarrierPhase,
    FixedIntensity,
    PerPulseRandomCarrierPhase,
    RandomPhaseChoice,
    WeakCoherentPulseSource,
)
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.runtime.binding import BindingContext
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.sinks import JsonlSink

from .configs import (
    DPS_ENCODING_PHASES,
    DPSAliceSourceConfig,
    DPSChannelConfig,
    DPSReceiverConfig,
)
from .helpers import (
    dps_differential_bits,
    dps_optical_differential_bits,
    dps_phase_histogram,
    dps_slot_period_ticks,
    dps_source_duration_s,
)

ACTION_TAP_PULSE = "tap_pulse"


@dataclass(slots=True)
class PulseTap(Component):
    """Terminating stand-in for the optical detector that does not exist yet.

    It owns one quantum ingress port, records what arrived, and does no physics.
    One sits on each interferometer output port. When the optical detector array
    arrives it replaces both and this class goes away.

    Because it does no physics, the intensity on each port is read from the
    interferometer's own reports rather than from what lands here. A tap counts
    slots; it does not decide clicks.

    It is not a test stub. ``tests/support/mock_components/SignalSink`` is the
    stub for that job, and an example must not import from ``tests``; this is
    the example's own placeholder receiver and it is named to say so.
    """

    device_id: str = "alice_tap"

    input_port: Port = field(init=False)
    received: list[tuple[int, Signal]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        """Record one delivered pulse."""
        if event.action != ACTION_TAP_PULSE:
            raise ValueError(
                f"{self.device_id} got unsupported action {event.action!r}"
            )

        delivery = event.payload_ref
        if not isinstance(delivery, PortDelivery):
            raise TypeError(f"{self.device_id} payload_ref must be PortDelivery")
        if delivery.target_port is not self.input_port:
            raise ValueError(f"{self.device_id} delivery arrived on unknown port")
        if not isinstance(delivery.payload, Signal):
            raise TypeError(f"{self.device_id} delivery payload must be Signal")

        self.received.append((timeline.current_time, delivery.payload))


def run_dps_transmitter_trial(
    *,
    master_seed: int = 2026,
    num_slots: int | None = None,
    mean_photon_number: float | None = None,
    randomize_carrier_phase: bool = False,
    channel_attenuation_db_per_km: float | None = None,
    channel_phase_noise_rad: float | None = None,
    source_overrides: dict[str, Any] | None = None,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run one DPS transmitter trial and return its counters and records.

    Parameters
    ----------
    master_seed : int, default=2026
        Seed for every deterministic RNG stream in the run.
    num_slots : int, optional
        Number of pulse slots. Defaults to the config value.
    mean_photon_number : float, optional
        Mean photon number per pulse. Defaults to the config value.
    randomize_carrier_phase : bool, default=False
        Draw an independent carrier phase per pulse instead of holding one.
        This **destroys** the differential-phase encoding and exists so that
        failure is reproducible rather than merely described.
    channel_attenuation_db_per_km : float, optional
        Fibre attenuation. Defaults to the config value, which is ``0.0``. On
        the coherent path this is deterministic: ``alpha -> sqrt(eta) * alpha``
        with no Bernoulli trial and no RNG consumed, so a lossy run still
        replays identically at any seed. Both interferometer arms come from the
        same attenuated pulse, so the encoding survives any amount of it.
    channel_phase_noise_rad : float, optional
        Per-pulse optical phase-noise standard deviation. Defaults to the config
        value, which is ``0.0``. Non-zero **destroys** the encoding, for the
        same reason ``randomize_carrier_phase`` does: the differential phase
        picks up ``theta_n - theta_{n-1}`` from independent draws.
    source_overrides : dict, optional
        Extra keyword arguments passed straight to
        :class:`WeakCoherentPulseSource`.
    log_file : path, optional
        JSONL event log destination. Logging is observational and never changes
        the run.

    Returns
    -------
    dict
        Counters, per-pulse records, and the decoded differential bits.
    """
    config = DPSAliceSourceConfig()
    if num_slots is None:
        num_slots = config.num_slots
    if mean_photon_number is None:
        mean_photon_number = config.mean_photon_number

    session_id = "dps-transmitter"
    sinks: tuple[JsonlSink, ...] = ()
    if log_file is not None:
        sinks = (
            JsonlSink(
                path=Path(log_file),
                session_id=session_id,
                auto_flush=True,
                append=False,
            ),
        )
        logger = SimulationLogger(
            level=LogLevel.DEBUG,
            sinks=sinks,
            session_id=session_id,
        )
    else:
        logger = NullLogger()

    timeline = Timeline(master_seed=master_seed, logger=logger)

    carrier_phase = (
        PerPulseRandomCarrierPhase()
        if randomize_carrier_phase
        else FixedCarrierPhase(config.carrier_phase_rad)
    )

    source_kwargs: dict[str, Any] = {
        "device_id": config.device_id,
        "frequency_hz": config.clock_hz,
        "intensity": FixedIntensity(mean_photon_number),
        "wavelength_nm": config.wavelength_nm,
        "duration_s": dps_source_duration_s(
            clock_hz=config.clock_hz,
            num_slots=num_slots,
        ),
        "carrier_phase": carrier_phase,
        # The alphabet is passed in from the one place it is defined. The
        # decoder in helpers.py is written against that same constant.
        "encoding_phase": RandomPhaseChoice(DPS_ENCODING_PHASES),
        "temporal_mode_sigma_s": config.temporal_mode_sigma_s,
    }
    if source_overrides:
        source_kwargs.update(source_overrides)

    source = WeakCoherentPulseSource(**source_kwargs)

    channel_config = DPSChannelConfig()
    if channel_attenuation_db_per_km is None:
        channel_attenuation_db_per_km = channel_config.attenuation_db_per_km
    if channel_phase_noise_rad is None:
        channel_phase_noise_rad = channel_config.phase_noise_stddev_rad

    channel = QuantumChannel(
        channel_id=channel_config.channel_id,
        length_m=channel_config.length_m,
        attenuation_db_per_km=channel_attenuation_db_per_km,
        phase_noise_stddev_rad=channel_phase_noise_rad,
        # Must stay zero: the coherent path rejects jitter at event time,
        # because a jittered pulse train would change the spacing at BS2 and
        # silently reduce the overlap of every pair.
        timing_jitter_stddev_ticks=0.0,
        session_id=session_id,
    )

    receiver = DPSReceiverConfig()
    # tau = T. Derived from the same clock the source was built from rather than
    # configured separately: the interferometer never sees the clock and cannot
    # check the two against each other, so a second copy could only drift. A
    # mismatch would show up as temporal_overlap collapsing on every slot, which
    # is why that value is reported below.
    interferometer = DelayInterferometer(
        device_id=receiver.device_id,
        # Optics only for now: this example reads its bits from the reported
        # intensities, and wiring the receiver over to real detection is its own
        # change -- it needs an agent to consume the reports. See the run report.
        detectors=None,
        delay_ticks=dps_slot_period_ticks(config.clock_hz),
    )
    tap_0 = PulseTap(device_id=f"{receiver.device_id}_out0_tap")
    tap_1 = PulseTap(device_id=f"{receiver.device_id}_out1_tap")

    connect_ports(
        source.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    connect_ports(
        channel.output_port,
        interferometer.input_port,
        target_action=ACTION_INTERFERE,
    )
    connect_ports(
        interferometer.output_port_0,
        tap_0.input_port,
        target_action=ACTION_TAP_PULSE,
    )
    connect_ports(
        interferometer.output_port_1,
        tap_1.input_port,
        target_action=ACTION_TAP_PULSE,
    )

    binding = BindingContext(timeline=timeline, logger=timeline.logger)
    channel.bind(binding)
    interferometer.bind(binding)
    source.schedule_start(timeline)
    # The source chains one emission event at a time and stops scheduling past
    # its stop tick, and the interferometer's last flush is a scheduled event
    # too, so draining the queue drains the whole train and both end slots.
    timeline.run_until_empty()

    for sink in sinks:
        sink.flush()

    encoding_indices = [report.encoding_phase_index for report in source.reports]
    carrier_phases = [report.carrier_phase_rad for report in source.reports]
    mean_photon_numbers = [report.mean_photon_number for report in source.reports]
    differential_bits = dps_differential_bits(encoding_indices)

    interference = interferometer.reports
    optical_bits = dps_optical_differential_bits(interference)
    # A pulse's arrival at BS1 is the short-arm BS2 tick of the combination that
    # pulse opened, so these are the arrival ticks with no separate counter kept
    # for them. They are emission ticks plus the channel delay, uniformly.
    arrival_ticks = tuple(
        report.short_bs2_tick
        for report in interference
        if report.short_pulse_index is not None
    )
    overlaps = [report.temporal_overlap for report in interference]
    energy_in = sum(report.mean_photon_number_in for report in interference)
    energy_out = sum(
        report.mean_photon_number_0 + report.mean_photon_number_1
        for report in interference
    )

    return {
        # --- counters ---------------------------------------------------
        "configured_slots": num_slots,
        "pulses_emitted": source.pulse_count,
        "pulses_delivered": len(arrival_ticks),
        "preparation_reports": len(source.reports),
        # A coherent source creates no quantum state. This staying at 0 is the
        # sharpest single statement that no photon number was ever sampled.
        "qstate_records": timeline.qstate.size(),
        "differential_bits": len(differential_bits),
        "slot_period_ticks": dps_slot_period_ticks(config.clock_hz),
        # --- preparation ------------------------------------------------
        "mean_photon_number_min": min(mean_photon_numbers, default=None),
        "mean_photon_number_max": max(mean_photon_numbers, default=None),
        "encoding_phase_histogram": dps_phase_histogram(encoding_indices),
        "carrier_phase_distinct_values": len(set(carrier_phases)),
        "carrier_phase_randomized": randomize_carrier_phase,
        "temporal_mode_sigma_s": config.temporal_mode_sigma_s,
        # --- receiver optics ---------------------------------------------
        # N pulses give N+1 combinations: the first pulse's short arm and the
        # last pulse's long arm each meet vacuum and carry no bit.
        "interference_slots": interferometer.interference_count,
        "optical_differential_bits": len(optical_bits),
        # Both are the ideal-device statement. An interferometer that loses
        # light, or a tau that does not match the period, breaks one or the
        # other and nothing else in this trial would notice.
        "interferometer_mu_in": energy_in,
        "interferometer_mu_out": energy_out,
        "temporal_overlap_min": min(overlaps, default=None),
        "held_arms_at_end": interferometer.held_arm_count,
        # --- transport ----------------------------------------------------
        "channel_delay_ticks": channel.resolved_delay_ticks,
        "channel_length_m": channel_config.length_m,
        "channel_attenuation_db_per_km": channel_attenuation_db_per_km,
        "channel_phase_noise_rad": channel_phase_noise_rad,
        "channel_power_transmission": channel.survival_probability,
        # An amplitude is never discarded: attenuation scales it instead, so a
        # totally dark fibre still delivers coherent vacuum on time. Reading
        # channel_lost == 0 as "lossless" is the trap this pair exists to make
        # visible.
        "channel_received": channel.received_count,
        "channel_delivered": channel.delivered_count,
        "channel_lost": channel.lost_count,
        "channel_attenuated": channel.attenuated_count,
        # The point of the receiver optics: Alice's bits come from alphabet
        # indices on the control plane, Bob's from which port is bright. They
        # are computed by different routes and this says whether they agree.
        "optical_bits_match_alice": optical_bits == differential_bits,
        # --- provenance -------------------------------------------------
        "master_seed": master_seed,
        "encoding_alphabet_rad": DPS_ENCODING_PHASES,
        # --- records ----------------------------------------------------
        "encoding_phase_indices": tuple(encoding_indices),
        "alice_differential_bits": differential_bits,
        "bob_optical_differential_bits": optical_bits,
        "arrival_ticks": arrival_ticks,
        "reports": tuple(source.reports),
        "interference_reports": tuple(interference),
    }


__all__ = ["ACTION_TAP_PULSE", "PulseTap", "run_dps_transmitter_trial"]
