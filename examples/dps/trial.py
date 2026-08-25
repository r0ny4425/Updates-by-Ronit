"""Trial runner for the DPS-QKD transmitter and receiver optics.

It wires Alice's weak coherent pulse source straight into Bob's delay
interferometer, terminates both interferometer output ports, runs the timeline,
and returns the preparation record, the interference record, and the counters
worth checking before anything else is built on top.

**Bob's bits come from clicks.** Alice's differential bits are decoded from
``encoding_phase_index`` on the control plane; Bob's are decoded from which
detector fired. The two are computed by entirely different routes, and the
distance between them is the QBER -- the number this example exists to produce.

The chain is the real one: source -> channel -> interferometer -> two detectors
-> collector. The channel is lossless and phase-noise free by default and the
dark-count rate is negligible against a 500 ps window, so a default run measures
a QBER of exactly zero. That is a statement about the wiring, not a claim about
physics: what limits it is dark counts, and there are effectively none until the
flag turns them up.

Each imperfection degrades the key in a different direction, which is what makes
them separable in the summary:

- attenuation costs clicks and not QBER; both arms scale together
- phase noise costs QBER and not clicks; light moves between ports, none is lost
- dark counts cost QBER and *add* clicks; a port fires that the light did not

``interferometer.reports`` still carries the exact per-port intensities, so the
ideal readout -- ``mu_0 > mu_1``, no detector, no statistics -- remains available
as the reference the detector is judged against. It is reported alongside, not
instead of, the detected bits.

Still a later step: there is no classical channel and no agents, so nothing here
is a protocol in the ``control/`` sense. Alice's bits are read from her own
reports in this module rather than learned from a message, and there is no
sifting exchange, error correction or privacy amplification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simyuj.components.channels import ACTION_TRANSMIT_QUANTUM, QuantumChannel
from simyuj.components.connections import PortDelivery, connect_ports
from simyuj.components.detectors import (
    FLAG_DOUBLE_CLICK,
    DetectionReport,
    SinglePhotonDetector,
    SinglePhotonDetectorParams,
)
from simyuj.components.detectors.primitives.click import ThresholdClickResolver
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
from simyuj.primitives.units import seconds_to_ticks
from simyuj.runtime.binding import BindingContext
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.sinks import JsonlSink

from .configs import (
    DPS_ENCODING_PHASES,
    DPSAliceSourceConfig,
    DPSChannelConfig,
    DPSDetectorConfig,
    DPSReceiverConfig,
)
from .helpers import (
    dps_detected_bit,
    dps_differential_bit,
    dps_differential_bits,
    dps_optical_differential_bits,
    dps_phase_histogram,
    dps_slot_arms,
    dps_slot_period_ticks,
    dps_source_duration_s,
)

ACTION_TAP_PULSE = "tap_pulse"
ACTION_RECEIVE_DETECTION = "receive_detection"


@dataclass(slots=True)
class PulseTap(Component):
    """Terminator for one interferometer optical port.

    It owns one quantum ingress port, records what arrived, and does no physics.

    **It is no longer the receiver.** It used to stand in for the detector that
    did not exist, and Bob's bits were read from the interferometer's own
    intensities; the detectors are inside the interferometer now and
    ``DetectionCollector`` reads the clicks. What is left is termination and
    inspection: the device always puts light on *both* output ports -- the
    destructive one carrying nearly nothing is a result, not an absence -- and
    ``_resolve`` requires each connection, so an unwired port is an error.

    Keeping the ports wired also keeps the exact amplitudes reachable, which is
    what lets a test compare what the detector decided against what the optics
    actually delivered.
    """

    device_id: str = "tap"

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


@dataclass(slots=True)
class DetectionCollector(Component):
    """Terminating consumer of the interferometer's ``detection`` port.

    It owns one classical ingress port and records every ``DetectionReport`` the
    receiver produces, in slot order. It decides nothing: the two detectors
    inside the interferometer already turned each output amplitude into a click
    or a silence, and the decode is a pure function over what lands here.

    **It is not a ``NodeAgent``, and that is the scope of this example.** There
    is no classical channel, no sifting exchange and no error correction here --
    Alice's bits are read from her preparation reports in the same process. Step
    6 turns this into a real two-agent protocol; until then this is a receiver
    that collects, not one that negotiates.

    It replaces the two ``PulseTap`` instances that stood in for a detector.
    The interferometer's optical ports stay wired to nothing by default: they
    remain an inspection point, and the run does not need them.
    """

    device_id: str = "bob_receiver"

    input_port: Port = field(init=False)
    reports: list[DetectionReport] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        """Record one slot decision.

        Type-checks the payload the way ``BB84BobAgent.on_report`` does: a
        report of the wrong type is a wiring error and must raise here rather
        than surface later as a missing counter.
        """
        if event.action != ACTION_RECEIVE_DETECTION:
            raise ValueError(
                f"{self.device_id} got unsupported action {event.action!r}"
            )

        delivery = event.payload_ref
        if not isinstance(delivery, PortDelivery):
            raise TypeError(f"{self.device_id} payload_ref must be PortDelivery")
        if delivery.target_port is not self.input_port:
            raise ValueError(f"{self.device_id} delivery arrived on unknown port")
        if not isinstance(delivery.payload, DetectionReport):
            raise TypeError(
                f"{self.device_id} expected a DetectionReport, got "
                f"{type(delivery.payload).__name__}"
            )

        self.reports.append(delivery.payload)


@dataclass(frozen=True)
class DPSSlotOutcome:
    """One interference slot after the receiver has read it.

    Parameters
    ----------
    interference_index : int
        The interferometer's own combination counter, and the join key back to
        its ``InterferenceReport``.
    long_pulse_index, short_pulse_index : int or None
        Source pulses of the two arms that met at BS2, previous first. ``None``
        marks the vacuum arm of an edge slot.
    kind : {"edge", "click", "no_click", "double_click"}
        What the slot turned out to be. Only ``"click"`` -- and a
        ``"double_click"`` under a policy that still names a port -- carries a
        bit.
    alice_bit, bob_bit : int or None
        The differential bit each side holds for this slot, or ``None`` when the
        slot carries none.
    """

    interference_index: int
    long_pulse_index: int | None
    short_pulse_index: int | None
    kind: str
    alice_bit: int | None
    bob_bit: int | None


def read_detection_slots(
    reports: Sequence[DetectionReport],
    encoding_indices: Sequence[int],
) -> tuple[DPSSlotOutcome, ...]:
    """Turn the receiver's reports into per-slot outcomes with both bits.

    Parameters
    ----------
    reports : sequence of DetectionReport
        Every slot the detection port emitted, in order.
    encoding_indices : sequence of int
        Alice's ``encoding_phase_index`` per pulse, in pulse order.

    Returns
    -------
    tuple[DPSSlotOutcome, ...]
        One entry per report, including the ones that carry no bit.

    Notes
    -----
    **The edge slots are dropped here, explicitly and with a counter.** The
    first pulse's short arm and the flushed last long arm each met vacuum, so
    both ports carry equal light and the slot is information-free. They arrive
    as ordinary reports -- the interferometer does not special-case them, and
    neither should anything upstream of this function. Two slots out of N+1 is
    vanishing in a long run and still has to be an explicit drop, or the bit
    stream ends up one position out of step with Alice's.

    **Alice's bit is joined per slot, not zipped positionally.** Each slot names
    the two pulses that made it, so the pairing survives a dropped or reordered
    slot; a positional zip would silently shift every later bit if one slot went
    missing.

    A double click is counted separately from a no-click even when the policy
    makes both unsuccessful, because they are different physics: one is two
    detectors firing and the other is neither.
    """
    outcomes: list[DPSSlotOutcome] = []

    for report in reports:
        meta = dict(report.meta)
        long_index, short_index = dps_slot_arms(report)
        interference_index = meta["interference_index"]

        if long_index is None or short_index is None:
            kind, alice_bit, bob_bit = "edge", None, None
        else:
            alice_bit = dps_differential_bit(
                encoding_indices[long_index - 1],
                encoding_indices[short_index - 1],
            )
            if FLAG_DOUBLE_CLICK in report.flags:
                kind = "double_click"
                bob_bit = dps_detected_bit(report.outcome) if report.success else None
            elif report.success:
                kind, bob_bit = "click", dps_detected_bit(report.outcome)
            else:
                kind, bob_bit = "no_click", None

            if bob_bit is None:
                alice_bit = None

        outcomes.append(
            DPSSlotOutcome(
                interference_index=interference_index,
                long_pulse_index=long_index,
                short_pulse_index=short_index,
                kind=kind,
                alice_bit=alice_bit,
                bob_bit=bob_bit,
            )
        )

    return tuple(outcomes)


def run_dps_transmitter_trial(
    *,
    master_seed: int = 2026,
    num_slots: int | None = None,
    mean_photon_number: float | None = None,
    randomize_carrier_phase: bool = False,
    channel_attenuation_db_per_km: float | None = None,
    channel_phase_noise_rad: float | None = None,
    detector_efficiency: float | None = None,
    dark_count_rate_hz: float | None = None,
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
    detector_efficiency : float, optional
        Quantum efficiency of both detectors. Defaults to the config value.
        This is **not** the probability of a click: for a coherent pulse that is
        ``1 - exp(-eta * mu)``, so even ``1.0`` leaves a bright port silent on
        most slots at ``mu = 0.2``.
    dark_count_rate_hz : float, optional
        Dark-count rate of both detectors. Defaults to the config value. A dark
        count fires a port the light did not, so unlike loss it produces a wrong
        bit rather than a missing one, and it is the only thing that puts a
        floor under the QBER of an otherwise ideal run.
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
    detector_config = DPSDetectorConfig()
    if detector_efficiency is None:
        detector_efficiency = detector_config.efficiency
    if dark_count_rate_hz is None:
        dark_count_rate_hz = detector_config.dark_count_rate_hz

    detector_params = SinglePhotonDetectorParams(
        efficiency=detector_efficiency,
        dark_count_rate_hz=dark_count_rate_hz,
        dead_time_ticks=seconds_to_ticks(detector_config.dead_time_s),
        jitter_stddev_ticks=seconds_to_ticks(detector_config.jitter_stddev_s),
        p_afterpulse=detector_config.p_afterpulse,
        afterpulse_decay_ticks=seconds_to_ticks(detector_config.afterpulse_decay_s),
        photon_number_resolving=False,
    )
    # Index 0 reads out_0, index 1 reads out_1. Two instances rather than one
    # shared object: dead time and afterpulsing are per-channel state, and one
    # detector on both ports would couple them.
    detectors = (
        SinglePhotonDetector(f"{receiver.device_id}_d0", params=detector_params),
        SinglePhotonDetector(f"{receiver.device_id}_d1", params=detector_params),
    )

    # tau = T. Derived from the same clock the source was built from rather than
    # configured separately: the interferometer never sees the clock and cannot
    # check the two against each other, so a second copy could only drift. A
    # mismatch would show up as temporal_overlap collapsing on every slot, which
    # is why that value is reported below.
    interferometer = DelayInterferometer(
        device_id=receiver.device_id,
        detectors=detectors,
        delay_ticks=dps_slot_period_ticks(config.clock_hz),
        detection_window_ticks=seconds_to_ticks(detector_config.detection_window_s),
        click_resolver=ThresholdClickResolver(
            double_click_policy=detector_config.double_click_policy,
        ),
    )
    collector = DetectionCollector(device_id="bob_receiver")
    # Both optical ports must still be terminated: the interferometer always
    # puts light on both and `_resolve` requires each connection, so an
    # unwired port is an error rather than an absence.
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
        interferometer.detection_port,
        collector.input_port,
        target_action=ACTION_RECEIVE_DETECTION,
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

    # --- what the detectors decided -------------------------------------
    slots = read_detection_slots(collector.reports, encoding_indices)
    kinds = [slot.kind for slot in slots]
    sifted = [slot for slot in slots if slot.bob_bit is not None]
    errors = sum(slot.alice_bit != slot.bob_bit for slot in sifted)
    raw_clicks = sum(len(report.raw_clicks) for report in collector.reports)

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
        # --- detection ---------------------------------------------------
        # Every slot lands in exactly one of these four, and they sum to
        # interference_slots. A slot that carries no bit is still a slot.
        "detection_slots": len(slots),
        "edge_slots_dropped": kinds.count("edge"),
        "slots_with_click": kinds.count("click"),
        "slots_no_click": kinds.count("no_click"),
        "slots_double_click": kinds.count("double_click"),
        # Slots that produced a usable bit. There is no public sifting exchange
        # here -- Alice's bits are read in-process -- so this is the raw sifted
        # set, before any error correction or privacy amplification.
        "sifted_bits": len(sifted),
        "sifted_errors": errors,
        # The number that matters. Alice decoded from alphabet indices on the
        # control plane, Bob from which port fired; two routes, no shared step.
        "qber": (errors / len(sifted)) if sifted else None,
        # End to end, and deliberately per *pulse* rather than per slot: it is
        # the fraction of light Alice sent that Bob turned into a detection
        # event, which is what a link budget is about.
        "clicks_per_pulse": (
            raw_clicks / source.pulse_count if source.pulse_count else None
        ),
        "raw_clicks": raw_clicks,
        "detector_efficiency": detector_efficiency,
        "dark_count_rate_hz": dark_count_rate_hz,
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
        "detection_reports": tuple(collector.reports),
        "detection_slots_detail": slots,
        "arrival_ticks": arrival_ticks,
        "reports": tuple(source.reports),
        "interference_reports": tuple(interference),
    }


__all__ = ["ACTION_TAP_PULSE", "PulseTap", "run_dps_transmitter_trial"]
