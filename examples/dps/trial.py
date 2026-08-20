"""Trial runner for the DPS-QKD transmitter.

This is stage 1 of the DPS build: **devices run and produce reports**. It wires
Alice's weak coherent pulse source to a terminating tap, runs the timeline, and
returns the preparation record plus the counters worth checking before anything
else is built on top.

There is no channel, no interferometer, no detector, and no agent yet. That is
not a simplification of this example -- ``QuantumChannel`` resolves qstate
targets unconditionally and rejects a signal with no ``state_ref``, so a
coherent pulse currently has nowhere else to go. Later steps add the receiver
and the agents to this same package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simyuj.components.connections import PortDelivery, connect_ports
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
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.sinks import JsonlSink

from .configs import DPS_ENCODING_PHASES, DPSAliceSourceConfig
from .helpers import (
    dps_differential_bits,
    dps_phase_histogram,
    dps_slot_period_ticks,
    dps_source_duration_s,
)

ACTION_TAP_PULSE = "tap_pulse"


@dataclass(slots=True)
class PulseTap(Component):
    """Terminating stand-in for the receiver that does not exist yet.

    It owns one quantum ingress port, records what arrived, and does no physics.
    At step 4 the delay interferometer replaces it and this class goes away.

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
    tap = PulseTap()

    connect_ports(
        source.output_port,
        tap.input_port,
        target_action=ACTION_TAP_PULSE,
    )

    source.schedule_start(timeline)
    # The source chains one emission event at a time and stops scheduling past
    # its stop tick, so draining the queue drains the whole train.
    timeline.run_until_empty()

    for sink in sinks:
        sink.flush()

    encoding_indices = [report.encoding_phase_index for report in source.reports]
    carrier_phases = [report.carrier_phase_rad for report in source.reports]
    mean_photon_numbers = [report.mean_photon_number for report in source.reports]
    differential_bits = dps_differential_bits(encoding_indices)

    return {
        # --- counters ---------------------------------------------------
        "configured_slots": num_slots,
        "pulses_emitted": source.pulse_count,
        "pulses_delivered": len(tap.received),
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
        # --- provenance -------------------------------------------------
        "master_seed": master_seed,
        "encoding_alphabet_rad": DPS_ENCODING_PHASES,
        # --- records ----------------------------------------------------
        "encoding_phase_indices": tuple(encoding_indices),
        "alice_differential_bits": differential_bits,
        "arrival_ticks": tuple(time for time, _ in tap.received),
        "reports": tuple(source.reports),
    }


__all__ = ["ACTION_TAP_PULSE", "PulseTap", "run_dps_transmitter_trial"]
