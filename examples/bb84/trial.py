"""Trial runner for the single-photon BB84 example.

This file is the network builder. It creates the physical devices, protocol
agents, quantum link, public classical links, event logger, and runtime for one
complete BB84 trial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from simyuj.components import (
    ACTION_TRANSMIT_CLASSICAL,
    ACTION_TRANSMIT_QUANTUM,
    ClassicalChannel,
    GaussianTiming,
    QuantumChannel,
    SinglePhotonSource,
)
from simyuj.components.detectors import (
    ACTION_DETECT_SIGNAL,
    DetectorArray,
    Measure,
    SinglePhotonDetector,
    SinglePhotonDetectorParams,
)
from simyuj.components.detectors.primitives.click import ThresholdClickResolver
from simyuj.control import AGENT_MESSAGE, AGENT_REPORT, SessionRuntime
from simyuj.engine import Timeline
from simyuj.network import Network, Node
from simyuj.primitives.units import seconds_to_ticks
from simyuj.qstate import StateSampler
from simyuj.qstate.noise import depolarizing
from simyuj.signal import EncodingScheme
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.sinks import JsonlSink

from .agents import BB84AliceAgent, BB84BobAgent
from .configs import (
    BB84AliceSourceConfig,
    BB84BobDetectorConfig,
    BB84ClassicalChannelConfig,
    BB84QuantumChannelConfig,
)
from .helpers import (
    BB84_LABELS,
    BB84_STATES,
    bb84_bob_sifting_guard_ticks,
    bb84_slot_assignment_window_ticks,
    bb84_slot_period_ticks,
    bb84_slot_zero_arrival_tick,
    bb84_source_done_delay_ticks,
)


def run_bb84_trial(
    *,
    distance_km: float = 25.0,
    master_seed: int = 2026,
    num_slots: int = 30_000,
    emission_probability: float | None = None,
    depolarizing_probability: float | None = None,
    detector_efficiency: float | None = None,
    source_overrides: dict[str, Any] | None = None,
    quantum_channel_overrides: dict[str, Any] | None = None,
    detector_overrides: dict[str, Any] | None = None,
    classical_channel_overrides: dict[str, Any] | None = None,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run one single-photon BB84 trial.

    The top-level keyword arguments are convenience knobs for common changes.
    For full control, pass override dictionaries whose keys match the config
    dataclasses in ``examples.bb84.configs``. Explicit convenience arguments
    take precedence over matching values inside the override dictionaries.
    """
    source_kwargs: dict[str, Any] = dict(source_overrides or {})
    source_kwargs.setdefault("num_slots", num_slots)
    if emission_probability is not None:
        source_kwargs["emission_probability"] = emission_probability

    quantum_kwargs: dict[str, Any] = dict(quantum_channel_overrides or {})
    quantum_kwargs.setdefault("length_m", distance_km * 1000)
    if depolarizing_probability is not None:
        quantum_kwargs["depolarizing_probability"] = depolarizing_probability

    detector_kwargs: dict[str, Any] = dict(detector_overrides or {})
    if detector_efficiency is not None:
        detector_kwargs["efficiency"] = detector_efficiency

    classical_kwargs: dict[str, Any] = dict(classical_channel_overrides or {})
    classical_kwargs.setdefault("length_m", distance_km * 1000)

    source_config = BB84AliceSourceConfig(**source_kwargs)
    quantum_config = BB84QuantumChannelConfig(**quantum_kwargs)
    classical_config = BB84ClassicalChannelConfig(**classical_kwargs)
    detector_config = BB84BobDetectorConfig(**detector_kwargs)
    session_id = f"bb84_distance_{quantum_config.length_m / 1000:g}_seed_{master_seed}"

    # The source emits the four BB84 states with equal probability. Loss and
    # detector inefficiency are modeled later by the channel and detector.
    sampler = StateSampler(
        states=BB84_STATES,
        probabilities=(0.25, 0.25, 0.25, 0.25),
        rep="ket",
        labels=BB84_LABELS,
    )

    source = SinglePhotonSource(
        device_id=source_config.device_id,
        frequency_hz=source_config.clock_hz,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_probability=source_config.emission_probability,
        wavelength_nm=source_config.wavelength_nm,
        duration_s=source_config.num_slots / source_config.clock_hz,
        sampler=sampler,
        timing_profile=GaussianTiming(
            mean_emission_delay_ticks=0.0,
            emission_delay_stddev_ticks=seconds_to_ticks(
                source_config.timing_jitter_stddev_s
            ),
            max_emission_delay_ticks=seconds_to_ticks(
                source_config.max_timing_jitter_s
            ),
        ),
    )

    quantum_channel = QuantumChannel(
        channel_id=quantum_config.channel_id,
        length_m=quantum_config.length_m,
        propagation_speed_m_per_s=quantum_config.propagation_speed_m_per_s,
        attenuation_db_per_km=quantum_config.attenuation_db_per_km,
        fixed_insertion_loss_db=quantum_config.fixed_insertion_loss_db,
        timing_jitter_stddev_ticks=seconds_to_ticks(
            quantum_config.timing_jitter_stddev_s
        ),
        noise_models=(depolarizing(quantum_config.depolarizing_probability),),
    )

    alice_to_bob_classical = ClassicalChannel(
        channel_id="alice_to_bob_classical",
        length_m=classical_config.length_m,
        fiber_speed_m_per_s=classical_config.fiber_speed_m_per_s,
        loss_probability=classical_config.loss_probability,
        session_id=session_id,
    )

    bob_to_alice_classical = ClassicalChannel(
        channel_id="bob_to_alice_classical",
        length_m=classical_config.length_m,
        fiber_speed_m_per_s=classical_config.fiber_speed_m_per_s,
        loss_probability=classical_config.loss_probability,
        session_id=session_id,
    )

    detector_params = SinglePhotonDetectorParams(
        efficiency=detector_config.efficiency,
        dark_count_rate_hz=detector_config.dark_count_rate_hz,
        dead_time_ticks=seconds_to_ticks(detector_config.dead_time_s),
        jitter_stddev_ticks=seconds_to_ticks(detector_config.jitter_stddev_s),
        p_afterpulse=detector_config.p_afterpulse,
        afterpulse_decay_ticks=seconds_to_ticks(detector_config.afterpulse_decay_s),
        photon_number_resolving=False,
    )

    detectors = (
        SinglePhotonDetector("bob_D_Z0", params=detector_params),
        SinglePhotonDetector("bob_D_Z1", params=detector_params),
        SinglePhotonDetector("bob_D_X0", params=detector_params),
        SinglePhotonDetector("bob_D_X1", params=detector_params),
    )

    detector = DetectorArray(
        device_id=detector_config.device_id,
        detectors=detectors,
        measurement=Measure.random(
            (
                (Measure.basis("z", label="Z"), 0.5),
                (Measure.basis("x", label="X"), 0.5),
            ),
            label="bob_random_bb84_basis",
        ),
        readout={
            "Z": {"0": "bob_D_Z0", "1": "bob_D_Z1"},
            "X": {"+": "bob_D_X0", "-": "bob_D_X1"},
        },
        click_resolver=ThresholdClickResolver(double_click_policy="random"),
        detection_window_ticks=seconds_to_ticks(detector_config.detection_window_s),
        consume_signal=True,
        output_latency_ticks=seconds_to_ticks(detector_config.output_latency_s),
    )

    alice = BB84AliceAgent(
        agent_id="alice_agent",
        node_id="alice",
        source=source,
        quantum_done_delay_ticks=bb84_source_done_delay_ticks(source_config),
    )

    # Bob assigns detections to slots by timing. These values tell the agent
    # what one source clock period means at the detector side of the fiber.
    bob = BB84BobAgent(
        agent_id="bob_agent",
        node_id="bob",
        sifting_guard_ticks=bb84_bob_sifting_guard_ticks(detector_config),
        slot_period_ticks=bb84_slot_period_ticks(source_config),
        slot_zero_arrival_tick=bb84_slot_zero_arrival_tick(quantum_config),
        slot_assignment_window_ticks=bb84_slot_assignment_window_ticks(
            source_config,
            quantum_config,
            detector_config,
        ),
        num_slots=source_config.num_slots,
    )

    network = Network(topology_id="bb84_two_node_network")

    alice_node = Node("alice")
    bob_node = Node("bob")

    alice_classical = alice.enable_classical()
    bob_classical = bob.enable_classical()

    alice_classical.add_route("bob_agent", "to_bob")
    bob_classical.add_route("alice_agent", "to_alice")

    alice_node.add_device("source", source)
    alice_node.add_agent(alice)

    bob_node.add_device("detector", detector)
    bob_node.add_agent(bob)

    network.add_node(alice_node)
    network.add_node(bob_node)

    # The named links describe topology. The wires below describe exactly
    # which output port delivers events to which input port.
    network.add_quantum_link(
        "alice_to_bob_fiber",
        "alice",
        "bob",
        channel=quantum_channel,
    )

    network.add_classical_link(
        "alice_to_bob_public",
        "alice",
        "bob",
        channel=alice_to_bob_classical,
    )

    network.add_classical_link(
        "bob_to_alice_public",
        "bob",
        "alice",
        channel=bob_to_alice_classical,
    )

    network.wire_ports(
        "alice_source_to_quantum_channel",
        source.output_port,
        quantum_channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )

    network.wire_ports(
        "quantum_channel_to_bob_detector",
        quantum_channel.output_port,
        detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )

    network.wire_ports(
        "alice_source_report_to_alice_agent",
        source.report_port,
        alice.report_port,
        target_action=AGENT_REPORT,
    )

    network.wire_ports(
        "bob_detector_report_to_bob_agent",
        detector.output_port,
        bob.report_port,
        target_action=AGENT_REPORT,
    )

    network.wire_ports(
        "alice_agent_to_alice_bob_classical_channel",
        alice_classical.out_port("to_bob"),
        alice_to_bob_classical.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
    )

    network.wire_ports(
        "alice_bob_classical_channel_to_bob_agent",
        alice_to_bob_classical.output_port,
        bob_classical.in_port("from_alice"),
        target_action=AGENT_MESSAGE,
    )

    network.wire_ports(
        "bob_agent_to_bob_alice_classical_channel",
        bob_classical.out_port("to_alice"),
        bob_to_alice_classical.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
    )

    network.wire_ports(
        "bob_alice_classical_channel_to_alice_agent",
        bob_to_alice_classical.output_port,
        alice_classical.in_port("from_bob"),
        target_action=AGENT_MESSAGE,
    )

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

    timeline = Timeline(
        master_seed=master_seed,
        logger=logger,
    )

    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        session_id=session_id,
        start_time=0,
    )

    runtime.bind_all()
    runtime.schedule_agent_starts()

    runtime.run_until_empty()

    for sink in sinks:
        sink.flush()

    length_km = quantum_config.length_m / 1000
    effective_num_slots = source_config.num_slots
    channel_loss_db = quantum_config.attenuation_db_per_km * length_km
    total_optical_loss_db = channel_loss_db + quantum_config.fixed_insertion_loss_db
    channel_transmission = 10 ** (-total_optical_loss_db / 10)

    bob_no_detection_slots = effective_num_slots - len(bob.detections_by_slot_index)

    return {
        "distance_km": length_km,
        "master_seed": master_seed,
        "num_slots": effective_num_slots,
        "emission_probability": source_config.emission_probability,
        "depolarizing_probability": quantum_config.depolarizing_probability,
        "detector_efficiency": detector_config.efficiency,
        "prepared_photons": len(alice.preparations),
        "emitted_slots": len(alice.preparations_by_slot_index),
        "no_emission_slots": (
            effective_num_slots - len(alice.preparations_by_slot_index)
        ),
        "channel_received": quantum_channel.received_count,
        "channel_delivered": quantum_channel.delivered_count,
        "channel_lost": quantum_channel.lost_count,
        "bob_reports": len(bob.detector_reports),
        "bob_detections": len(bob.detections),
        "bob_detected_slots": len(bob.detections_by_slot_index),
        "bob_no_usable_detection_slots": bob_no_detection_slots,
        "bob_failed_reports": len(bob.failed_reports),
        "bob_unassigned_reports": len(bob.unassigned_reports),
        "bob_duplicate_slot_reports": len(bob.duplicate_slot_reports),
        "alice_no_emission_sift_rejections": alice.no_emission_sift_rejections,
        "sifted_bits": len(bob.sifted_bits),
        "sifted_slots": len(bob.sifted_slot_indices),
        "sample_size": len(bob.sample_positions),
        "sample_errors": bob.sample_errors,
        "estimated_qber": bob.estimated_qber,
        "qber_accepted": bob.qber_accepted,
        "cascade_complete": bob.cascade_complete,
        "cascade_parity_requests": bob.cascade_parity_requests,
        "cascade_corrections": bob.cascade_corrections,
        "cascade_leaked_bits": bob.cascade_leaked_bits,
        "reconciled_key_length": len(bob.reconciled_bits),
        "reconciled_bits_equal": alice.reconciled_bits == bob.reconciled_bits,
        "verification_accepted": bob.verification_accepted,
        "verification_leaked_bits": bob.verification_leaked_bits,
        "privacy_complete": bob.privacy_complete,
        "privacy_revealed_bits": bob.privacy_revealed_bits,
        "final_key_length": bob.final_key_length,
        "final_keys_equal": (bob.privacy_complete and alice.final_key == bob.final_key),
        "protocol_complete": (
            bob.privacy_complete and alice.final_key == bob.final_key
        ),
        "alice_abort": alice.aborted_reason,
        "bob_abort": bob.aborted_reason,
        "channel_transmission": channel_transmission,
        "expected_detection_rate_per_prepared": (
            channel_transmission * detector_config.efficiency
        ),
        "observed_detection_rate_per_prepared": (
            len(bob.detections) / len(alice.preparations) if alice.preparations else 0.0
        ),
        "sifting_fraction": (
            len(bob.sifted_bits) / len(bob.detections) if bob.detections else 0.0
        ),
    }


__all__ = ["run_bb84_trial"]
