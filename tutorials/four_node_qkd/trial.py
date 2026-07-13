"""Build and run the concurrent four-node BB84 and E91 tutorial network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.bb84.agents import BB84AliceAgent, BB84BobAgent
from examples.bb84.helpers import (
    BB84_LABELS,
    BB84_STATES,
    bb84_bob_sifting_guard_ticks,
    bb84_slot_assignment_window_ticks,
    bb84_slot_period_ticks,
    bb84_slot_zero_arrival_tick,
    bb84_source_done_delay_ticks,
)
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
from simyuj.components.sources import EntangledPairSource
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

from .configs import (
    E91BasisSetting,
    E91DetectorConfig,
    FourNodeQKDConfig,
    PublicClassicalChannelConfig,
)
from .e91_agents import E91BAgent, E91DAgent, E91SourceAgent
from .helpers import (
    e91_receiver_guard_ticks,
    e91_source_done_delay_ticks,
    photon_basis_from_e91_angle,
    validate_e91_config,
)


@dataclass(slots=True)
class FourNodeTrialArtifacts:
    """Internal handles retained for integration tests and summary building."""

    config: FourNodeQKDConfig
    timeline: Timeline
    network: Network
    runtime: SessionRuntime
    a_bb84: BB84AliceAgent
    b_bb84: BB84BobAgent
    c_e91: E91SourceAgent
    b_e91: E91BAgent
    d_e91: E91DAgent
    bb84_source: SinglePhotonSource
    e91_source: EntangledPairSource
    b_bb84_detector: DetectorArray
    b_e91_detector: DetectorArray
    d_e91_detector: DetectorArray
    quantum_channels: dict[str, QuantumChannel]
    classical_channels: dict[str, ClassicalChannel]
    sinks: tuple[JsonlSink, ...]


def _make_classical_channel(
    channel_id: str,
    config: PublicClassicalChannelConfig,
    *,
    session_id: str,
) -> ClassicalChannel:
    return ClassicalChannel(
        channel_id=channel_id,
        length_m=config.length_m,
        fiber_speed_m_per_s=config.fiber_speed_m_per_s,
        loss_probability=config.loss_probability,
        session_id=session_id,
    )


def _make_e91_detector(
    config: E91DetectorConfig,
    settings: tuple[E91BasisSetting, ...],
) -> DetectorArray:
    detector_zero_id = f"{config.device_id}_D0"
    detector_one_id = f"{config.device_id}_D1"
    params = SinglePhotonDetectorParams(
        efficiency=config.efficiency,
        dark_count_rate_hz=config.dark_count_rate_hz,
        dead_time_ticks=seconds_to_ticks(config.dead_time_s),
        jitter_stddev_ticks=seconds_to_ticks(config.jitter_stddev_s),
        p_afterpulse=config.p_afterpulse,
        afterpulse_decay_ticks=seconds_to_ticks(config.afterpulse_decay_s),
        photon_number_resolving=False,
    )
    bases = {
        setting.label: photon_basis_from_e91_angle(
            setting.label,
            setting.angle_rad,
        )
        for setting in settings
    }
    return DetectorArray(
        device_id=config.device_id,
        detectors=(
            SinglePhotonDetector(detector_zero_id, params=params),
            SinglePhotonDetector(detector_one_id, params=params),
        ),
        measurement=Measure.random(
            tuple(
                (
                    Measure.basis(bases[setting.label], label=setting.label),
                    setting.probability,
                )
                for setting in settings
            ),
            label=f"{config.device_id}_random_basis",
        ),
        readout={
            setting.label: {"0": detector_zero_id, "1": detector_one_id}
            for setting in settings
        },
        click_resolver=ThresholdClickResolver(double_click_policy="random"),
        detection_window_ticks=seconds_to_ticks(config.detection_window_s),
        consume_signal=True,
        output_latency_ticks=seconds_to_ticks(config.output_latency_s),
    )


def _validate_config(config: FourNodeQKDConfig) -> None:
    validate_e91_config(config.e91_postprocessing)
    classical_losses = (
        config.bb84_classical.loss_probability,
        config.c_to_b_classical.loss_probability,
        config.c_to_d_classical.loss_probability,
        config.b_to_d_classical.loss_probability,
        config.d_to_b_classical.loss_probability,
    )
    if any(loss != 0.0 for loss in classical_losses):
        raise ValueError(
            "the tutorial requires reliable public channels; "
            "retransmission is out of scope"
        )


def build_four_node_qkd_trial(
    config: FourNodeQKDConfig | None = None,
    *,
    log_file: str | Path | None = None,
) -> FourNodeTrialArtifacts:
    """Build, but do not run, one concurrent physical QKD scenario."""
    config = FourNodeQKDConfig() if config is None else config
    if not isinstance(config, FourNodeQKDConfig):
        raise TypeError("config must be FourNodeQKDConfig or None")
    _validate_config(config)

    bb84_sampler = StateSampler(
        states=BB84_STATES,
        probabilities=(0.25, 0.25, 0.25, 0.25),
        rep="ket",
        labels=BB84_LABELS,
    )
    bb84_source = SinglePhotonSource(
        device_id=config.bb84_source.device_id,
        frequency_hz=config.bb84_source.clock_hz,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_probability=config.bb84_source.emission_probability,
        wavelength_nm=config.bb84_source.wavelength_nm,
        duration_s=config.bb84_source.num_slots / config.bb84_source.clock_hz,
        sampler=bb84_sampler,
        timing_profile=GaussianTiming(
            mean_emission_delay_ticks=0.0,
            emission_delay_stddev_ticks=seconds_to_ticks(
                config.bb84_source.timing_jitter_stddev_s
            ),
            max_emission_delay_ticks=seconds_to_ticks(
                config.bb84_source.max_timing_jitter_s
            ),
        ),
    )
    a_to_b_quantum = QuantumChannel(
        channel_id=config.bb84_quantum.channel_id,
        length_m=config.bb84_quantum.length_m,
        propagation_speed_m_per_s=config.bb84_quantum.propagation_speed_m_per_s,
        attenuation_db_per_km=config.bb84_quantum.attenuation_db_per_km,
        fixed_insertion_loss_db=config.bb84_quantum.fixed_insertion_loss_db,
        timing_jitter_stddev_ticks=seconds_to_ticks(
            config.bb84_quantum.timing_jitter_stddev_s
        ),
        noise_models=(depolarizing(config.bb84_quantum.depolarizing_probability),),
    )

    bb84_detector_params = SinglePhotonDetectorParams(
        efficiency=config.bb84_detector.efficiency,
        dark_count_rate_hz=config.bb84_detector.dark_count_rate_hz,
        dead_time_ticks=seconds_to_ticks(config.bb84_detector.dead_time_s),
        jitter_stddev_ticks=seconds_to_ticks(config.bb84_detector.jitter_stddev_s),
        p_afterpulse=config.bb84_detector.p_afterpulse,
        afterpulse_decay_ticks=seconds_to_ticks(
            config.bb84_detector.afterpulse_decay_s
        ),
        photon_number_resolving=False,
    )
    b_bb84_detector = DetectorArray(
        device_id=config.bb84_detector.device_id,
        detectors=(
            SinglePhotonDetector("b_bb84_D_Z0", params=bb84_detector_params),
            SinglePhotonDetector("b_bb84_D_Z1", params=bb84_detector_params),
            SinglePhotonDetector("b_bb84_D_X0", params=bb84_detector_params),
            SinglePhotonDetector("b_bb84_D_X1", params=bb84_detector_params),
        ),
        measurement=Measure.random(
            (
                (Measure.basis("z", label="Z"), 0.5),
                (Measure.basis("x", label="X"), 0.5),
            ),
            label="b_random_bb84_basis",
        ),
        readout={
            "Z": {"0": "b_bb84_D_Z0", "1": "b_bb84_D_Z1"},
            "X": {"+": "b_bb84_D_X0", "-": "b_bb84_D_X1"},
        },
        click_resolver=ThresholdClickResolver(double_click_policy="random"),
        detection_window_ticks=seconds_to_ticks(
            config.bb84_detector.detection_window_s
        ),
        consume_signal=True,
        output_latency_ticks=seconds_to_ticks(config.bb84_detector.output_latency_s),
    )

    e91_sampler = StateSampler(
        states=("psi-",),
        probabilities=(1.0,),
        rep="ket",
        labels=("psi-",),
    )
    e91_source = EntangledPairSource(
        device_id=config.e91_source.device_id,
        frequency_hz=config.e91_source.clock_hz,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_probability=config.e91_source.emission_probability,
        wavelength_nm=config.e91_source.wavelength_nm,
        duration_s=config.e91_source.num_slots / config.e91_source.clock_hz,
        sampler=e91_sampler,
        timing_profile=GaussianTiming(
            mean_emission_delay_ticks=0.0,
            emission_delay_stddev_ticks=seconds_to_ticks(
                config.e91_source.timing_jitter_stddev_s
            ),
            max_emission_delay_ticks=seconds_to_ticks(
                config.e91_source.max_timing_jitter_s
            ),
        ),
    )
    c_to_b_quantum = QuantumChannel(
        channel_id=config.e91_c_to_b.channel_id,
        length_m=config.e91_c_to_b.length_m,
        propagation_speed_m_per_s=config.e91_c_to_b.propagation_speed_m_per_s,
        attenuation_db_per_km=config.e91_c_to_b.attenuation_db_per_km,
        fixed_insertion_loss_db=config.e91_c_to_b.fixed_insertion_loss_db,
        timing_jitter_stddev_ticks=seconds_to_ticks(
            config.e91_c_to_b.timing_jitter_stddev_s
        ),
        noise_models=(depolarizing(config.e91_c_to_b.depolarizing_probability),),
    )
    c_to_d_quantum = QuantumChannel(
        channel_id=config.e91_c_to_d.channel_id,
        length_m=config.e91_c_to_d.length_m,
        propagation_speed_m_per_s=config.e91_c_to_d.propagation_speed_m_per_s,
        attenuation_db_per_km=config.e91_c_to_d.attenuation_db_per_km,
        fixed_insertion_loss_db=config.e91_c_to_d.fixed_insertion_loss_db,
        timing_jitter_stddev_ticks=seconds_to_ticks(
            config.e91_c_to_d.timing_jitter_stddev_s
        ),
        noise_models=(depolarizing(config.e91_c_to_d.depolarizing_probability),),
    )
    b_e91_detector = _make_e91_detector(
        config.e91_b_detector,
        config.e91_postprocessing.b_settings,
    )
    d_e91_detector = _make_e91_detector(
        config.e91_d_detector,
        config.e91_postprocessing.d_settings,
    )

    a_bb84 = BB84AliceAgent(
        agent_id="a_bb84_agent",
        node_id="A",
        source=bb84_source,
        quantum_done_delay_ticks=bb84_source_done_delay_ticks(config.bb84_source),
        peer_id="b_bb84_agent",
        out_port="to_b_bb84",
        qber_settings=config.bb84_qber,
        privacy_settings=config.bb84_privacy,
    )
    b_bb84 = BB84BobAgent(
        agent_id="b_bb84_agent",
        node_id="B",
        sifting_guard_ticks=bb84_bob_sifting_guard_ticks(config.bb84_detector),
        slot_period_ticks=bb84_slot_period_ticks(config.bb84_source),
        slot_zero_arrival_tick=bb84_slot_zero_arrival_tick(config.bb84_quantum),
        slot_assignment_window_ticks=bb84_slot_assignment_window_ticks(
            config.bb84_source,
            config.bb84_quantum,
            config.bb84_detector,
        ),
        num_slots=config.bb84_source.num_slots,
        peer_id="a_bb84_agent",
        out_port="to_a_bb84",
        qber_settings=config.bb84_qber,
        cascade_settings=config.bb84_cascade,
        verification_settings=config.bb84_verification,
        privacy_settings=config.bb84_privacy,
    )
    c_e91 = E91SourceAgent(
        agent_id="c_e91_source_agent",
        node_id="C",
        source=e91_source,
        frame_done_delay_ticks=e91_source_done_delay_ticks(config.e91_source),
    )
    b_e91 = E91BAgent(
        agent_id="b_e91_agent",
        node_id="B",
        postprocessing=config.e91_postprocessing,
        readiness_guard_ticks=e91_receiver_guard_ticks(
            quantum=config.e91_c_to_b,
            classical=config.c_to_b_classical,
            detector=config.e91_b_detector,
        ),
    )
    d_e91 = E91DAgent(
        agent_id="d_e91_agent",
        node_id="D",
        postprocessing=config.e91_postprocessing,
        readiness_guard_ticks=e91_receiver_guard_ticks(
            quantum=config.e91_c_to_d,
            classical=config.c_to_d_classical,
            detector=config.e91_d_detector,
        ),
    )

    a_to_b_classical_config = PublicClassicalChannelConfig(
        length_m=config.bb84_classical.length_m,
        fiber_speed_m_per_s=config.bb84_classical.fiber_speed_m_per_s,
        loss_probability=config.bb84_classical.loss_probability,
    )
    classical_channels = {
        "a_to_b_bb84": _make_classical_channel(
            "a_to_b_bb84_classical",
            a_to_b_classical_config,
            session_id=config.session_id,
        ),
        "b_to_a_bb84": _make_classical_channel(
            "b_to_a_bb84_classical",
            a_to_b_classical_config,
            session_id=config.session_id,
        ),
        "c_to_b_e91": _make_classical_channel(
            "c_to_b_e91_classical",
            config.c_to_b_classical,
            session_id=config.session_id,
        ),
        "c_to_d_e91": _make_classical_channel(
            "c_to_d_e91_classical",
            config.c_to_d_classical,
            session_id=config.session_id,
        ),
        "b_to_d_e91": _make_classical_channel(
            "b_to_d_e91_classical",
            config.b_to_d_classical,
            session_id=config.session_id,
        ),
        "d_to_b_e91": _make_classical_channel(
            "d_to_b_e91_classical",
            config.d_to_b_classical,
            session_id=config.session_id,
        ),
    }

    network = Network("four_node_concurrent_bb84_e91")
    nodes = {node_id: Node(node_id) for node_id in ("A", "B", "C", "D")}

    a_classical = a_bb84.enable_classical()
    b_bb84_classical = b_bb84.enable_classical()
    c_classical = c_e91.enable_classical()
    b_e91_classical = b_e91.enable_classical()
    d_e91_classical = d_e91.enable_classical()
    a_classical.add_route("b_bb84_agent", "to_b_bb84")
    b_bb84_classical.add_route("a_bb84_agent", "to_a_bb84")
    c_classical.add_route("b_e91_agent", "to_b_e91")
    c_classical.add_route("d_e91_agent", "to_d_e91")
    b_e91_classical.add_route("d_e91_agent", "to_d_e91")
    d_e91_classical.add_route("b_e91_agent", "to_b_e91")

    nodes["A"].add_device("bb84_source", bb84_source)
    nodes["A"].add_agent(a_bb84)
    nodes["B"].add_device("bb84_detector", b_bb84_detector)
    nodes["B"].add_device("e91_detector", b_e91_detector)
    nodes["B"].add_agent(b_bb84)
    nodes["B"].add_agent(b_e91)
    nodes["C"].add_device("e91_pair_source", e91_source)
    nodes["C"].add_agent(c_e91)
    nodes["D"].add_device("e91_detector", d_e91_detector)
    nodes["D"].add_agent(d_e91)
    for node in nodes.values():
        network.add_node(node)

    quantum_channels = {
        "A_to_B_BB84": a_to_b_quantum,
        "C_to_B_E91": c_to_b_quantum,
        "C_to_D_E91": c_to_d_quantum,
    }
    network.add_quantum_link(
        "A_to_B_BB84",
        "A",
        "B",
        channel=a_to_b_quantum,
    )
    network.add_quantum_link(
        "C_to_B_E91",
        "C",
        "B",
        channel=c_to_b_quantum,
    )
    network.add_quantum_link(
        "C_to_D_E91",
        "C",
        "D",
        channel=c_to_d_quantum,
    )

    classical_link_specs = (
        ("A_to_B_BB84_public", "A", "B", "a_to_b_bb84"),
        ("B_to_A_BB84_public", "B", "A", "b_to_a_bb84"),
        ("C_to_B_E91_frame", "C", "B", "c_to_b_e91"),
        ("C_to_D_E91_frame", "C", "D", "c_to_d_e91"),
        ("B_to_D_E91_public", "B", "D", "b_to_d_e91"),
        ("D_to_B_E91_public", "D", "B", "d_to_b_e91"),
    )
    for link_id, source_node, target_node, channel_key in classical_link_specs:
        network.add_classical_link(
            link_id,
            source_node,
            target_node,
            channel=classical_channels[channel_key],
        )

    network.wire_ports(
        "a_bb84_source_to_quantum",
        bb84_source.output_port,
        a_to_b_quantum.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    network.wire_ports(
        "a_to_b_quantum_to_bb84_detector",
        a_to_b_quantum.output_port,
        b_bb84_detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )
    network.wire_ports(
        "a_bb84_source_report_to_agent",
        bb84_source.report_port,
        a_bb84.report_port,
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "b_bb84_detector_report_to_agent",
        b_bb84_detector.output_port,
        b_bb84.report_port,
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "c_e91_left_to_b_quantum",
        e91_source.left_output_port,
        c_to_b_quantum.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    network.wire_ports(
        "c_e91_right_to_d_quantum",
        e91_source.right_output_port,
        c_to_d_quantum.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    network.wire_ports(
        "c_to_b_quantum_to_e91_detector",
        c_to_b_quantum.output_port,
        b_e91_detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )
    network.wire_ports(
        "c_to_d_quantum_to_e91_detector",
        c_to_d_quantum.output_port,
        d_e91_detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )
    network.wire_ports(
        "c_e91_source_report_to_agent",
        e91_source.report_port,
        c_e91.report_port,
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "b_e91_detector_report_to_agent",
        b_e91_detector.output_port,
        b_e91.report_port,
        target_action=AGENT_REPORT,
    )
    network.wire_ports(
        "d_e91_detector_report_to_agent",
        d_e91_detector.output_port,
        d_e91.report_port,
        target_action=AGENT_REPORT,
    )

    classical_wire_specs = (
        (
            "a_agent_to_b_channel",
            a_classical.out_port("to_b_bb84"),
            "a_to_b_bb84",
            b_bb84_classical.in_port("from_a_bb84"),
        ),
        (
            "b_agent_to_a_channel",
            b_bb84_classical.out_port("to_a_bb84"),
            "b_to_a_bb84",
            a_classical.in_port("from_b_bb84"),
        ),
        (
            "c_agent_to_b_frame_channel",
            c_classical.out_port("to_b_e91"),
            "c_to_b_e91",
            b_e91_classical.in_port("from_c_e91"),
        ),
        (
            "c_agent_to_d_frame_channel",
            c_classical.out_port("to_d_e91"),
            "c_to_d_e91",
            d_e91_classical.in_port("from_c_e91"),
        ),
        (
            "b_agent_to_d_public_channel",
            b_e91_classical.out_port("to_d_e91"),
            "b_to_d_e91",
            d_e91_classical.in_port("from_b_e91"),
        ),
        (
            "d_agent_to_b_public_channel",
            d_e91_classical.out_port("to_b_e91"),
            "d_to_b_e91",
            b_e91_classical.in_port("from_d_e91"),
        ),
    )
    for prefix, source_port, channel_key, target_port in classical_wire_specs:
        channel = classical_channels[channel_key]
        network.wire_ports(
            f"{prefix}_input",
            source_port,
            channel.input_port,
            target_action=ACTION_TRANSMIT_CLASSICAL,
        )
        network.wire_ports(
            f"{prefix}_output",
            channel.output_port,
            target_port,
            target_action=AGENT_MESSAGE,
        )

    sinks: tuple[JsonlSink, ...] = ()
    logger: NullLogger | SimulationLogger
    if log_file is None:
        logger = NullLogger()
    else:
        sinks = (
            JsonlSink(
                path=Path(log_file),
                session_id=config.session_id,
                auto_flush=True,
                append=False,
            ),
        )
        logger = SimulationLogger(
            level=LogLevel.DEBUG,
            sinks=sinks,
            session_id=config.session_id,
        )

    timeline = Timeline(master_seed=config.master_seed, logger=logger)
    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        session_id=config.session_id,
        start_time=0,
    )
    return FourNodeTrialArtifacts(
        config=config,
        timeline=timeline,
        network=network,
        runtime=runtime,
        a_bb84=a_bb84,
        b_bb84=b_bb84,
        c_e91=c_e91,
        b_e91=b_e91,
        d_e91=d_e91,
        bb84_source=bb84_source,
        e91_source=e91_source,
        b_bb84_detector=b_bb84_detector,
        b_e91_detector=b_e91_detector,
        d_e91_detector=d_e91_detector,
        quantum_channels=quantum_channels,
        classical_channels=classical_channels,
        sinks=sinks,
    )


def _channel_summary(channel: Any) -> dict[str, int | float]:
    summary: dict[str, int | float] = {
        "received": channel.received_count,
        "delivered": channel.delivered_count,
        "delivery_fraction": (
            channel.delivered_count / channel.received_count
            if channel.received_count
            else 0.0
        ),
    }
    if hasattr(channel, "lost_count"):
        summary["lost"] = channel.lost_count
    if hasattr(channel, "dropped_count"):
        summary["dropped"] = channel.dropped_count
    if hasattr(channel, "resolved_delay_ticks"):
        summary["delay_ticks"] = channel.resolved_delay_ticks
    return summary


def _build_summary(artifacts: FourNodeTrialArtifacts) -> dict[str, Any]:
    config = artifacts.config
    a = artifacts.a_bb84
    bb = artifacts.b_bb84
    c = artifacts.c_e91
    be = artifacts.b_e91
    de = artifacts.d_e91
    network = artifacts.network

    bb84_final_equal = (
        bb.privacy_complete and bool(bb.final_key) and a.final_key == bb.final_key
    )
    e91_final_equal = (
        de.privacy_complete
        and be.privacy_complete
        and bool(de.final_key)
        and be.final_key == de.final_key
    )
    coincident_pairs = set(be.detections_by_pair) & set(de.detections_by_pair)
    classified_pairs = len(de.key_pair_indices) + sum(de.bell_counts.values())
    bb84_frame_end = a.quantum_done_delay_ticks
    e91_frame_end = c.frame_done_delay_ticks
    overlap_ticks = max(0, min(bb84_frame_end, e91_frame_end))

    return {
        "session_id": config.session_id,
        "master_seed": config.master_seed,
        "timeline_final_time": artifacts.timeline.current_time,
        "topology": {
            "nodes": list(network.nodes),
            "quantum_links": list(network.quantum_links),
            "classical_links": list(network.classical_links),
            "wires": list(network.wires),
            "devices": {
                node_id: list(network.nodes[node_id].devices)
                for node_id in network.nodes
            },
            "agents": {
                node_id: list(network.nodes[node_id].agents)
                for node_id in network.nodes
            },
        },
        "concurrency": {
            "bb84_quantum_frame": [0, bb84_frame_end],
            "e91_quantum_frame": [0, e91_frame_end],
            "quantum_frame_overlap_ticks": overlap_ticks,
            "quantum_frames_overlapped": overlap_ticks > 0,
            "bb84_postprocessing_started_before_e91_frame_end": (
                bb.quantum_done_time is not None
                and bb.quantum_done_time < e91_frame_end
            ),
        },
        "channels": {
            "quantum": {
                name: _channel_summary(channel)
                for name, channel in artifacts.quantum_channels.items()
            },
            "classical": {
                name: _channel_summary(channel)
                for name, channel in artifacts.classical_channels.items()
            },
        },
        "bb84": {
            "configured_slots": config.bb84_source.num_slots,
            "prepared_photons": len(a.preparations),
            "detector_reports": len(bb.detector_reports),
            "successful_detections": len(bb.detections),
            "failed_reports": len(bb.failed_reports),
            "unassigned_reports": len(bb.unassigned_reports),
            "duplicate_slot_reports": len(bb.duplicate_slot_reports),
            "sifted_bits": len(bb.sifted_bits),
            "sample_size": len(bb.sample_positions),
            "sample_errors": bb.sample_errors,
            "estimated_qber": bb.estimated_qber,
            "qber_accepted": bb.qber_accepted,
            "cascade_complete": bb.cascade_complete,
            "cascade_parity_requests": bb.cascade_parity_requests,
            "cascade_corrections": bb.cascade_corrections,
            "cascade_leaked_bits": bb.cascade_leaked_bits,
            "reconciled_key_length": len(bb.reconciled_bits),
            "reconciled_bits_equal": a.reconciled_bits == bb.reconciled_bits,
            "verification_accepted": bb.verification_accepted,
            "verification_leaked_bits": bb.verification_leaked_bits,
            "privacy_complete": bb.privacy_complete,
            "privacy_revealed_bits": bb.privacy_revealed_bits,
            "final_key_length": bb.final_key_length,
            "final_keys_equal": bb84_final_equal,
            "protocol_complete": bb84_final_equal,
            "a_abort": a.aborted_reason,
            "b_abort": bb.aborted_reason,
        },
        "e91": {
            "configured_slots": config.e91_source.num_slots,
            "prepared_pairs": len(c.preparations),
            "b_detector_reports": len(be.detector_reports),
            "d_detector_reports": len(de.detector_reports),
            "b_successful_detections": len(be.detections_by_pair),
            "d_successful_detections": len(de.detections_by_pair),
            "b_failed_reports": len(be.failed_reports),
            "d_failed_reports": len(de.failed_reports),
            "coincident_successful_detections": len(coincident_pairs),
            "key_rounds": len(de.key_pair_indices),
            "bell_rounds": sum(de.bell_counts.values()),
            "unused_coincident_rounds": max(
                0, len(coincident_pairs) - classified_pairs
            ),
            "bell_counts": de.bell_counts,
            "bell_correlations": de.bell_correlations,
            "observed_s": de.observed_s,
            "s_lower": de.s_lower,
            "bell_accepted": de.bell_accepted,
            "sample_size": len(de.sample_positions),
            "sample_errors": de.sample_errors,
            "estimated_qber": de.estimated_qber,
            "qber_accepted": de.qber_accepted,
            "cascade_complete": de.cascade_complete,
            "cascade_parity_requests": de.cascade_parity_requests,
            "cascade_corrections": de.cascade_corrections,
            "cascade_leaked_bits": de.cascade_leaked_bits,
            "reconciled_key_length": len(de.reconciled_bits),
            "reconciled_bits_equal": be.reconciled_bits == de.reconciled_bits,
            "verification_accepted": de.verification_accepted,
            "verification_leaked_bits": de.verification_leaked_bits,
            "privacy_budget": (
                de.privacy_budget.as_dict() if de.privacy_budget is not None else None
            ),
            "privacy_complete": de.privacy_complete,
            "final_key_length": de.final_key_length,
            "final_keys_equal": e91_final_equal,
            "protocol_complete": e91_final_equal,
            "b_abort": be.aborted_reason,
            "d_abort": de.aborted_reason,
        },
        "report_isolation": {
            "b_bb84_reports_only_from_bb84_detector": all(
                report.device_id == config.bb84_detector.device_id
                for report in bb.detector_reports
            ),
            "b_e91_reports_only_from_e91_detector": all(
                report.device_id == config.e91_b_detector.device_id
                for report in be.detector_reports
            ),
        },
        "protocols_complete": bb84_final_equal and e91_final_equal,
    }


def run_four_node_qkd_trial(
    config: FourNodeQKDConfig | None = None,
    *,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build and execute one concurrent BB84/E91 trial."""
    artifacts = build_four_node_qkd_trial(config, log_file=log_file)
    artifacts.runtime.bind_all()
    artifacts.runtime.schedule_agent_starts()
    artifacts.runtime.run_until_empty()
    for sink in artifacts.sinks:
        sink.flush()
    return _build_summary(artifacts)


__all__ = [
    "FourNodeTrialArtifacts",
    "build_four_node_qkd_trial",
    "run_four_node_qkd_trial",
]
