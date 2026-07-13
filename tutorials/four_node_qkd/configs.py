"""Configuration for the concurrent four-node BB84 and E91 tutorial."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from examples.bb84.configs import (
    BB84AliceSourceConfig,
    BB84BobDetectorConfig,
    BB84CascadeConfig,
    BB84ClassicalChannelConfig,
    BB84PrivacyAmplificationConfig,
    BB84QBERConfig,
    BB84QuantumChannelConfig,
    BB84VerificationConfig,
)


@dataclass(frozen=True)
class E91SourceConfig:
    """Central entangled-pair source settings."""

    device_id: str = "c_e91_source"
    clock_hz: float = 50e6
    num_slots: int = 50_000
    emission_probability: float = 0.90
    wavelength_nm: float = 1550.0
    timing_jitter_stddev_s: float = 20e-12
    max_timing_jitter_s: float = 100e-12


@dataclass(frozen=True)
class E91QuantumChannelConfig:
    """Settings for one entangled-photon fiber arm."""

    channel_id: str
    length_m: float = 10_000.0
    attenuation_db_per_km: float = 0.20
    fixed_insertion_loss_db: float = 1.0
    propagation_speed_m_per_s: float = 2.0e8
    timing_jitter_stddev_s: float = 50e-12
    depolarizing_probability: float = 0.03


@dataclass(frozen=True)
class E91DetectorConfig:
    """Settings for one E91 polarization receiver."""

    device_id: str
    efficiency: float = 0.90
    dark_count_rate_hz: float = 100.0
    dead_time_s: float = 50e-9
    jitter_stddev_s: float = 80e-12
    p_afterpulse: float = 0.0
    afterpulse_decay_s: float = 100e-9
    detection_window_s: float = 1e-9
    output_latency_s: float = 5e-9


@dataclass(frozen=True)
class E91BasisSetting:
    """One E91 analyzer setting and its random selection probability."""

    label: str
    angle_rad: float
    probability: float = 1.0 / 3.0


def default_b_settings() -> tuple[E91BasisSetting, ...]:
    return (
        E91BasisSetting("B0", 0.0),
        E91BasisSetting("B1", math.pi / 4),
        E91BasisSetting("B2", math.pi / 2),
    )


def default_d_settings() -> tuple[E91BasisSetting, ...]:
    return (
        E91BasisSetting("D0", math.pi / 4),
        E91BasisSetting("D1", math.pi / 2),
        E91BasisSetting("D2", 3 * math.pi / 4),
    )


@dataclass(frozen=True)
class E91PostProcessingConfig:
    """Teaching-level E91 post-processing and extraction settings."""

    b_settings: tuple[E91BasisSetting, ...] = field(default_factory=default_b_settings)
    d_settings: tuple[E91BasisSetting, ...] = field(default_factory=default_d_settings)
    min_chsh_samples_per_category: int = 32
    chsh_safety_margin: float = 0.05
    qber_sample_fraction: float = 0.20
    min_qber_sample_bits: int = 16
    min_remaining_bits: int = 64
    qber_safety_margin: float = 0.02
    qber_abort_threshold: float = 0.07
    cascade_passes: int = 4
    cascade_min_block_size: int = 2
    cascade_block_qber_floor: float = 0.02
    cascade_max_first_block_size: int = 32
    verification_tag_len: int = 64
    security_margin_bits: int = 64
    min_final_key_bits: int = 32


@dataclass(frozen=True)
class PublicClassicalChannelConfig:
    """Reliable authenticated public-fiber transport settings."""

    length_m: float
    fiber_speed_m_per_s: float = 2.0e8
    loss_probability: float = 0.0


@dataclass(frozen=True)
class FourNodeQKDConfig:
    """Complete physical and protocol configuration for one tutorial run."""

    master_seed: int = 2026
    session_id: str = "four_node_bb84_e91"

    bb84_source: BB84AliceSourceConfig = field(default_factory=BB84AliceSourceConfig)
    bb84_quantum: BB84QuantumChannelConfig = field(
        default_factory=BB84QuantumChannelConfig
    )
    bb84_detector: BB84BobDetectorConfig = field(default_factory=BB84BobDetectorConfig)
    bb84_classical: BB84ClassicalChannelConfig = field(
        default_factory=BB84ClassicalChannelConfig
    )
    bb84_qber: BB84QBERConfig = field(default_factory=BB84QBERConfig)
    bb84_cascade: BB84CascadeConfig = field(default_factory=BB84CascadeConfig)
    bb84_verification: BB84VerificationConfig = field(
        default_factory=BB84VerificationConfig
    )
    bb84_privacy: BB84PrivacyAmplificationConfig = field(
        default_factory=BB84PrivacyAmplificationConfig
    )

    e91_source: E91SourceConfig = field(default_factory=E91SourceConfig)
    e91_c_to_b: E91QuantumChannelConfig = field(
        default_factory=lambda: E91QuantumChannelConfig("c_to_b_e91_quantum")
    )
    e91_c_to_d: E91QuantumChannelConfig = field(
        default_factory=lambda: E91QuantumChannelConfig("c_to_d_e91_quantum")
    )
    e91_b_detector: E91DetectorConfig = field(
        default_factory=lambda: E91DetectorConfig("b_e91_detector")
    )
    e91_d_detector: E91DetectorConfig = field(
        default_factory=lambda: E91DetectorConfig("d_e91_detector")
    )
    e91_postprocessing: E91PostProcessingConfig = field(
        default_factory=E91PostProcessingConfig
    )

    c_to_b_classical: PublicClassicalChannelConfig = field(
        default_factory=lambda: PublicClassicalChannelConfig(10_000.0)
    )
    c_to_d_classical: PublicClassicalChannelConfig = field(
        default_factory=lambda: PublicClassicalChannelConfig(10_000.0)
    )
    b_to_d_classical: PublicClassicalChannelConfig = field(
        default_factory=lambda: PublicClassicalChannelConfig(20_000.0)
    )
    d_to_b_classical: PublicClassicalChannelConfig = field(
        default_factory=lambda: PublicClassicalChannelConfig(20_000.0)
    )


__all__ = [
    "E91BasisSetting",
    "E91DetectorConfig",
    "E91PostProcessingConfig",
    "E91QuantumChannelConfig",
    "E91SourceConfig",
    "FourNodeQKDConfig",
    "PublicClassicalChannelConfig",
]
