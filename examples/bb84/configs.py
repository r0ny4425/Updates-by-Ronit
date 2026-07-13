"""Configuration dataclasses for the single-photon BB84 example."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BB84AliceSourceConfig:
    """Settings for Alice's single-photon source."""

    device_id: str = "alice_source"

    clock_hz: float = 100e6

    # Number of emission slots to simulate.
    num_slots: int = 30_000

    emission_probability: float = 0.70

    wavelength_nm: float = 1550.0

    # Source timing jitter.
    timing_jitter_stddev_s: float = 20e-12
    max_timing_jitter_s: float = 100e-12


@dataclass(frozen=True)
class BB84QuantumChannelConfig:
    """Settings for the Alice-to-Bob quantum fiber."""

    channel_id: str = "alice_to_bob_quantum"

    length_m: float = 25_000.0

    attenuation_db_per_km: float = 0.20

    fixed_insertion_loss_db: float = 2.0

    propagation_speed_m_per_s: float = 2.0e8

    # Channel timing jitter.
    timing_jitter_stddev_s: float = 50e-12

    depolarizing_probability: float = 0.02


@dataclass(frozen=True)
class BB84ClassicalChannelConfig:
    """Settings for the public classical channels."""

    length_m: float = 25_000
    fiber_speed_m_per_s: float = 2.0e8

    # Authenticated public channel should be reliable for now.
    # We can model message loss later, but post-processing is easier to debug first.
    loss_probability: float = 0.0


@dataclass(frozen=True)
class BB84BobDetectorConfig:
    """Settings for Bob's BB84 detector array."""

    device_id: str = "bob_detector"

    efficiency: float = 0.65
    dark_count_rate_hz: float = 100.0

    dead_time_s: float = 50e-9

    jitter_stddev_s: float = 50e-12

    p_afterpulse: float = 0.001
    afterpulse_decay_s: float = 100e-9

    detection_window_s: float = 500e-12

    output_latency_s: float = 5e-9


@dataclass(frozen=True)
class BB84QBERConfig:
    """Settings for public sample-based QBER estimation."""

    sample_fraction: float = 0.20
    min_sample_bits: int = 16
    min_remaining_bits: int = 32
    abort_threshold: float = 0.11


@dataclass(frozen=True)
class BB84CascadeConfig:
    """Settings for the Cascade reconciliation pass."""

    passes: int = 4
    min_block_size: int = 2

    # Used only to choose Cascade block sizes.
    # This prevents QBER=0 sample from making blocks too large.
    block_qber_floor: float = 0.02
    max_first_block_size: int = 32


@dataclass(frozen=True)
class BB84VerificationConfig:
    """Settings for the post-Cascade verification hash."""

    # Failure probability is about 2^-tag_len if the reconciled keys differ.
    tag_len: int = 64


@dataclass(frozen=True)
class BB84PrivacyAmplificationConfig:
    """Settings for the teaching-level privacy-amplification budget."""

    # Conservative cushion because QBER was estimated from a sample.
    qber_safety_margin: float = 0.02

    # Security slack for this notebook-level finite-key model.
    security_margin_bits: int = 64

    # Abort if the extracted key would be too small to be meaningful.
    min_final_key_bits: int = 32


__all__ = [
    "BB84AliceSourceConfig",
    "BB84BobDetectorConfig",
    "BB84CascadeConfig",
    "BB84ClassicalChannelConfig",
    "BB84PrivacyAmplificationConfig",
    "BB84QBERConfig",
    "BB84QuantumChannelConfig",
    "BB84VerificationConfig",
]
