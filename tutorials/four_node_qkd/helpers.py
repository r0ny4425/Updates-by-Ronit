"""Pure E91 helpers shared by agents, trial construction, and tests."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from examples.postprocessing.bb84_event.helpers import binary_entropy
from simyuj.primitives.units import seconds_to_ticks
from simyuj.qstate.measure import MeasurementBasis

from .configs import (
    E91DetectorConfig,
    E91PostProcessingConfig,
    E91QuantumChannelConfig,
    E91SourceConfig,
    PublicClassicalChannelConfig,
)

CHSH_CATEGORIES = ("E_B0_D0", "E_B0_D2", "E_B2_D0", "E_B2_D2")
CHSH_SETTING_PAIRS = {
    "E_B0_D0": ("B0", "D0"),
    "E_B0_D2": ("B0", "D2"),
    "E_B2_D0": ("B2", "D0"),
    "E_B2_D2": ("B2", "D2"),
}
KEY_SETTING_PAIRS = {("B1", "D0"), ("B2", "D1")}

_PAIR_ID_PATTERN = re.compile(r":pair:(\d+):(left|right)$")


@dataclass(frozen=True)
class E91PrivacyBudget:
    """Auditable terms in the tutorial's illustrative extraction budget."""

    reconciled_length: int
    s_lower: float
    qber_upper: float
    eve_information_fraction: float
    eve_information_bits: int
    cascade_disclosures: int
    asymptotic_ec_bits: int
    error_correction_leak_bits: int
    verification_leak_bits: int
    security_margin_bits: int
    final_key_length: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def validate_e91_config(config: E91PostProcessingConfig) -> None:
    """Reject configurations that would make protocol semantics ambiguous."""
    for name, settings in (("B", config.b_settings), ("D", config.d_settings)):
        labels = [setting.label for setting in settings]
        if len(labels) != len(set(labels)):
            raise ValueError(f"E91 {name} basis labels must be unique")
        probability = sum(setting.probability for setting in settings)
        if not math.isclose(probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"E91 {name} basis probabilities must sum to 1")
        if any(setting.probability <= 0 for setting in settings):
            raise ValueError(f"E91 {name} basis probabilities must be positive")

    required_b = {"B0", "B1", "B2"}
    required_d = {"D0", "D1", "D2"}
    if {setting.label for setting in config.b_settings} != required_b:
        raise ValueError("E91 B settings must be exactly B0, B1, and B2")
    if {setting.label for setting in config.d_settings} != required_d:
        raise ValueError("E91 D settings must be exactly D0, D1, and D2")

    if config.min_chsh_samples_per_category <= 0:
        raise ValueError("minimum CHSH samples must be positive")
    if not 0 <= config.chsh_safety_margin < 2 * math.sqrt(2) - 2:
        raise ValueError("CHSH safety margin is outside the useful range")
    if not 0 < config.qber_sample_fraction < 1:
        raise ValueError("QBER sample fraction must be in (0, 1)")
    if config.min_qber_sample_bits <= 0 or config.min_remaining_bits <= 0:
        raise ValueError("minimum QBER sample and remaining bits must be positive")
    if not 0 <= config.qber_abort_threshold <= 0.5:
        raise ValueError("QBER abort threshold must be in [0, 0.5]")
    if config.cascade_passes <= 0:
        raise ValueError("Cascade passes must be positive")
    if config.verification_tag_len <= 0:
        raise ValueError("verification tag length must be positive")
    if config.security_margin_bits < 0 or config.min_final_key_bits <= 0:
        raise ValueError("privacy-amplification margins are invalid")


def linear_polarization_basis(name: str, angle_deg: float) -> MeasurementBasis:
    """Return a linear-polarization basis rotated from the Z basis."""
    theta = np.deg2rad(angle_deg)
    ket_0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    ket_1 = np.array([-np.sin(theta), np.cos(theta)], dtype=np.complex128)
    return MeasurementBasis(name=name, vectors=(ket_0, ket_1), labels=("0", "1"))


def photon_basis_from_e91_angle(name: str, angle_rad: float) -> MeasurementBasis:
    """Convert an E91 analyzer angle to its photon-polarization basis."""
    return linear_polarization_basis(name, np.rad2deg(angle_rad / 2))


def pair_index_from_signal_id(signal_id: object) -> int:
    """Extract the source pair index from an entangled-member signal ID."""
    match = _PAIR_ID_PATTERN.search(str(signal_id))
    if match is None:
        raise ValueError(f"unexpected E91 signal id: {signal_id!r}")
    return int(match.group(1))


def bit_from_outcome(outcome: object) -> int:
    if outcome == "0":
        return 0
    if outcome == "1":
        return 1
    raise ValueError(f"unexpected E91 outcome: {outcome!r}")


def singlet_corrected_bit(outcome: object) -> int:
    """Flip one endpoint's bit because psi-minus is anticorrelated."""
    return 1 - bit_from_outcome(outcome)


def outcome_value(outcome: object) -> int:
    return 1 if bit_from_outcome(outcome) == 0 else -1


def observed_correlation(outcome_pairs: Iterable[tuple[object, object]]) -> float:
    products = [
        outcome_value(left) * outcome_value(right) for left, right in outcome_pairs
    ]
    if not products:
        raise ValueError("cannot calculate correlation from an empty sample")
    return sum(products) / len(products)


def chsh_value(correlations: dict[str, float]) -> float:
    missing = [name for name in CHSH_CATEGORIES if name not in correlations]
    if missing:
        raise ValueError(f"missing CHSH correlations: {missing}")
    return (
        correlations["E_B0_D0"]
        - correlations["E_B0_D2"]
        + correlations["E_B2_D0"]
        + correlations["E_B2_D2"]
    )


def conservative_chsh_value(observed_s: float, margin: float) -> float:
    return max(2.0, min(2 * math.sqrt(2), abs(observed_s) - margin))


def e91_privacy_budget(
    *,
    reconciled_length: int,
    s_lower: float,
    estimated_qber: float,
    qber_safety_margin: float,
    cascade_disclosures: int,
    verification_tag_len: int,
    security_margin_bits: int,
) -> E91PrivacyBudget:
    """Return the tutorial's CHSH/QBER extraction budget."""
    if reconciled_length <= 0:
        raise ValueError("reconciled length must be positive")
    if not 2 < s_lower <= 2 * math.sqrt(2) + 1e-12:
        raise ValueError("s_lower must show a CHSH violation")
    if cascade_disclosures < 0:
        raise ValueError("Cascade disclosures must be non-negative")

    qber_upper = min(0.5, max(0.0, estimated_qber + qber_safety_margin))
    root_term = max(0.0, (s_lower / 2) ** 2 - 1)
    eve_probability = (1 + math.sqrt(root_term)) / 2
    eve_fraction = binary_entropy(eve_probability)
    eve_bits = math.ceil(reconciled_length * eve_fraction)
    asymptotic_ec_bits = math.ceil(reconciled_length * binary_entropy(qber_upper))
    error_correction_leak = max(cascade_disclosures, asymptotic_ec_bits)
    final_length = max(
        0,
        reconciled_length
        - eve_bits
        - error_correction_leak
        - verification_tag_len
        - security_margin_bits,
    )
    return E91PrivacyBudget(
        reconciled_length=reconciled_length,
        s_lower=s_lower,
        qber_upper=qber_upper,
        eve_information_fraction=eve_fraction,
        eve_information_bits=eve_bits,
        cascade_disclosures=cascade_disclosures,
        asymptotic_ec_bits=asymptotic_ec_bits,
        error_correction_leak_bits=error_correction_leak,
        verification_leak_bits=verification_tag_len,
        security_margin_bits=security_margin_bits,
        final_key_length=final_length,
    )


def e91_source_done_delay_ticks(config: E91SourceConfig) -> int:
    return seconds_to_ticks(
        config.num_slots / config.clock_hz + config.max_timing_jitter_s
    )


def e91_receiver_guard_ticks(
    *,
    quantum: E91QuantumChannelConfig,
    classical: PublicClassicalChannelConfig,
    detector: E91DetectorConfig,
) -> int:
    quantum_delay_s = quantum.length_m / quantum.propagation_speed_m_per_s
    classical_delay_s = classical.length_m / classical.fiber_speed_m_per_s
    guard_s = (
        max(0.0, quantum_delay_s - classical_delay_s)
        + 6 * quantum.timing_jitter_stddev_s
        + 6 * detector.jitter_stddev_s
        + detector.detection_window_s
        + detector.output_latency_s
    )
    return seconds_to_ticks(guard_s)


__all__ = [
    "CHSH_CATEGORIES",
    "CHSH_SETTING_PAIRS",
    "E91PrivacyBudget",
    "KEY_SETTING_PAIRS",
    "bit_from_outcome",
    "chsh_value",
    "conservative_chsh_value",
    "e91_privacy_budget",
    "e91_receiver_guard_ticks",
    "e91_source_done_delay_ticks",
    "linear_polarization_basis",
    "observed_correlation",
    "outcome_value",
    "pair_index_from_signal_id",
    "photon_basis_from_e91_angle",
    "singlet_corrected_bit",
    "validate_e91_config",
]
