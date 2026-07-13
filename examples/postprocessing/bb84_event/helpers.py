"""Reusable helper functions for BB84 example post-processing."""

from __future__ import annotations

import math
from typing import Any


def require_binary_bits(name: str, bits: list[int]) -> None:
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must contain only 0/1 bits")


def require_optional_binary_bits(name: str, bits: list[int | None]) -> None:
    if any(bit not in (0, 1, None) for bit in bits):
        raise ValueError(f"{name} must contain only 0/1 bits or None")


def require_measured_bits(name: str, bits: list[int | None]) -> None:
    if any(bit is None for bit in bits):
        raise ValueError(f"{name} must not contain missed detections")
    require_optional_binary_bits(name, bits)


def measured_bit(bits: list[int | None], index: int) -> int:
    bit = bits[index]
    if bit is None:
        raise ValueError(f"missing detection at position {index}")
    return bit


def require_bases(name: str, bases: list[str]) -> None:
    if any(basis not in ("Z", "X") for basis in bases):
        raise ValueError(f"{name} must contain only 'Z'/'X' bases")


def require_positions(name: str, positions: list[int], n: int) -> None:
    """Validate public positions before indexing local key material."""
    if len(set(positions)) != len(positions):
        raise ValueError(f"{name} must not contain duplicates")
    if any(pos < 0 or pos >= n for pos in positions):
        raise ValueError(f"{name} contains out-of-range positions")


def parity(bits: list[int], indices: list[int] | tuple[int, ...]) -> int:
    """Return the XOR parity over selected positions."""
    return sum(bits[i] for i in indices) & 1


def remove_positions(bits: list[int], positions: list[int]) -> list[int]:
    """Drop public sample positions from local key material."""
    drop = set(positions)
    return [bit for index, bit in enumerate(bits) if index not in drop]


def shuffled_range(rng: Any, n: int) -> list[int]:
    values = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        values[i], values[j] = values[j], values[i]
    return values


def choose_sample_positions(rng: Any, n: int, sample_size: int) -> list[int]:
    """Choose sorted public sample positions from a reproducible RNG."""
    if sample_size < 0 or sample_size > n:
        raise ValueError("sample_size must be in [0, n]")
    return sorted(shuffled_range(rng, n)[:sample_size])


def choose_permutation(rng: Any, n: int) -> list[int]:
    """Choose the shuffled index order used by one Cascade pass."""
    return shuffled_range(rng, n)


def random_bits(rng: Any, count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [1 if rng.random() < 0.5 else 0 for _ in range(count)]


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


binary_entropy = h2


def toeplitz_hash(bits: list[int], seed: list[int], output_len: int) -> list[int]:
    """Apply a binary Toeplitz hash represented by its seed bits."""
    if output_len <= 0:
        raise ValueError("output_len must be positive")

    require_binary_bits("bits", bits)
    require_binary_bits("seed", seed)

    input_len = len(bits)
    expected = input_len + output_len - 1
    if len(seed) != expected:
        raise ValueError(f"Toeplitz seed must have length {expected}")

    return [
        sum((bit & seed[row - col + input_len - 1]) for col, bit in enumerate(bits)) & 1
        for row in range(output_len)
    ]


def demo_entropy_budget(
    *,
    n: int,
    estimated_qber: float,
    statistical_margin: float,
    cascade_leaked_bits: int,
    verification_leaked_bits: int,
    security_margin_bits: int,
) -> tuple[int, float, int]:
    """Return a compact teaching budget for privacy amplification.

    Assumption: this intentionally treats the observed QBER plus a fixed margin
    as a demo phase-error bound and subtracts public leakage. It is useful for
    showing where the terms enter the event flow, but it is not a composable
    finite-key proof.
    """
    phase_error_bound = min(0.5, max(0.0, estimated_qber + statistical_margin))
    eve_info_bound = math.ceil(n * h2(phase_error_bound))
    available = (
        n
        - eve_info_bound
        - cascade_leaked_bits
        - verification_leaked_bits
        - security_margin_bits
    )
    return available, phase_error_bound, eve_info_bound
