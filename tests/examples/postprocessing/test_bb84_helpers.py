from __future__ import annotations

import pytest

from examples.postprocessing.bb84_event.helpers import (
    demo_entropy_budget,
    remove_positions,
    require_positions,
    toeplitz_hash,
)


def test_remove_positions_discards_opened_indices():
    assert remove_positions([0, 1, 1, 0, 1], [1, 3]) == [0, 1, 1]


def test_require_positions_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        require_positions("sample_positions", [0, 2, 2], 5)


def test_require_positions_rejects_out_of_range_indices():
    with pytest.raises(ValueError, match="out-of-range"):
        require_positions("sample_positions", [0, 5], 5)


def test_toeplitz_hash_rejects_wrong_seed_length():
    with pytest.raises(ValueError, match="Toeplitz seed"):
        toeplitz_hash([1, 0, 1], [0, 1, 0], output_len=4)


def test_toeplitz_hash_is_deterministic_for_same_input():
    bits = [1, 0, 1, 1]
    seed = [1, 0, 0, 1, 1, 0]

    assert toeplitz_hash(bits, seed, output_len=3) == toeplitz_hash(
        bits,
        seed,
        output_len=3,
    )


def test_demo_entropy_budget_decreases_when_qber_increases():
    low_qber_budget, _, _ = demo_entropy_budget(
        n=256,
        estimated_qber=0.01,
        statistical_margin=0.02,
        cascade_leaked_bits=10,
        verification_leaked_bits=32,
        security_margin_bits=16,
    )
    high_qber_budget, _, _ = demo_entropy_budget(
        n=256,
        estimated_qber=0.10,
        statistical_margin=0.02,
        cascade_leaked_bits=10,
        verification_leaked_bits=32,
        security_margin_bits=16,
    )

    assert high_qber_budget < low_qber_budget


def test_demo_entropy_budget_decreases_when_public_leakage_increases():
    low_leakage_budget, _, _ = demo_entropy_budget(
        n=256,
        estimated_qber=0.01,
        statistical_margin=0.02,
        cascade_leaked_bits=10,
        verification_leaked_bits=32,
        security_margin_bits=16,
    )
    high_leakage_budget, _, _ = demo_entropy_budget(
        n=256,
        estimated_qber=0.01,
        statistical_margin=0.02,
        cascade_leaked_bits=80,
        verification_leaked_bits=32,
        security_margin_bits=16,
    )

    assert high_leakage_budget < low_leakage_budget
