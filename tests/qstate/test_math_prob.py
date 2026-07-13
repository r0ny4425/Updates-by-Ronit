from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate.math.prob import (
    argmax_prob,
    check_prob_vector,
    clip_prob,
    is_deterministic,
    normalize_prob_vector,
    normalize_weights,
    safe_real,
)


def test_safe_real_accepts_tiny_imaginary_part_and_rejects_bad_values() -> None:
    assert safe_real(0.25 + 1e-13j) == pytest.approx(0.25)
    assert safe_real(np.float64(0.5), name="weight") == pytest.approx(0.5)

    with pytest.raises(ValueError, match="finite"):
        safe_real(np.inf, name="weight")
    with pytest.raises(ValueError, match="real"):
        safe_real(0.25 + 1e-3j, name="weight")


def test_clip_prob_clips_only_tolerance_drift() -> None:
    assert clip_prob(-1e-13) == 0.0
    assert clip_prob(1.0 + 1e-13) == 1.0
    assert clip_prob(0.25) == pytest.approx(0.25)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        clip_prob(-1e-3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        clip_prob(1.001)


def test_normalize_prob_vector_checks_sequence_and_positive_total() -> None:
    assert normalize_prob_vector([0.25, 0.75]) == pytest.approx((0.25, 0.75))
    assert normalize_prob_vector(np.array([0.2, 0.2])) == pytest.approx((0.5, 0.5))
    assert normalize_prob_vector([1e-13, 1.0]) == pytest.approx((1e-13, 1.0))

    with pytest.raises(TypeError, match="sequence"):
        normalize_prob_vector({0.5, 0.5})
    with pytest.raises(ValueError, match="non-empty"):
        normalize_prob_vector([])
    with pytest.raises(ValueError, match="positive total"):
        normalize_prob_vector([0.0, 0.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        normalize_prob_vector([2.0, 1.0])


def test_normalize_weights_allows_sampling_weights() -> None:
    assert normalize_weights([-1.0, 2.0]) == pytest.approx((0.0, 1.0))
    assert normalize_weights([0.0, 3.0]) == pytest.approx((0.0, 1.0))
    assert normalize_weights(np.array([2.0, 2.0])) == pytest.approx((0.5, 0.5))

    with pytest.raises(TypeError, match="sequence"):
        normalize_weights({0.5, 0.5})
    with pytest.raises(ValueError, match="non-empty"):
        normalize_weights([])
    with pytest.raises(ValueError, match="positive total"):
        normalize_weights([-1.0, 0.0])


def test_check_prob_vector_requires_sum_to_one() -> None:
    assert check_prob_vector([0.25, 0.75]) == pytest.approx((0.25, 0.75))
    assert check_prob_vector(np.array([0.5, 0.5])) == pytest.approx((0.5, 0.5))

    with pytest.raises(TypeError, match="sequence"):
        check_prob_vector("01")
    with pytest.raises(ValueError, match="sum to one"):
        check_prob_vector([0.2, 0.2])


def test_argmax_and_determinism_use_normalized_probabilities() -> None:
    assert argmax_prob([0.1, 0.4, 0.4]) == 1
    assert is_deterministic([0.0, 1.0])
    assert not is_deterministic([1.0, 1.0])
