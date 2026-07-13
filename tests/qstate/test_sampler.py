from __future__ import annotations

import numpy as np
import pytest

from simyuj.qstate import DensityState, KetState, StateSample, StateSampler
from simyuj.qstate.errors import InvalidReprError
from simyuj.qstate.state import BellDiagState


class FixedRNG:
    def __init__(self, draw: float) -> None:
        self._draw = draw

    def random(self) -> float:
        return self._draw


def test_ket_sampler_accepts_predefined_and_custom_states() -> None:
    custom = np.array([0.6, 0.8], dtype=np.complex128)
    sampler = StateSampler(
        states=("|0>", "|+>", custom),
        probabilities=(0.25, 0.25, 0.5),
        labels=("zero", "plus", "custom"),
    )

    sample = sampler.sample(rng=FixedRNG(0.6))

    assert sampler.size == 3
    assert sampler.num_qubits == 1
    assert sampler.rep == "ket"
    assert sampler.probabilities == pytest.approx((0.25, 0.25, 0.5))
    assert sample == StateSample(
        state=sampler.states[2],
        rep="ket",
        index=2,
        label="custom",
    )
    probabilities = sampler.probabilities
    assert probabilities is not None
    assert probabilities[sample.index] == pytest.approx(0.5)
    assert isinstance(sample.state, KetState)
    np.testing.assert_allclose(sample.state.vector, custom, atol=1e-12)


def test_sampler_defaults_to_uniform_probabilities_and_empty_labels() -> None:
    sampler = StateSampler(states=("|0>", "|1>", "|+>", "|->"))

    sample = sampler.sample(rng=FixedRNG(0.74))

    assert sampler.probabilities == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert sampler.labels == (None, None, None, None)
    assert sample.index == 2
    assert sample.label is None


def test_deterministic_sampler_does_not_require_rng() -> None:
    sampler = StateSampler(
        states=("|0>", "|1>"),
        probabilities=(0.0, 1.0),
        labels=("zero", "one"),
    )

    sample = sampler.sample()

    assert sample.index == 1
    assert sample.label == "one"
    probabilities = sampler.probabilities
    assert probabilities is not None
    assert probabilities[sample.index] == pytest.approx(1.0)


def test_density_sampler_accepts_labels_vectors_and_density_matrices() -> None:
    sampler = StateSampler(
        states=(
            "|0>",
            [[0.5, 0.0], [0.0, 0.5]],
        ),
        probabilities=(0.0, 1.0),
        rep="density",
        labels=("pure-zero", "mixed"),
    )

    sample = sampler.sample()

    assert sampler.num_qubits == 1
    assert isinstance(sampler.states[0], DensityState)
    assert isinstance(sample.state, DensityState)
    assert sample.label == "mixed"
    np.testing.assert_allclose(
        sample.state.rho,
        np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128),
        atol=1e-12,
    )


def test_bell_diag_sampler_accepts_labels_mappings_and_payloads() -> None:
    payload = BellDiagState((0.25, 0.25, 0.25, 0.25))
    sampler = StateSampler(
        states=("phi+", {"phi+": 0.7, "phi-": 0.3}, payload),
        probabilities=(0.0, 0.0, 1.0),
        rep="bell_diag",
    )

    sample = sampler.sample()

    assert sampler.num_qubits == 2
    assert isinstance(sampler.states[0], BellDiagState)
    assert sample.state is payload
    assert sample.rep == "bell_diag"


def test_probabilistic_sampling_requires_explicit_rng() -> None:
    sampler = StateSampler(states=("|0>", "|1>"))

    with pytest.raises(ValueError, match="explicit rng"):
        sampler.sample()


def test_sampler_validates_public_inputs() -> None:
    with pytest.raises(TypeError, match="states must be a sequence"):
        StateSampler(states="|0>")
    with pytest.raises(ValueError, match="states must be non-empty"):
        StateSampler(states=())
    with pytest.raises(ValueError, match="probabilities length"):
        StateSampler(states=("|0>", "|1>"), probabilities=(0.5, 0.5, 0.0))
    with pytest.raises(ValueError, match="labels length"):
        StateSampler(states=("|0>", "|1>"), labels=("zero",))
    with pytest.raises(ValueError, match="labels entries"):
        StateSampler(states=("|0>",), labels=("",))
    with pytest.raises(ValueError, match="same qubit count"):
        StateSampler(states=("|0>", "phi+"))
    with pytest.raises(InvalidReprError, match="supports"):
        StateSampler(states=("|0>",), rep="graph")
