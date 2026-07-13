from __future__ import annotations

import pytest

from simyuj.qstate.rng import choice, rand, sample_index


class RandomRng:
    def __init__(self, draw: object) -> None:
        self.draw = draw

    def random(self) -> object:
        return self.draw


class RandRng:
    def __init__(self, draw: object) -> None:
        self.draw = draw

    def rand(self) -> object:
        return self.draw


def test_sample_index_uses_explicit_rng_thresholds_deterministically() -> None:
    assert sample_index(RandomRng(0.0), (0.25, 0.75)) == 0
    assert sample_index(RandomRng(0.25), (0.25, 0.75)) == 1
    assert sample_index(RandomRng(0.999), (0.25, 0.75)) == 1


def test_rng_helpers_accept_random_or_rand_scalar_streams() -> None:
    assert rand(RandomRng(0.4)) == pytest.approx(0.4)
    assert rand(RandRng(0.6)) == pytest.approx(0.6)
    assert choice(RandomRng(0.99), ("a", "b", "c")) == "c"


def test_rng_helpers_require_explicit_scalar_streams() -> None:
    with pytest.raises(ValueError, match="explicit rng"):
        sample_index(None, (1.0,))
    with pytest.raises(TypeError, match="random\\(\\) or rand\\(\\)"):
        rand(object())
    with pytest.raises(TypeError, match="scalar"):
        rand(RandomRng([0.1]))


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ((), "non-empty"),
        ((0.5, -0.5, 1.0), "non-negative"),
        ((0.4, 0.4), "sum to 1"),
    ],
)
def test_sample_index_rejects_invalid_probability_vectors(
    probabilities: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_index(RandomRng(0.1), probabilities)
