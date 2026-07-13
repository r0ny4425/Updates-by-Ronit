from __future__ import annotations

"""Probability normalization and sampling helpers for measurements.

Measurement probabilities are normalized through ``normalize_weights``: negative
real entries are treated as zero and the remaining positive total is scaled to
one.  Probabilistic sampling requires an explicit RNG unless one outcome is
already certain within ``PROB_ATOL``.
"""

from collections.abc import Sequence
from typing import Any

from ..math.const import PROB_ATOL
from ..math.prob import normalize_weights
from ..rng import sample_index


def normalize_probs(probs: Sequence[float]) -> tuple[float, ...]:
    """Normalize measurement probabilities as sampling weights.

    Parameters
    ----------
    probs : sequence of float
        Probability-like weights.  Negative real values are treated as zero by
        ``normalize_weights``.

    Returns
    -------
    tuple of float
        Unit-sum probabilities in the original order.

    Raises
    ------
    TypeError
        If ``probs`` is not a supported sequence type.
    ValueError
        If ``probs`` is empty, has no positive total, or contains a non-finite
        or materially complex value.
    """
    return normalize_weights(probs, name="probs")


def sample_probs(probs: Sequence[float], *, rng: Any | None = None) -> int:
    """Sample an index from measurement probabilities.

    Parameters
    ----------
    probs : sequence of float
        Probability-like weights normalized by :func:`normalize_probs`.
    rng : object, optional
        Random source passed to ``sample_index``.  It must provide ``random()``
        or ``rand()`` when sampling is actually required.

    Returns
    -------
    int
        Sampled outcome index.  If any normalized probability is within
        ``PROB_ATOL`` of one, the first such index is returned without using
        ``rng``.

    Raises
    ------
    TypeError
        If probability normalization or RNG drawing receives an invalid type.
    ValueError
        If probabilities cannot be normalized, if no outcome is certain and
        ``rng`` is ``None``, or if normalized probabilities fail the downstream
        sampling checks.
    """
    checked = normalize_probs(probs)
    certain = [
        index for index, prob in enumerate(checked) if abs(prob - 1.0) <= PROB_ATOL
    ]
    if certain:
        return certain[0]
    return sample_index(rng, checked)


__all__ = ["normalize_probs", "sample_probs"]
