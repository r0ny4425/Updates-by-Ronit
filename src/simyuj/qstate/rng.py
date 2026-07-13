from __future__ import annotations

"""Small random-number helpers for explicit qstate sampling streams.

Sampling code in ``qstate`` receives RNG streams from callers instead of
using NumPy's global random state. These helpers accept the small interface used
by NumPy ``Generator``-like objects and a few simpler test doubles.
"""

from collections.abc import Sequence
from typing import Any

from .math.const import PROB_ATOL


def as_rng(rng: Any) -> Any:
    """Return an explicit RNG stream.

    Parameters
    ----------
    rng : Any
        Random-number stream supplied by the caller.

    Returns
    -------
    Any
        The same RNG object.

    Raises
    ------
    ValueError
        If ``rng`` is ``None``.
    """
    if rng is None:
        raise ValueError("an explicit rng stream is required")
    return rng


def rand(rng: Any) -> float:
    """Draw one scalar sample from an explicit RNG stream.

    The helper accepts objects exposing ``random()`` or ``rand()`` and converts
    the returned scalar to ``float``. It does not check that the draw lies in the
    unit interval.

    Parameters
    ----------
    rng : Any
        Random-number stream.

    Returns
    -------
    float
        Scalar draw converted to ``float``.

    Raises
    ------
    ValueError
        If ``rng`` is ``None``.
    TypeError
        If no supported draw method exists or the draw is sequence-like.
    """
    rng = as_rng(rng)
    if hasattr(rng, "random"):
        value = rng.random()
    elif hasattr(rng, "rand"):
        value = rng.rand()
    else:
        raise TypeError("rng must provide random() or rand()")

    if isinstance(value, Sequence):
        raise TypeError("rng draw must be scalar")
    return float(value)


def choice(rng: Any, values: Sequence[Any], p: Sequence[float] | None = None) -> Any:
    """Choose one value from a non-empty sequence.

    If the RNG exposes ``choice()``, that method is used directly. Otherwise
    unweighted choices use a scalar draw and weighted choices use
    :func:`sample_index`.

    Parameters
    ----------
    rng : Any
        Random-number stream.
    values : Sequence[Any]
        Candidate values.
    p : Sequence[float] or None, optional
        Optional probability vector for weighted fallback sampling.

    Returns
    -------
    Any
        One selected value from ``values``.

    Raises
    ------
    ValueError
        If ``rng`` is ``None`` or ``values`` is empty.
    TypeError
        If fallback drawing requires ``random()`` or ``rand()`` and neither
        method exists.
    """
    rng = as_rng(rng)
    if not values:
        raise ValueError("values must be non-empty")

    if hasattr(rng, "choice"):
        if p is None:
            return rng.choice(values)
        try:
            return rng.choice(values, p=p)
        except TypeError:
            return rng.choice(values)

    if p is not None:
        return values[sample_index(rng, p)]

    index = int(rand(rng) * len(values))
    if index >= len(values):
        index = len(values) - 1
    return values[index]


def sample_index(rng: Any, probs: Sequence[float]) -> int:
    """Sample an index from a normalized probability vector.

    Parameters
    ----------
    rng : Any
        Random-number stream.
    probs : Sequence[float]
        Non-empty probabilities that must be non-negative and sum to one within
        :data:`simyuj.qstate.math.const.PROB_ATOL`.

    Returns
    -------
    int
        Selected index in ``range(len(probs))``.

    Raises
    ------
    ValueError
        If ``rng`` is ``None``, ``probs`` is empty, a probability is negative,
        or the probabilities do not sum to one.
    TypeError
        If the RNG cannot produce a scalar draw.
    """
    rng = as_rng(rng)
    if not probs:
        raise ValueError("probs must be non-empty")

    checked = tuple(float(probability) for probability in probs)
    if any(probability < 0.0 for probability in checked):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(checked) - 1.0) > PROB_ATOL:
        raise ValueError("probabilities must sum to 1")

    threshold = rand(rng)
    cumulative = 0.0
    for index, probability in enumerate(checked):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(checked) - 1


__all__ = ["as_rng", "choice", "rand", "sample_index"]
