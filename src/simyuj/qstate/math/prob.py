from __future__ import annotations

"""Probability coercion and normalization helpers.

These helpers centralize the package's probability tolerance behavior.  Values
are converted through Python ``complex`` first so tiny imaginary roundoff can be
accepted, while materially complex or non-finite values are rejected.
"""

from typing import Any, cast

import numpy as np

from .const import PROB_ATOL


def safe_real(value: object, *, name: str = "value", atol: float = PROB_ATOL) -> float:
    """Coerce a scalar value to a finite real float.

    Parameters
    ----------
    value : object
        Scalar accepted by Python ``complex``.
    name : str, default="value"
        Name used in error messages.
    atol : float, default=PROB_ATOL
        Maximum allowed magnitude of the imaginary component.

    Returns
    -------
    float
        Real component of ``value``.

    Raises
    ------
    ValueError
        If the real or imaginary component is non-finite, or if the imaginary
        component exceeds ``atol``.
    """
    number = complex(cast(Any, value))

    if not np.isfinite(number.real) or not np.isfinite(number.imag):
        raise ValueError(f"{name} must be finite")
    if abs(number.imag) > atol:
        raise ValueError(f"{name} must be real up to tolerance")
    return float(number.real)


def clip_prob(value: object, *, name: str = "probability") -> float:
    """Coerce and clip a probability with tolerance for roundoff drift.

    Parameters
    ----------
    value : object
        Scalar probability candidate.
    name : str, default="probability"
        Name used in error messages.

    Returns
    -------
    float
        Probability in the closed interval ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``value`` is non-finite, materially complex, or outside ``[0, 1]`` by
        more than ``PROB_ATOL``.

    Notes
    -----
    Values slightly below ``0`` or above ``1`` are clipped only when they are
    within ``PROB_ATOL``.
    """
    probability = safe_real(value, name=name)

    if probability < -PROB_ATOL or probability > 1.0 + PROB_ATOL:
        raise ValueError(f"{name} must be in [0, 1]")
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return probability


def normalize_prob_vector(
    values: object,
    *,
    name: str = "probabilities",
) -> tuple[float, ...]:
    """Normalize a finite sequence of probability-like values.

    Parameters
    ----------
    values : object
        Tuple, list, or NumPy array of scalar probabilities.  Each entry must be
        in ``[0, 1]`` up to ``PROB_ATOL``.
    name : str, default="probabilities"
        Name used in error messages.

    Returns
    -------
    tuple of float
        Unit-sum probabilities in the original order.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple, list, or NumPy array.
    ValueError
        If the sequence is empty, has no positive total probability, or contains
        an invalid probability.
    """
    if not isinstance(values, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must be a sequence")

    raw = tuple(clip_prob(value, name=name) for value in values)
    if not raw:
        raise ValueError(f"{name} must be non-empty")

    total = float(sum(raw))
    if total <= PROB_ATOL:
        raise ValueError(f"{name} must have positive total probability")
    return tuple(value / total for value in raw)


def normalize_weights(
    values: object,
    *,
    name: str = "probabilities",
) -> tuple[float, ...]:
    """Normalize non-negative sampling weights.

    Parameters
    ----------
    values : object
        Tuple, list, or NumPy array of scalar weights.  Negative real values are
        treated as zero.
    name : str, default="probabilities"
        Name used in error messages.

    Returns
    -------
    tuple of float
        Unit-sum weights in the original order.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple, list, or NumPy array.
    ValueError
        If the sequence is empty, has no positive total weight, or contains a
        non-finite or materially complex value.

    Notes
    -----
    Unlike :func:`normalize_prob_vector`, this function does not require entries
    to be at most ``1`` before normalization.
    """
    if not isinstance(values, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must be a sequence")

    raw = tuple(max(0.0, safe_real(value, name=name)) for value in values)
    if not raw:
        raise ValueError(f"{name} must be non-empty")

    total = float(sum(raw))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total")
    return tuple(value / total for value in raw)


def check_prob_vector(
    values: object,
    *,
    name: str = "probabilities",
) -> tuple[float, ...]:
    """Validate a unit-sum probability vector.

    Parameters
    ----------
    values : object
        Tuple, list, or NumPy array of scalar probabilities.
    name : str, default="probabilities"
        Name used in error messages.

    Returns
    -------
    tuple of float
        Checked probabilities, with entries near ``0`` or ``1`` clipped by
        :func:`clip_prob`.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple, list, or NumPy array.
    ValueError
        If the sequence is empty, contains an invalid probability, or does not
        sum to one within ``PROB_ATOL``.
    """
    if not isinstance(values, (tuple, list, np.ndarray)):
        raise TypeError(f"{name} must be a sequence")

    checked = tuple(clip_prob(value, name=name) for value in values)
    if not checked:
        raise ValueError(f"{name} must be non-empty")

    total = sum(checked)
    if abs(total - 1.0) > PROB_ATOL:
        raise ValueError(f"{name} must sum to one")
    return checked


def argmax_prob(values: object) -> int:
    """Return the first index of the largest normalized probability.

    Parameters
    ----------
    values : object
        Probability-like sequence accepted by :func:`normalize_prob_vector`.

    Returns
    -------
    int
        Index of the first strictly maximal normalized probability.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple, list, or NumPy array.
    ValueError
        If ``values`` cannot be normalized as probabilities.
    """
    probabilities = normalize_prob_vector(values)

    best_index = 0
    best_value = probabilities[0]
    for index, value in enumerate(probabilities[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def is_deterministic(values: object) -> bool:
    """Return whether normalized probabilities contain a deterministic outcome.

    Parameters
    ----------
    values : object
        Probability-like sequence accepted by :func:`normalize_prob_vector`.

    Returns
    -------
    bool
        ``True`` if any normalized probability is at least
        ``1 - PROB_ATOL``.

    Raises
    ------
    TypeError
        If ``values`` is not a tuple, list, or NumPy array.
    ValueError
        If ``values`` cannot be normalized as probabilities.
    """
    probabilities = normalize_prob_vector(values)
    return any(value >= 1.0 - PROB_ATOL for value in probabilities)


__all__ = [
    "argmax_prob",
    "check_prob_vector",
    "clip_prob",
    "is_deterministic",
    "normalize_prob_vector",
    "normalize_weights",
    "safe_real",
]
