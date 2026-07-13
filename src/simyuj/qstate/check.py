from __future__ import annotations

"""Validation helpers for qstate references, dimensions, and metadata.

The checks in this module sit at public qstate boundaries. They normalize small
vocabularies, such as representation names and metadata tuples, while leaving
array-level state validation to the representation-specific state modules.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from .errors import DimensionError, InvalidReprError
from .math.prob import check_prob_vector as _check_prob_vector
from .math.prob import clip_prob

RepName: TypeAlias = str
Meta: TypeAlias = tuple[tuple[str, object], ...]
MetaInput: TypeAlias = Meta | Mapping[str, object] | None

CANONICAL_REPS = frozenset({"ket", "density", "bell_diag", "stabilizer", "graph"})

_REP_ALIASES = {
    "statevector": "ket",
    "state_vector": "ket",
    "density_matrix": "density",
    "dm": "density",
    "bell_diagonal": "bell_diag",
}


def check_state_ref(state_ref: object) -> int:
    """Return a live-state reference after basic scalar validation.

    Parameters
    ----------
    state_ref : object
        Candidate state reference.

    Returns
    -------
    int
        The non-negative state reference.

    Raises
    ------
    TypeError
        If ``state_ref`` is not exactly an ``int``.
    ValueError
        If ``state_ref`` is negative.
    """
    if type(state_ref) is not int:
        raise TypeError("state_ref must be int")
    if state_ref < 0:
        raise ValueError("state_ref must be non-negative")
    return state_ref


def normalize_rep(rep: object) -> RepName:
    """Normalize a representation name.

    The input is stripped, lowercased, and mapped through the supported alias
    table before it is checked against the representation vocabulary.

    Parameters
    ----------
    rep : object
        Candidate representation name.

    Returns
    -------
    RepName
        Normalized representation name.

    Raises
    ------
    TypeError
        If ``rep`` is not a string.
    InvalidReprError
        If the normalized name is unsupported.
    """
    if not isinstance(rep, str):
        raise TypeError("rep must be str")
    normalized = rep.strip().lower()
    normalized = _REP_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_REPS:
        raise InvalidReprError(f"unsupported state representation: {rep!r}")
    return normalized


def check_rep(rep: object) -> RepName:
    """Return a normalized qstate representation name.

    This wrapper exists so callers can use the same ``check_*`` naming pattern
    as other scalar validators.
    """
    return normalize_rep(rep)


def check_probability(value: object, name: str = "probability") -> float:
    """Validate and clip a scalar probability.

    Parameters
    ----------
    value : object
        Candidate scalar probability.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    float
        Probability clipped by :func:`simyuj.qstate.math.prob.clip_prob`.

    Raises
    ------
    TypeError
        If ``value`` is not an ``int`` or ``float``.
    ValueError
        If ``value`` lies outside the clipping tolerance.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float")
    return clip_prob(value, name=name)


def check_prob_vector(values: object, name: str = "probabilities") -> tuple[float, ...]:
    """Validate a non-empty probability vector.

    Parameters
    ----------
    values : object
        Candidate probability sequence.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    tuple[float, ...]
        Checked probabilities as a tuple.

    Raises
    ------
    TypeError
        If ``values`` is not a non-string sequence.
    ValueError
        If the sequence is empty, contains invalid probabilities, or does not
        sum to one within the configured tolerance.
    """
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    try:
        return _check_prob_vector(tuple(values), name=name)
    except ValueError as exc:
        if str(exc) == f"{name} must sum to one":
            raise ValueError(f"{name} must sum to 1") from exc
        raise


def check_dim(dim: object, name: str = "dim") -> int:
    """Validate a positive Hilbert-space dimension.

    Parameters
    ----------
    dim : object
        Candidate dimension.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    int
        Positive dimension.

    Raises
    ------
    TypeError
        If ``dim`` is not exactly an ``int``.
    DimensionError
        If ``dim`` is not positive.
    """
    if type(dim) is not int:
        raise TypeError(f"{name} must be int")
    if dim <= 0:
        raise DimensionError(f"{name} must be positive")
    return dim


def check_dims(dims: object, name: str = "dims") -> tuple[int, ...]:
    """Validate a tuple of positive Hilbert-space dimensions.

    Parameters
    ----------
    dims : object
        Candidate dimension tuple.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    tuple[int, ...]
        Checked dimensions.

    Raises
    ------
    TypeError
        If ``dims`` is not a tuple or an entry is not exactly an ``int``.
    DimensionError
        If any entry is not positive.
    """
    if not isinstance(dims, tuple):
        raise TypeError(f"{name} must be tuple")
    return tuple(check_dim(dim, f"{name}[{index}]") for index, dim in enumerate(dims))


def check_targets(targets: object, allow_empty: bool = False) -> tuple[object, ...]:
    """Validate a target tuple without inspecting target item types.

    Parameters
    ----------
    targets : object
        Candidate target tuple.
    allow_empty : bool, optional
        Whether an empty tuple is accepted.

    Returns
    -------
    tuple[object, ...]
        Checked target tuple.

    Raises
    ------
    TypeError
        If ``targets`` is not a tuple.
    ValueError
        If ``targets`` is empty and ``allow_empty`` is false.
    """
    if not isinstance(targets, tuple):
        raise TypeError("targets must be tuple")
    if not allow_empty and not targets:
        raise ValueError("targets must be non-empty")
    return targets


def coerce_meta(meta: MetaInput) -> Meta:
    """Coerce metadata to an immutable tuple of key-value pairs.

    Metadata values are passed through unchanged. Keys must be non-empty
    strings after stripping, but the original key spelling is preserved.

    Parameters
    ----------
    meta : MetaInput
        ``None``, a mapping, or a tuple of ``(key, value)`` pairs.

    Returns
    -------
    Meta
        Metadata as a tuple of pairs.

    Raises
    ------
    TypeError
        If ``meta`` or any item has the wrong shape or key type.
    ValueError
        If a key is empty after stripping whitespace.
    """
    if meta is None:
        return ()
    if isinstance(meta, Mapping):
        items = tuple(meta.items())
    elif isinstance(meta, tuple):
        items = meta
    else:
        raise TypeError("meta must be None, mapping, or tuple")

    normalized = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("meta items must be 2-tuples")
        key, value = item
        if not isinstance(key, str):
            raise TypeError("meta keys must be str")
        if not key.strip():
            raise ValueError("meta keys must be non-empty")
        normalized.append((key, value))
    return tuple(normalized)


def check_meta(meta: MetaInput) -> Meta:
    """Return metadata in tuple form after validation."""
    return coerce_meta(meta)


__all__ = [
    "CANONICAL_REPS",
    "Meta",
    "MetaInput",
    "RepName",
    "check_dim",
    "check_dims",
    "check_meta",
    "check_prob_vector",
    "check_probability",
    "check_rep",
    "check_state_ref",
    "check_targets",
    "coerce_meta",
    "normalize_rep",
]
