from __future__ import annotations

"""Dimension helpers for ``qstate`` tensor layouts.

Dimensions are represented as positive Python ``int`` values.  Dimension tuples
may be empty, which represents an empty tensor layout with total Hilbert-space
dimension ``1`` under Python's product convention.
"""

from collections.abc import Iterable
from math import prod
from typing import cast

from ..check import check_dim as _check_dim
from ..check import check_dims as _check_dims


def check_dim(dim: object, *, name: str = "dim") -> int:
    """Validate one local Hilbert-space dimension.

    Parameters
    ----------
    dim : object
        Candidate dimension.  The implementation accepts values whose type is
        exactly ``int`` and rejects booleans through the same check.
    name : str, default="dim"
        Field name used in error messages.

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
    return _check_dim(dim, name=name)


def check_dims(dims: object, *, name: str = "dims") -> tuple[int, ...]:
    """Validate a tuple of local Hilbert-space dimensions.

    Parameters
    ----------
    dims : object
        Candidate dimension tuple.  The implementation requires an actual
        ``tuple``; lists and other iterables are rejected.
    name : str, default="dims"
        Field name used in error messages.

    Returns
    -------
    tuple of int
        Checked dimensions in the original order.

    Raises
    ------
    TypeError
        If ``dims`` is not a tuple or an entry is not exactly an ``int``.
    DimensionError
        If any dimension is not positive.

    Notes
    -----
    Empty tuples are accepted.  This is used for layouts with no subsystems.
    """
    return _check_dims(dims, name=name)


def total_dim(dims: object) -> int:
    """Return the product of checked dimensions.

    Parameters
    ----------
    dims : object
        Dimension tuple accepted by :func:`check_dims`.

    Returns
    -------
    int
        Product of all dimensions.  For an empty tuple this returns ``1``.

    Raises
    ------
    TypeError
        If ``dims`` is not a tuple or an entry is not exactly an ``int``.
    DimensionError
        If any dimension is not positive.
    """
    return prod(check_dims(dims))


def qubit_dims(n: object) -> tuple[int, ...]:
    """Return the dimension tuple for ``n`` qubits.

    Parameters
    ----------
    n : object
        Number of qubits.  Must be exactly an ``int`` and may be zero.

    Returns
    -------
    tuple of int
        Tuple ``(2,) * n``.

    Raises
    ------
    TypeError
        If ``n`` is not exactly an ``int``.
    ValueError
        If ``n`` is negative.
    """
    if type(n) is not int:
        raise TypeError("n must be int")
    if n < 0:
        raise ValueError("n must be non-negative")
    return (2,) * n


def concat_dims(*parts: tuple[int, ...]) -> tuple[int, ...]:
    """Concatenate checked dimension tuples.

    Parameters
    ----------
    *parts : tuple of int
        Dimension tuples.  Each part is validated with :func:`check_dims`.

    Returns
    -------
    tuple of int
        Dimensions from all parts in left-to-right order.

    Raises
    ------
    TypeError
        If any part is not a tuple or contains a non-``int`` entry.
    DimensionError
        If any dimension is not positive.
    """
    dims: list[int] = []
    for index, part in enumerate(parts):
        dims.extend(check_dims(part, name=f"parts[{index}]"))
    return tuple(dims)


def remove_dims(dims: object, axes: object) -> tuple[int, ...]:
    """Return dimensions with selected axes removed.

    Parameters
    ----------
    dims : object
        Dimension tuple accepted by :func:`check_dims`.
    axes : object
        Iterable of integer axes to remove.

    Returns
    -------
    tuple of int
        Dimensions whose indices are not listed in ``axes``.

    Raises
    ------
    TypeError
        If ``dims`` is not a tuple, ``axes`` is not iterable, or an axis is not
        exactly an ``int``.
    DimensionError
        If any dimension is not positive.
    ValueError
        If an axis is outside ``range(len(dims))``.

    Notes
    -----
    Duplicate axes are accepted and have the same effect as listing the axis
    once.
    """
    checked_dims = check_dims(dims)
    try:
        axis_tuple: tuple[object, ...] = tuple(cast(Iterable[object], axes))
    except TypeError:
        raise TypeError("axes must be iterable") from None

    remove = set()
    for axis in axis_tuple:
        if type(axis) is not int:
            raise TypeError("axes must be ints")
        if axis < 0 or axis >= len(checked_dims):
            raise ValueError("axes must be in range")
        remove.add(axis)
    return tuple(dim for index, dim in enumerate(checked_dims) if index not in remove)


__all__ = [
    "check_dim",
    "check_dims",
    "concat_dims",
    "qubit_dims",
    "remove_dims",
    "total_dim",
]
