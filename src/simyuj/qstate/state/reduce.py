from __future__ import annotations

"""Density-matrix reduction, discard, reset, and axis-reordering helpers.

Reduction helpers operate on qubit axes where axis ``0`` is the most significant
state-vector axis.  Ket and Bell-diagonal inputs are converted to density
matrices before tracing out axes.
"""

import numpy as np

from ..errors import DimensionError
from ..math.tensor import kron
from .convert import bell_diag_to_density, ket_to_density
from .density import DensityState
from .ket import KetState
from .make import basis, make_ket, plus


def partial_trace(
    state: object,
    *,
    keep_axes: tuple[int, ...] | None = None,
    drop_axes: tuple[int, ...] | None = None,
) -> DensityState:
    """Trace out selected axes and return a density payload.

    Parameters
    ----------
    state : object
        ``DensityState``, ``KetState``, or ``BellDiagState``.
    keep_axes : tuple of int, optional
        Axes to retain, in the requested output order.
    drop_axes : tuple of int, optional
        Axes to trace out.  The remaining axes keep their original order.

    Returns
    -------
    DensityState
        Reduced density matrix.

    Raises
    ------
    TypeError
        If ``state`` or an axis tuple is invalid.
    ValueError
        If neither or both axis selections are provided, or if axes are
        duplicated.
    DimensionError
        If an axis is outside the payload range.
    """
    density = _as_density(state)
    if keep_axes is None and drop_axes is None:
        raise ValueError("keep_axes or drop_axes is required")
    if keep_axes is not None and drop_axes is not None:
        raise ValueError("specify either keep_axes or drop_axes, not both")

    if keep_axes is not None:
        checked_keep = _check_axes(
            keep_axes,
            density.num_qubits,
            name="keep_axes",
            allow_empty=True,
        )
    else:
        checked_drop = _check_axes(
            drop_axes,
            density.num_qubits,
            name="drop_axes",
            allow_empty=True,
        )
        checked_keep = tuple(
            axis for axis in range(density.num_qubits) if axis not in checked_drop
        )

    return keep_axes_density(density, checked_keep)


def keep_axes(state: object, axes: tuple[int, ...]) -> DensityState:
    """Return the reduced state on selected axes."""
    return partial_trace(state, keep_axes=axes)


def drop_axes(state: object, axes: tuple[int, ...]) -> DensityState:
    """Return the reduced state after tracing out selected axes."""
    return partial_trace(state, drop_axes=axes)


def discard_density(state: DensityState, *, drop_axes: tuple[int, ...]) -> DensityState:
    """Trace axes out of an existing density payload.

    Parameters
    ----------
    state : DensityState
        Density payload to reduce.
    drop_axes : tuple of int
        Axes to trace out.

    Returns
    -------
    DensityState
        Reduced density matrix.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    return partial_trace(state, drop_axes=drop_axes)


def reset_density(
    density: DensityState,
    *,
    axes: tuple[int, ...],
    state: object = "|0>",
) -> DensityState:
    """Discard axes and replace them with a fresh reset state.

    Parameters
    ----------
    density : DensityState
        Input density payload.
    axes : tuple of int
        Axes to reset.  The returned density matrix is restored to the original
        axis order.
    state : object, default="|0>"
        Ket constructor input for the reset subsystem.  ``"0"`` and ``"1"`` are
        expanded across multi-axis resets; ``"+"`` is accepted only for a
        single-axis reset through the current implementation path.

    Returns
    -------
    DensityState
        Density matrix with selected axes replaced by ``state``.

    Raises
    ------
    TypeError
        If ``density`` is not a ``DensityState`` or ``axes`` is not a tuple.
    ValueError
        If ``axes`` is empty or duplicated.
    DimensionError
        If an axis is out of range or the reset ket width does not match
        ``len(axes)``.
    """
    if not isinstance(density, DensityState):
        raise TypeError("density must be DensityState")

    checked_axes = _check_axes(axes, density.num_qubits, name="axes")
    keep = tuple(axis for axis in range(density.num_qubits) if axis not in checked_axes)

    kept_state = keep_axes_density(density, keep)
    reset_state = ket_to_density(_make_reset_ket(state, len(checked_axes)))

    if keep:
        combined = DensityState._from_trusted(kron(kept_state.rho, reset_state.rho))
        current_order = keep + checked_axes
    else:
        combined = reset_state
        current_order = checked_axes

    return reorder_density_axes(
        combined,
        current_order=current_order,
        target_order=tuple(range(density.num_qubits)),
    )


def keep_axes_density(state: DensityState, keep_axes: tuple[int, ...]) -> DensityState:
    """Reduce a density matrix to selected axes.

    Parameters
    ----------
    state : DensityState
        Density payload to reduce.
    keep_axes : tuple of int
        Axes to retain.  The order of this tuple becomes the output axis order.
        An empty tuple returns a scalar ``[[1.0]]`` density payload.

    Returns
    -------
    DensityState
        Reduced density matrix, or ``state`` itself when all axes are kept in
        their existing order.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState`` or ``keep_axes`` is not a tuple.
    ValueError
        If ``keep_axes`` contains duplicate axes.
    DimensionError
        If an axis is outside the payload range.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")

    checked_keep = _check_axes(
        keep_axes,
        state.num_qubits,
        name="keep_axes",
        allow_empty=True,
    )
    drop = tuple(axis for axis in range(state.num_qubits) if axis not in checked_keep)

    if not drop:
        return state
    if not checked_keep:
        return DensityState._from_trusted([[1.0]])

    keep_dim = 2 ** len(checked_keep)
    drop_dim = 2 ** len(drop)
    num_qubits = state.num_qubits
    tensor = state.rho.reshape((2,) * num_qubits * 2)
    permutation = (
        checked_keep
        + drop
        + tuple(axis + num_qubits for axis in checked_keep)
        + tuple(axis + num_qubits for axis in drop)
    )
    moved = np.transpose(tensor, permutation).reshape(
        keep_dim,
        drop_dim,
        keep_dim,
        drop_dim,
    )
    return DensityState._from_trusted(np.trace(moved, axis1=1, axis2=3))


def reorder_density_axes(
    state: DensityState,
    *,
    current_order: tuple[int, ...],
    target_order: tuple[int, ...],
) -> DensityState:
    """Reorder density-matrix axes by permuting bra and ket halves together.

    Parameters
    ----------
    state : DensityState
        Density payload whose current axes are labeled by ``current_order``.
    current_order : tuple of int
        Labels for the current axis order.  Labels need not be contiguous.
    target_order : tuple of int
        Requested order containing the same labels as ``current_order``.

    Returns
    -------
    DensityState
        Reordered density payload, or ``state`` itself if the orders already
        match.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState`` or either order is not a tuple.
    DimensionError
        If an order length does not match ``state.num_qubits``.
    ValueError
        If orders contain duplicates or do not contain the same labels.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    if not isinstance(current_order, tuple):
        raise TypeError("current_order must be tuple")
    if not isinstance(target_order, tuple):
        raise TypeError("target_order must be tuple")
    if len(current_order) != state.num_qubits:
        raise DimensionError("current_order length must match state")
    if len(target_order) != state.num_qubits:
        raise DimensionError("target_order length must match state")

    checked_current = _check_order(current_order, name="current_order")
    checked_target = _check_order(target_order, name="target_order")
    if set(checked_current) != set(checked_target):
        raise ValueError("current_order and target_order must contain the same axes")
    if checked_current == checked_target:
        return state

    permutation = tuple(checked_current.index(axis) for axis in checked_target)
    num_qubits = state.num_qubits
    tensor = state.rho.reshape((2,) * num_qubits * 2)
    moved = np.transpose(
        tensor,
        permutation + tuple(axis + num_qubits for axis in permutation),
    )
    return DensityState._from_trusted(moved.reshape(state.rho.shape))


def _as_density(state: object) -> DensityState:
    from .bell_diag import BellDiagState

    if isinstance(state, DensityState):
        return state
    if isinstance(state, KetState):
        return ket_to_density(state)
    if isinstance(state, BellDiagState):
        return bell_diag_to_density(state)
    raise TypeError("state must be KetState, DensityState, or BellDiagState")


def _check_axes(
    axes: tuple[int, ...] | None,
    num_qubits: int,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    if not isinstance(axes, tuple):
        raise TypeError(f"{name} must be tuple")
    if not axes and not allow_empty:
        raise ValueError(f"{name} must be non-empty")

    checked: list[int] = []
    for axis in axes:
        if type(axis) is not int:
            raise TypeError(f"{name} entries must be int")
        if axis < 0 or axis >= num_qubits:
            raise DimensionError(f"{name} entries must be in range")
        checked.append(axis)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} entries must be unique")
    return tuple(checked)


def _check_order(order: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    checked: list[int] = []
    for axis in order:
        if type(axis) is not int:
            raise TypeError(f"{name} entries must be int")
        checked.append(axis)
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} entries must be unique")
    return tuple(checked)


def _make_reset_ket(value: object, width: int) -> KetState:
    if type(width) is not int:
        raise TypeError("width must be int")
    if width <= 0:
        raise DimensionError("width must be positive")

    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"|0>", "0", "zero"}:
            ket = basis("0" * width)
        elif key in {"|1>", "1", "one"}:
            ket = basis("1" * width)
        elif width == 1 and key in {"|+>", "+", "plus"}:
            ket = plus()
        else:
            ket = make_ket(value)
    else:
        ket = make_ket(value)

    if ket.num_qubits != width:
        raise DimensionError("reset state width must match axes")
    return ket


__all__ = [
    "discard_density",
    "drop_axes",
    "keep_axes",
    "keep_axes_density",
    "partial_trace",
    "reorder_density_axes",
    "reset_density",
]
