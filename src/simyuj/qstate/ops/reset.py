from __future__ import annotations

"""Density-state reset convenience functions.

The functions in this module delegate to ``state.reduce.reset_density``.  They
trace out selected qubit axes, prepare a replacement ket state, and restore the
original axis order in the returned density payload.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state.density import DensityState


def reset_zero(state: DensityState, *, axes: tuple[int, ...]) -> DensityState:
    """Reset selected density axes to computational :math:`|0\\ldots0\\rangle`.

    Parameters
    ----------
    state : DensityState
        Density payload to update.
    axes : tuple of int
        Non-empty axes to discard and replace.  Axis ``0`` is the most
        significant computational-basis bit.

    Returns
    -------
    DensityState
        Density payload with the selected axes reset to :math:`|0\\rangle`
        states.

    Raises
    ------
    TypeError
        If delegated density or axis validation fails.
    ValueError
        If ``axes`` is empty or contains duplicates.
    DimensionError
        If an axis is out of range.
    """
    from ..state.reduce import reset_density

    return reset_density(state, axes=axes, state="|0>")


def reset_one(state: DensityState, *, axes: tuple[int, ...]) -> DensityState:
    """Reset selected density axes to computational :math:`|1\\ldots1\\rangle`.

    Parameters
    ----------
    state : DensityState
        Density payload to update.
    axes : tuple of int
        Non-empty axes to discard and replace.  Axis ``0`` is the most
        significant computational-basis bit.

    Returns
    -------
    DensityState
        Density payload with the selected axes reset to :math:`|1\\rangle`
        states.

    Raises
    ------
    TypeError
        If delegated density or axis validation fails.
    ValueError
        If ``axes`` is empty or contains duplicates.
    DimensionError
        If an axis is out of range.
    """
    from ..state.reduce import reset_density

    return reset_density(state, axes=axes, state="|1>")


def reset_plus(state: DensityState, *, axes: tuple[int, ...]) -> DensityState:
    """Reset one selected density axis to the :math:`|+\\rangle` state.

    Parameters
    ----------
    state : DensityState
        Density payload to update.
    axes : tuple of int
        Non-empty axes to discard and replace.  The current reset path accepts
        the :math:`|+\\rangle` shortcut only when ``len(axes) == 1``.

    Returns
    -------
    DensityState
        Density payload with the selected axis reset to :math:`|+\\rangle`.

    Raises
    ------
    TypeError
        If delegated density or axis validation fails.
    ValueError
        If ``axes`` is empty or contains duplicates.
    DimensionError
        If an axis is out of range or the prepared :math:`|+\\rangle` ket width
        does not match ``len(axes)``.
    """
    from ..state.reduce import reset_density

    return reset_density(state, axes=axes, state="|+>")


def discard_and_prepare(
    state: DensityState,
    *,
    axes: tuple[int, ...],
    prepared: object = "|0>",
) -> DensityState:
    """Discard selected density axes and prepare a replacement state.

    Parameters
    ----------
    state : DensityState
        Density payload to update.
    axes : tuple of int
        Non-empty axes to discard and replace.
    prepared : object, default="|0>"
        Ket constructor input passed to ``reset_density``.  Its qubit width
        must match ``len(axes)`` after construction.

    Returns
    -------
    DensityState
        Density payload with selected axes replaced by ``prepared``.

    Raises
    ------
    TypeError
        If delegated density, axis, or ket-constructor validation fails.
    ValueError
        If ``axes`` is empty or contains duplicates.
    DimensionError
        If an axis is out of range or the prepared ket width does not match
        ``len(axes)``.
    InvalidStateError
        If ``prepared`` cannot construct a valid normalized ket.
    """
    from ..state.reduce import reset_density

    return reset_density(state, axes=axes, state=prepared)


__all__ = [
    "discard_and_prepare",
    "reset_one",
    "reset_plus",
    "reset_zero",
]
