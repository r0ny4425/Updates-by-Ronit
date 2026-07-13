from __future__ import annotations

"""Resolve user-facing subsystem targets to layout axes.

Targets may be supplied as ``SubsystemId`` instances, subsystem names, or integer
axes.  Resolved axes always refer to the order stored by ``StateLayout``.
"""

from typing import TypeAlias, cast

from ..check import check_targets
from ..errors import SubsystemNotFoundError
from .layout import StateLayout
from .subsystem import SubsystemId

Target: TypeAlias = SubsystemId | str | int


def resolve_one(layout: StateLayout, target: Target) -> int:
    """Resolve one target to a layout axis.

    Parameters
    ----------
    layout : StateLayout
        Layout that defines subsystem and axis order.
    target : SubsystemId or str or int
        Target subsystem.  Integers are treated as axes, strings are stripped
        and matched against ``SubsystemId.name``, and ``SubsystemId`` values are
        looked up directly.

    Returns
    -------
    int
        Axis index in ``layout``.

    Raises
    ------
    TypeError
        If ``layout`` is not a ``StateLayout`` or ``target`` is not a supported
        target type.
    ValueError
        If a string target is empty after stripping whitespace.
    SubsystemNotFoundError
        If an integer axis is out of range or a subsystem/name is not present.
    """
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")

    if type(target) is int:
        if target < 0 or target >= layout.size:
            raise SubsystemNotFoundError(f"target axis not in layout: {target}")
        return target

    if isinstance(target, SubsystemId):
        return layout.axis_of(target)

    if isinstance(target, str):
        name = target.strip()
        if not name:
            raise ValueError("target name must be non-empty")
        for axis, subsystem in enumerate(layout.subsystems):
            if subsystem.name == name:
                return axis
        raise SubsystemNotFoundError(f"target subsystem not in layout: {target}")

    raise TypeError("target must be SubsystemId, str, or int")


def resolve_targets(layout: StateLayout, targets: object) -> tuple[int, ...]:
    """Resolve a non-empty target tuple to unique layout axes.

    Parameters
    ----------
    layout : StateLayout
        Layout that defines subsystem and axis order.
    targets : object
        Tuple of targets accepted by :func:`resolve_one`.

    Returns
    -------
    tuple of int
        Resolved axes in the same order as ``targets``.

    Raises
    ------
    TypeError
        If ``targets`` is not a tuple, ``layout`` is invalid, or a target has an
        unsupported type.
    ValueError
        If ``targets`` is empty, contains an empty string target, or resolves to
        duplicate axes.
    SubsystemNotFoundError
        If any target is not present in ``layout``.
    """
    checked_targets = check_targets(targets)
    axes = tuple(
        resolve_one(layout, cast(Target, target)) for target in checked_targets
    )
    check_targets_unique(axes)
    return axes


def resolve_two(layout: StateLayout, targets: object) -> tuple[int, int]:
    """Resolve exactly two targets to layout axes.

    Parameters
    ----------
    layout : StateLayout
        Layout that defines subsystem and axis order.
    targets : object
        Tuple of two targets accepted by :func:`resolve_one`.

    Returns
    -------
    tuple of int
        Two resolved axes.

    Raises
    ------
    TypeError
        If ``targets`` is not a tuple, ``layout`` is invalid, or a target has an
        unsupported type.
    ValueError
        If the resolved target count is not exactly two or targets resolve to
        duplicate axes.
    SubsystemNotFoundError
        If any target is not present in ``layout``.
    """
    axes = resolve_targets(layout, targets)
    if len(axes) != 2:
        raise ValueError("exactly two targets expected")
    return axes


def check_targets_unique(axes: tuple[int, ...]) -> None:
    """Require resolved target axes to be unique.

    Parameters
    ----------
    axes : tuple of int
        Resolved axes.

    Raises
    ------
    ValueError
        If any axis appears more than once.

    Notes
    -----
    This function assumes its input is already a tuple of resolved axis values.
    It does not validate axis type or range.
    """
    if len(set(axes)) != len(axes):
        raise ValueError("duplicate target axes")


def check_targets_in_layout(layout: StateLayout, targets: object) -> None:
    """Validate that targets resolve to unique axes in a layout.

    Parameters
    ----------
    layout : StateLayout
        Layout that defines subsystem and axis order.
    targets : object
        Tuple of targets accepted by :func:`resolve_one`.

    Raises
    ------
    TypeError
        If ``layout`` or ``targets`` is invalid.
    ValueError
        If target resolution produces duplicate axes or invalid target values.
    SubsystemNotFoundError
        If any target is not present in ``layout``.
    """
    resolve_targets(layout, targets)


__all__ = [
    "Target",
    "check_targets_in_layout",
    "check_targets_unique",
    "resolve_one",
    "resolve_targets",
    "resolve_two",
]
