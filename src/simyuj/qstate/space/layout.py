from __future__ import annotations

"""Immutable mappings between logical subsystems and tensor axes.

``StateLayout`` is the small record that keeps subsystem order and local
Hilbert-space dimensions together.  Axis ``0`` is the first subsystem in the
layout and the most significant axis used by dense state-vector reshaping.
"""

from dataclasses import dataclass
from math import prod
from typing import Iterable

from ..errors import InvalidLayoutError, SubsystemNotFoundError
from .dim import check_dims
from .subsystem import SubsystemId


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Ordered mapping from logical subsystems to tensor axes.

    Parameters
    ----------
    subsystems : iterable of SubsystemId
        Logical subsystems in tensor-axis order.  Entries must be unique.
    dims : iterable of int
        Local Hilbert-space dimensions in the same order as ``subsystems``.
        Each dimension must be positive.

    Attributes
    ----------
    subsystems : tuple of SubsystemId
        Immutable subsystem order after constructor coercion.
    dims : tuple of int
        Immutable dimension tuple after validation.

    Raises
    ------
    InvalidLayoutError
        If either input is not iterable, a dimension is invalid, the lengths do
        not match, a subsystem is not a ``SubsystemId``, or a subsystem appears
        more than once.

    Notes
    -----
    Empty layouts are valid and have ``size == 0`` and ``hilbert_dim == 1``.

    Examples
    --------
    >>> from simyuj.qstate.space import StateLayout, SubsystemId
    >>> q0 = SubsystemId("q0")
    >>> q1 = SubsystemId("q1")
    >>> layout = StateLayout((q0, q1), (2, 2))
    >>> layout.axis_of(q1)
    1
    >>> layout.dim_of(q0)
    2
    >>> tuple(str(subsystem) for subsystem in layout.without((q0,)).subsystems)
    ('q1',)
    """

    subsystems: tuple[SubsystemId, ...]
    dims: tuple[int, ...]

    def __post_init__(self) -> None:
        """Coerce constructor inputs to checked immutable tuples."""
        try:
            subsystems = tuple(self.subsystems)
        except TypeError:
            raise InvalidLayoutError("subsystems must be iterable") from None
        try:
            dims = tuple(self.dims)
        except TypeError:
            raise InvalidLayoutError("dims must be iterable") from None
        try:
            dims = check_dims(dims)
        except Exception as exc:
            raise InvalidLayoutError(str(exc)) from exc

        if any(not isinstance(subsystem, SubsystemId) for subsystem in subsystems):
            raise InvalidLayoutError("subsystems must be SubsystemId")
        if len(subsystems) != len(dims):
            raise InvalidLayoutError("subsystems and dims must have the same length")
        if len(set(subsystems)) != len(subsystems):
            raise InvalidLayoutError("subsystems must be unique")

        object.__setattr__(self, "subsystems", subsystems)
        object.__setattr__(self, "dims", dims)

    @property
    def size(self) -> int:
        """Number of logical subsystems in the layout."""
        return len(self.subsystems)

    @property
    def hilbert_dim(self) -> int:
        """Product of local Hilbert-space dimensions."""
        return prod(self.dims)

    def __len__(self) -> int:
        """Return ``size`` so layouts can be used with ``len``."""
        return self.size

    def axis_of(self, subsystem: SubsystemId) -> int:
        """Return the tensor axis for a subsystem.

        Parameters
        ----------
        subsystem : SubsystemId
            Subsystem to locate.

        Returns
        -------
        int
            Axis index in ``subsystems``.

        Raises
        ------
        TypeError
            If ``subsystem`` is not a ``SubsystemId``.
        SubsystemNotFoundError
            If ``subsystem`` is not present in the layout.
        """
        if not isinstance(subsystem, SubsystemId):
            raise TypeError("subsystem must be SubsystemId")
        try:
            return self.subsystems.index(subsystem)
        except ValueError:
            raise SubsystemNotFoundError(
                f"subsystem not in layout: {subsystem}"
            ) from None

    def axes_of(self, subsystems: Iterable[SubsystemId]) -> tuple[int, ...]:
        """Return axes for subsystems in the requested order.

        Parameters
        ----------
        subsystems : iterable of SubsystemId
            Subsystems to resolve.

        Returns
        -------
        tuple of int
            Axis indices in the same order as ``subsystems``.

        Raises
        ------
        TypeError
            If ``subsystems`` is not iterable or an entry is not a
            ``SubsystemId``.
        SubsystemNotFoundError
            If any subsystem is not present in the layout.
        """
        return tuple(self.axis_of(subsystem) for subsystem in subsystems)

    def dim_of(self, subsystem: SubsystemId) -> int:
        """Return the local dimension for a subsystem.

        Parameters
        ----------
        subsystem : SubsystemId
            Subsystem whose dimension should be returned.

        Returns
        -------
        int
            Local Hilbert-space dimension.

        Raises
        ------
        TypeError
            If ``subsystem`` is not a ``SubsystemId``.
        SubsystemNotFoundError
            If ``subsystem`` is not present in the layout.
        """
        return self.dims[self.axis_of(subsystem)]

    def subsystem_at(self, axis: object) -> SubsystemId:
        """Return the subsystem at an axis.

        Parameters
        ----------
        axis : object
            Axis index.  Must be exactly an ``int``.

        Returns
        -------
        SubsystemId
            Subsystem stored at ``axis``.

        Raises
        ------
        TypeError
            If ``axis`` is not exactly an ``int``.
        InvalidLayoutError
            If ``axis`` is out of range.
        """
        return self.subsystems[self._check_axis(axis)]

    def dim_at(self, axis: object) -> int:
        """Return the local dimension at an axis.

        Parameters
        ----------
        axis : object
            Axis index.  Must be exactly an ``int``.

        Returns
        -------
        int
            Local Hilbert-space dimension stored at ``axis``.

        Raises
        ------
        TypeError
            If ``axis`` is not exactly an ``int``.
        InvalidLayoutError
            If ``axis`` is out of range.
        """
        return self.dims[self._check_axis(axis)]

    def without(self, subsystems: Iterable[SubsystemId]) -> StateLayout:
        """Return a layout with selected subsystems removed.

        Parameters
        ----------
        subsystems : iterable of SubsystemId
            Subsystems to remove.

        Returns
        -------
        StateLayout
            New layout preserving the order of all remaining subsystems.

        Raises
        ------
        TypeError
            If ``subsystems`` is not iterable or an entry is not a
            ``SubsystemId``.
        SubsystemNotFoundError
            If any subsystem is not present in the layout.

        Notes
        -----
        Duplicate subsystems in the input are accepted and have the same effect
        as listing the subsystem once.
        """
        drop_axes = set(self.axes_of(subsystems))
        return StateLayout(
            tuple(
                subsystem
                for axis, subsystem in enumerate(self.subsystems)
                if axis not in drop_axes
            ),
            tuple(dim for axis, dim in enumerate(self.dims) if axis not in drop_axes),
        )

    def reorder(self, subsystems: Iterable[SubsystemId]) -> StateLayout:
        """Return a layout with the same subsystems in a new order.

        Parameters
        ----------
        subsystems : iterable of SubsystemId
            Requested subsystem order.

        Returns
        -------
        StateLayout
            New layout with dimensions paired to the reordered subsystems.

        Raises
        ------
        TypeError
            If ``subsystems`` is not iterable.
        InvalidLayoutError
            If the requested order has the wrong size, does not contain exactly
            the layout's subsystems, or contains duplicates.
        """
        ordered = tuple(subsystems)
        if len(ordered) != self.size:
            raise InvalidLayoutError("reordered subsystems must match layout size")
        if set(ordered) != set(self.subsystems):
            raise InvalidLayoutError(
                "reordered subsystems must match layout subsystems"
            )
        if len(set(ordered)) != len(ordered):
            raise InvalidLayoutError("reordered subsystems must be unique")
        return StateLayout(
            ordered, tuple(self.dim_of(subsystem) for subsystem in ordered)
        )

    def combine(self, other: StateLayout) -> StateLayout:
        """Return a layout formed by appending another layout.

        Parameters
        ----------
        other : StateLayout
            Layout appended after this layout.

        Returns
        -------
        StateLayout
            Combined layout with ``self`` subsystems first and ``other``
            subsystems after them.

        Raises
        ------
        TypeError
            If ``other`` is not a ``StateLayout``.
        InvalidLayoutError
            If the two layouts share any subsystem.
        """
        if not isinstance(other, StateLayout):
            raise TypeError("other must be StateLayout")
        if set(self.subsystems) & set(other.subsystems):
            raise InvalidLayoutError("layouts must not share subsystems")
        return StateLayout(self.subsystems + other.subsystems, self.dims + other.dims)

    def _check_axis(self, axis: object) -> int:
        if type(axis) is not int:
            raise TypeError("axis must be int")
        if axis < 0 or axis >= self.size:
            raise InvalidLayoutError("axis out of range")
        return axis


__all__ = ["StateLayout"]
