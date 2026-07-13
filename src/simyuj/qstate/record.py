from __future__ import annotations

"""Immutable records used by the qstate ownership store.

Records bind opaque representation payloads to tensor layouts and metadata.
They intentionally validate only the structural fields needed by the store; the
payload's numerical validity is checked by representation-specific code.
"""

from dataclasses import dataclass, field
from typing import Any

from .check import Meta, MetaInput, check_rep, coerce_meta
from .ids import StateRef
from .space.layout import StateLayout


@dataclass(frozen=True, slots=True)
class SubsystemLocation:
    """Location of one subsystem inside a live state record.

    Parameters
    ----------
    state_ref : StateRef
        Store-assigned state reference that owns the subsystem.
    axis : int
        Tensor axis of the subsystem in the owning record layout.
    dim : int
        Hilbert-space dimension of the subsystem.

    Notes
    -----
    The dataclass is frozen. The constructor validates scalar shape only; it
    does not check that the referenced state is currently live.
    """

    state_ref: StateRef
    axis: int
    dim: int

    def __post_init__(self) -> None:
        """Validate scalar location fields after dataclass construction."""
        if type(self.state_ref) is not int:
            raise TypeError("state_ref must be int")
        if self.state_ref < 0:
            raise ValueError("state_ref must be non-negative")
        if type(self.axis) is not int:
            raise TypeError("axis must be int")
        if self.axis < 0:
            raise ValueError("axis must be non-negative")
        if type(self.dim) is not int:
            raise TypeError("dim must be int")
        if self.dim <= 0:
            raise ValueError("dim must be positive")


@dataclass(frozen=True, slots=True)
class QuantumStateRecord:
    """Immutable payload, representation, layout, and metadata bundle.

    Parameters
    ----------
    payload : Any
        Representation-specific state object. The record stores it without
        copying or validating array contents.
    rep : str
        Representation name accepted by :func:`simyuj.qstate.check.check_rep`.
    layout : StateLayout
        Tensor layout describing the payload axes.
    meta : MetaInput, optional
        Optional metadata stored as an immutable tuple of key-value pairs.

    Notes
    -----
    The constructor normalizes ``rep`` and ``meta`` but does not check that the
    payload shape matches ``layout``. Use the state invariant helpers or manager
    workflows for payload-layout compatibility checks.
    """

    payload: Any
    rep: str
    layout: StateLayout
    meta: MetaInput = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize representation and metadata after construction."""
        if not isinstance(self.layout, StateLayout):
            raise TypeError("layout must be StateLayout")
        object.__setattr__(self, "rep", check_rep(self.rep))
        object.__setattr__(self, "meta", coerce_meta(self.meta))

    @classmethod
    def _from_trusted(
        cls,
        *,
        payload: Any,
        rep: str,
        layout: StateLayout,
        meta: Meta,
    ) -> "QuantumStateRecord":
        """Build an internally checked record without constructor validation."""
        record = object.__new__(cls)
        object.__setattr__(record, "payload", payload)
        object.__setattr__(record, "rep", rep)
        object.__setattr__(record, "layout", layout)
        object.__setattr__(record, "meta", meta)
        return record


__all__ = ["Meta", "QuantumStateRecord", "SubsystemLocation"]
