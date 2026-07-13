from __future__ import annotations

"""Logical subsystem identifiers used by quantum-state layouts."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubsystemId:
    """Stable identity for one logical discrete-variable subsystem.

    Parameters
    ----------
    name : str
        Non-empty subsystem name.  Surrounding whitespace is allowed in storage
        but a whitespace-only name is rejected.

    Notes
    -----
    Equality and hashing come from the frozen dataclass fields, so two
    ``SubsystemId`` instances with the same ``name`` are treated as the same
    logical subsystem.
    """

    name: str

    def __post_init__(self) -> None:
        """Validate the subsystem name after dataclass construction."""
        if not isinstance(self.name, str):
            raise TypeError("subsystem name must be str")
        if not self.name.strip():
            raise ValueError("subsystem name must be non-empty")

    def __str__(self) -> str:
        """Return the subsystem name."""
        return self.name


__all__ = ["SubsystemId"]
