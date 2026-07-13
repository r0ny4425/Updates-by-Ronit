"""
Logical subsystem identity records.

Subsystem handles identify qubits or modes carried by signals and component
payloads. They contain bookkeeping metadata only; ownership, state evolution,
and measurement behavior live in qstate and component layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .meta import Meta, validate_meta

SubsystemMetadata: TypeAlias = Meta


@dataclass(frozen=True, slots=True)
class SubsystemHandle:
    """Stable identity handle for a logical qubit or mode.

    Parameters
    ----------
    label : str
        Non-empty human-readable subsystem label.
    kind : {"qubit", "mode"}, default="qubit"
        Logical subsystem kind.
    index : int or None, default=None
        Optional non-negative position or backend index.
    metadata : SubsystemMetadata, optional
        Metadata tuple associated with the handle.

    Raises
    ------
    ValueError
        If `label` is empty, `kind` is not supported, or `index` is negative.
    TypeError
        If `index` is not ``int`` or ``None``, or if `metadata` is not a tuple
        of ``(str, object)`` pairs.

    Notes
    -----
    The dataclass is frozen and slot-backed. It does not imply that the
    represented subsystem currently exists in a qstate backend; it only records
    the identity a payload expects other layers to interpret.

    Metadata values may be unhashable when carrying opaque backend hints. In
    that case, the frozen handle may still be unhashable.
    """

    label: str
    kind: Literal["qubit", "mode"] = "qubit"
    index: int | None = None
    metadata: SubsystemMetadata = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty str")

        if self.kind not in ("qubit", "mode"):
            raise ValueError("kind must be 'qubit' or 'mode'")

        if self.index is not None:
            if type(self.index) is not int:
                raise TypeError("index must be int or None")
            if self.index < 0:
                raise ValueError("index must be non-negative")

        validate_meta(self.metadata, field_name="metadata", require_hashable=False)


__all__ = [
    "SubsystemHandle",
    "SubsystemMetadata",
]
