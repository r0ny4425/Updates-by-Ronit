"""Signal envelope types used by component transport paths.

The signal module defines immutable records for in-flight simulator signals.
Signals carry identity, kind, encoding, optional qstate references and
subsystem targets, and structured metadata. They do not own quantum-state math;
state operations remain in qstate stores, managers, and component code that
consumes the references carried here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypeAlias, Union
from uuid import UUID

from simyuj.primitives.ids import _require_optional_correlation_id
from simyuj.primitives.subsystems import SubsystemHandle


class SignalKind(Enum):
    """
    Category of physical signal represented by a :class:`Signal`.

    The enum values are string labels used in metadata, logs, and tests.
    """

    PHOTON = "photon"
    PULSE = "pulse"
    ENTANGLED_MEMBER = "entangled_member"


class EncodingScheme(Enum):
    """
    Encoding basis used to interpret a signal's carried quantum information.

    The scheme is descriptive transport metadata. The signal record does not
    apply encoding-specific quantum operations itself.
    """

    PHASE = "phase"
    POLARIZATION = "polarization"
    FREQUENCY = "frequency"
    TIME_BIN = "time_bin"


# descriptive type aliases
# capitalization is intentional, following Python type alias conventions.
ProtocolParams: TypeAlias = tuple[tuple[str, Any], ...]
"""Immutable outer tuple for protocol-level signal metadata."""

Meta: TypeAlias = tuple[tuple[str, Any], ...]
"""Immutable outer tuple for general signal metadata."""


@dataclass(frozen=True, slots=True)
class Signal:
    """
    Immutable transport envelope for simulator signals.

    ``Signal`` is the unit passed between source, channel, memory, and detector
    components. It carries the identifiers and metadata those components need
    to locate qstate records, target subsystems, and preserve protocol/timing
    context during event-driven transport.

    Parameters
    ----------
    id : int or str or UUID
        Unique signal identifier chosen by the producing component.
    signal_kind : SignalKind
        Physical signal category.
    encoding_scheme : EncodingScheme
        Encoding scheme used by the producing component.
    emission_time : int
        Non-negative simulation tick at which the signal was emitted.
    origin : str
        Non-empty source identifier for the component that produced the signal.
    wavelength_nm : int or float, default=1550.0
        Positive carrier wavelength in nanometers.
    correlation_id : int or UUID, optional
        Identifier shared by related signals, such as entangled-pair members.
    correlation_meta : tuple, optional
        Free-form tuple metadata describing the correlation relationship.
    state_ref : int, optional
        Reference into timeline-owned qstate storage.
    state_targets : tuple[SubsystemHandle, ...], optional
        Explicit subsystem handles targeted by qstate-backed operations.
    protocol_params : ProtocolParams, optional
        Protocol-level metadata as ``(key, value)`` pairs with string keys.
    meta : Meta, optional
        General simulator metadata as ``(key, value)`` pairs with string keys.
    timing_meta : Meta, optional
        Timing/debug metadata as ``(key, value)`` pairs with string keys.
    validation_flag : bool, default=True
        When ``False``, skip all construction-time validation.

    Raises
    ------
    TypeError
        If a validated field has the wrong container or member type.
    ValueError
        If ``emission_time`` is negative, ``wavelength_nm`` is not positive, or
        ``origin`` is not a non-empty string.

    Notes
    -----
    The dataclass is frozen and slot-backed. Equality compares every field.
    Hashing works when all field values contained in tuple fields are hashable.

    ``state_ref`` is storage-only metadata produced by the qstate or timeline
    layer; user code should preserve it rather than inventing a new reference.
    For shared states, qstate-backed components use ``state_targets`` to
    identify the subsystem(s) affected by an operation. ``correlation_id`` links
    related signals but is not a subsystem identifier.

    Metadata containers are validated only at the outer tuple shape and string
    key level. Values are accepted as supplied and are not copied or
    recursively validated.

    ``protocol_params`` currently accepts free-form values. Existing callers use
    namespaced symbolic keys such as ``"bb84.basis"`` and ``"timebin.index"``.

    ``validation_flag`` is checked before all other validation. Passing a falsey
    value skips validation; the flag itself is not type-checked.

    Examples
    --------
    >>> target = SubsystemHandle(label="source:photon:0", kind="qubit", index=0)
    >>> signal = Signal(
    ...     id="sig-1",
    ...     signal_kind=SignalKind.PHOTON,
    ...     encoding_scheme=EncodingScheme.POLARIZATION,
    ...     emission_time=0,
    ...     origin="source",
    ...     state_ref=3,
    ...     state_targets=(target,),
    ...     protocol_params=(("bb84.basis", "Z"),),
    ... )
    >>> signal.state_targets[0].label
    'source:photon:0'
    """

    id: Union[int, str, UUID]
    "Unique identifier for the signal."

    signal_kind: SignalKind
    "Physical category of the signal."

    encoding_scheme: EncodingScheme
    "Encoding scheme used to interpret the signal."

    emission_time: int
    "Simulation tick when the signal was emitted."

    origin: str
    "Identifier of the component that created this signal."

    wavelength_nm: float = 1550.0
    "Carrier wavelength in nanometers."

    correlation_id: Optional[Union[int, UUID]] = None
    "Optional identifier linking correlated signals."

    correlation_meta: Optional[tuple] = None
    "Optional tuple metadata describing the correlation relationship."

    state_ref: Optional[int] = None
    "Optional reference into timeline-owned qstate storage."

    state_targets: tuple[SubsystemHandle, ...] = field(default_factory=tuple)
    """Explicit subsystem handles targeted by this signal.

    For shared states this provides the machine-readable subsystem identity that
    transport/device components must use for state operations."""

    protocol_params: ProtocolParams = field(default_factory=tuple)
    "Protocol-level metadata as immutable ``(key, value)`` pairs."

    meta: Meta = field(default_factory=tuple)
    "General simulator metadata as immutable ``(key, value)`` pairs."

    timing_meta: Meta = field(default_factory=tuple)
    "Timing/debug metadata as immutable ``(key, value)`` pairs."

    validation_flag: bool = True
    "Whether to run construction-time validation."

    def __post_init__(self):

        # Fast-path constructor:
        # When validation_flag is False, all structural and semantic validation
        # in __post_init__ is intentionally skipped for performance reasons.
        if not self.validation_flag:
            return

        if not isinstance(self.signal_kind, SignalKind):
            raise TypeError(
                "signal_kind must be SignalKind enum member,"
                f"got {type(self.signal_kind).__name__}. "
            )

        if not isinstance(self.encoding_scheme, EncodingScheme):
            raise TypeError(
                "encoding_scheme must be EncodingScheme enum member, "
                f"got {type(self.encoding_scheme).__name__}. "
            )

        if not isinstance(self.protocol_params, tuple):
            raise TypeError("protocol_params must be a tuple")

        for item in self.protocol_params:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("protocol_params must be a tuple of (key, value) pairs")
            if not isinstance(item[0], str):
                raise TypeError("protocol_params keys must be strings")

        if not isinstance(self.meta, tuple):
            raise TypeError("meta must be a tuple")

        for item in self.meta:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("meta must be a tuple of (key, value) pairs")
            if not isinstance(item[0], str):
                raise TypeError("meta keys must be strings")

        if not isinstance(self.timing_meta, tuple):
            raise TypeError("timing_meta must be a tuple")

        for item in self.timing_meta:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("timing_meta must be a tuple of (key, value) pairs")
            if not isinstance(item[0], str):
                raise TypeError("timing_meta keys must be strings")

        if not isinstance(self.emission_time, int):
            raise TypeError("emission_time must be int")
        if self.emission_time < 0:
            raise ValueError("emission_time cannot be negative")

        if not isinstance(self.wavelength_nm, (int, float)):
            raise TypeError("wavelength_nm must be a number (float)")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")

        object.__setattr__(
            self,
            "correlation_id",
            _require_optional_correlation_id(
                self.correlation_id,
                field_name="correlation_id",
            ),
        )

        if self.correlation_meta is not None and not isinstance(
            self.correlation_meta, tuple
        ):
            raise TypeError("correlation_meta must be a tuple or None")

        if self.state_ref is not None and not isinstance(self.state_ref, int):
            raise TypeError("state_ref must be int or None")

        if not isinstance(self.state_targets, tuple):
            raise TypeError("state_targets must be tuple[SubsystemHandle, ...]")
        for handle in self.state_targets:
            if not isinstance(handle, SubsystemHandle):
                raise TypeError("state_targets must contain SubsystemHandle instances")

        if not isinstance(self.id, (int, str, UUID)):
            raise TypeError("id must be int, str, or UUID")

        if not isinstance(self.origin, str) or not self.origin:
            raise ValueError("origin must be a non-empty string")

    def _with_metadata(
        self,
        *,
        meta: Meta,
        timing_meta: Meta,
    ) -> "Signal":
        """Return an internally annotated signal without revalidating fields."""
        signal = object.__new__(type(self))
        object.__setattr__(signal, "id", self.id)
        object.__setattr__(signal, "signal_kind", self.signal_kind)
        object.__setattr__(signal, "encoding_scheme", self.encoding_scheme)
        object.__setattr__(signal, "emission_time", self.emission_time)
        object.__setattr__(signal, "origin", self.origin)
        object.__setattr__(signal, "wavelength_nm", self.wavelength_nm)
        object.__setattr__(signal, "correlation_id", self.correlation_id)
        object.__setattr__(signal, "correlation_meta", self.correlation_meta)
        object.__setattr__(signal, "state_ref", self.state_ref)
        object.__setattr__(signal, "state_targets", self.state_targets)
        object.__setattr__(signal, "protocol_params", self.protocol_params)
        object.__setattr__(signal, "meta", meta)
        object.__setattr__(signal, "timing_meta", timing_meta)
        object.__setattr__(signal, "validation_flag", self.validation_flag)
        return signal
