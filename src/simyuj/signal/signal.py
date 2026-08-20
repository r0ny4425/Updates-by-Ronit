"""Signal envelope types used by component transport paths.

The signal module defines immutable records for in-flight simulator signals.
Signals carry identity, kind, encoding, optional qstate references and
subsystem targets, and structured metadata. They do not own quantum-state math;
state operations remain in qstate stores, managers, and component code that
consumes the references carried here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from math import isfinite
from typing import Any, Optional, TypeAlias, Union
from uuid import UUID

from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.ids import _require_optional_correlation_id
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.primitives.validation import require_optional_positive_real

_POLARIZATION_NORM_ATOL = 1e-12
"""Absolute tolerance on ``|u_H|**2 + |u_V|**2 == 1`` for a Jones vector."""


class _Keep:
    """Sentinel type meaning "leave this field alone" in ``Signal._derived``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "_KEEP"


_KEEP = _Keep()
"""Explicit "no change" marker accepted by :meth:`Signal._derived`.

Omitting a field name from the call already means "keep it", so this sentinel is
only needed by a caller that computes a value which may or may not be a
replacement. It exists because ``None`` is a legal value for ``coherent_state``
and ``polarization``: clearing an optical amplitude and preserving one must stay
distinguishable, and a bare ``None`` cannot express both.
"""


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

    # The three optical fields below are appended at the end of the field list
    # rather than grouped next to ``wavelength_nm``, where they belong
    # conceptually. Appending makes "is any call site constructing Signal
    # positionally?" unanswerable rather than answered once -- including for a
    # call site added later. Do not tidy them into place.

    coherent_state: Optional[CoherentState] = None
    """Optical amplitude carried by this signal, or ``None``.

    A coherent pulse is not a qubit: a signal carrying this has no ``state_ref``
    and no ``state_targets``. Mean photon number and phase are derived from
    :class:`~simyuj.primitives.coherent_state.CoherentState`, never stored
    beside it."""

    temporal_mode_sigma_s: Optional[float] = None
    """Field-envelope standard deviation in seconds, or ``None``.

    Defined by ``f(t) = (pi*sigma**2)**-0.25 * exp(-(t-t0)**2 / (2*sigma**2))``
    with ``integral |f|**2 == 1``, so this is the **field** envelope's standard
    deviation and not the intensity envelope's.

    A property of the occupied mode rather than of the state occupying it, which
    is why it sits beside ``wavelength_nm`` rather than inside
    ``coherent_state``. It is **not** converted with ``seconds_to_ticks``: that
    helper rounds to integer picoseconds, which would quantize any overlap
    computed from it. The seconds-to-ticks rule governs event times, which must
    be integers; a continuous width does not."""

    polarization: Optional[tuple[complex, complex]] = None
    """Jones vector ``(u_H, u_V)`` of the occupied polarization mode, or ``None``.

    Normalized to ``|u_H|**2 + |u_V|**2 == 1``. This is a descriptor of the mode
    the pulse occupies, **not** an independent quantum system attached to it: the
    physical field is ``(alpha*u_H, alpha*u_V)``, one amplitude times a
    direction. ``None`` means no polarization is modelled for this signal."""

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

        if self.coherent_state is not None and not isinstance(
            self.coherent_state,
            CoherentState,
        ):
            raise TypeError("coherent_state must be CoherentState or None")

        object.__setattr__(
            self,
            "temporal_mode_sigma_s",
            require_optional_positive_real(
                self.temporal_mode_sigma_s,
                field_name="temporal_mode_sigma_s",
            ),
        )

        if self.polarization is not None:
            object.__setattr__(
                self,
                "polarization",
                _normalized_polarization(self.polarization),
            )

    def _derived(self, **replacements: Any) -> "Signal":
        """Return a copy of this signal with named fields replaced.

        Parameters
        ----------
        **replacements
            Field names to substitute. Any field not named is carried over
            unchanged. Passing :data:`_KEEP` as a value is equivalent to
            omitting the name, which lets a caller compute a "replace or keep"
            value without branching at the call site.

        Returns
        -------
        Signal
            New signal with every field copied and the named ones replaced.

        Raises
        ------
        TypeError
            If a name is not a field of :class:`Signal`.

        Notes
        -----
        Construction-time validation is **not** re-run. This is an internal
        transform for component code that already holds a validated signal.

        The field list is read from ``dataclasses.fields(Signal)`` once, at
        import, rather than written out by hand. That is the whole point of the
        method: ``Signal`` is ``slots=True``, so a field that a hand-written
        copy forgot to set is left *unset*, and the first read of it raises
        ``AttributeError`` on the far side of a channel, far from the edit that
        caused it. Deriving the names makes that failure impossible rather than
        merely unlikely. ``test_derived_covers_every_field`` asserts the list
        still matches the dataclass.

        Building a fresh ``Signal(...)`` at the call site instead would be
        strictly worse: a newly added field would silently take its declared
        default at every construction site, with nothing raising at all.
        """
        unknown = tuple(
            name for name in replacements if name not in _SIGNAL_FIELD_NAMES
        )
        if unknown:
            raise TypeError(f"unknown Signal field(s): {', '.join(sorted(unknown))}")

        signal = object.__new__(type(self))
        for name in _SIGNAL_FIELD_NAMES:
            value = replacements.get(name, _KEEP)
            if value is _KEEP:
                value = getattr(self, name)
            object.__setattr__(signal, name, value)
        return signal

    def _with_metadata(
        self,
        *,
        meta: Meta,
        timing_meta: Meta,
    ) -> "Signal":
        """Return an internally annotated signal without revalidating fields.

        Thin wrapper over :meth:`_derived`, kept with an unchanged signature so
        existing transport code does not move.
        """
        return self._derived(meta=meta, timing_meta=timing_meta)


_SIGNAL_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Signal))
"""Every ``Signal`` field name, in declaration order.

Derived from the dataclass rather than written out, so a field added later is
carried through :meth:`Signal._derived` with no edit here. Guarded by a test.
"""


def _normalized_polarization(value: object) -> tuple[complex, complex]:
    """Validate a Jones vector and return it as a 2-tuple of ``complex``.

    Accepts ``int`` and ``float`` components and converts them, so ``(1.0, 0.0)``
    is a valid horizontal state. ``bool`` components are rejected.
    """
    if not isinstance(value, tuple):
        raise TypeError("polarization must be a tuple of two complex components")
    if len(value) != 2:
        raise ValueError("polarization must have exactly two components")

    resolved: list[complex] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(
            component,
            (int, float, complex),
        ):
            raise TypeError("polarization components must be int, float, or complex")
        as_complex = complex(component)
        if not isfinite(as_complex.real) or not isfinite(as_complex.imag):
            raise ValueError("polarization components must be finite")
        resolved.append(as_complex)
        del index

    norm = sum(c.real * c.real + c.imag * c.imag for c in resolved)
    if abs(norm - 1.0) > _POLARIZATION_NORM_ATOL:
        raise ValueError(
            "polarization must be normalized to |u_H|**2 + |u_V|**2 == 1, "
            f"got {norm!r}"
        )

    return (resolved[0], resolved[1])
