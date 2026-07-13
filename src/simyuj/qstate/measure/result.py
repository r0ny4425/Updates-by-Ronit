from __future__ import annotations

"""Immutable result records returned by measurement routines.

The records store sampled outcomes, probability tables, optional collapsed
post-measurement states, store references, and metadata.  They validate scalar
probabilities and result-shape fields, while leaving the physical consistency of
``post_state`` and probability tables to the measurement routine that creates
them.
"""

from dataclasses import dataclass, field

from simyuj.primitives.validation import validate_bool

from ..check import MetaInput, coerce_meta
from ..ids import StateRef

ProbabilityTable = tuple[tuple[tuple[str, ...], float], ...]
BellProbabilityTable = tuple[tuple[str, float], ...]
POVMProbabilityTable = tuple[tuple[str, float], ...]


def _check_state_ref(value: object, *, name: str) -> StateRef | None:
    """Validate an optional integer state reference."""
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{name} must be int or None")
    return value


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """Result of a projective measurement.

    Parameters
    ----------
    outcome : tuple of int
        Measured bit tuple.  Entries must be exactly integer ``0`` or ``1``.
    outcome_labels : tuple of str
        Human-readable labels for each measured bit, usually taken from the
        measurement basis.
    probability : float
        Probability assigned to the sampled outcome.  Integers and floats are
        accepted and stored as ``float``.
    probabilities : ProbabilityTable
        Probability table for all possible labeled outcomes.  This generic
        record stores the value unchanged and does not validate table shape or
        normalization.
    post_state : object, optional
        Collapsed post-measurement payload, or ``None`` when no collapse was
        requested.
    state_ref : StateRef, optional
        State reference measured by a manager-level call.
    post_state_ref : StateRef, optional
        State reference containing ``post_state`` after a manager-level
        collapse.
    collapsed : bool, default=True
        Whether the creating routine performed state collapse.
    meta : mapping or iterable, optional
        Extra metadata coerced with ``coerce_meta``.

    Attributes
    ----------
    label : str or tuple of str
        Convenience property returning the single label for one-axis results or
        the full label tuple for multi-axis results.

    Raises
    ------
    TypeError
        If outcome containers, labels, probability, references, ``collapsed``,
        or metadata have invalid types.
    ValueError
        If outcome and label lengths differ, bits are not ``0`` or ``1``,
        labels are empty, or ``probability`` is outside ``[0, 1]``.
    """

    outcome: tuple[int, ...]
    outcome_labels: tuple[str, ...]
    probability: float
    probabilities: ProbabilityTable
    post_state: object | None = None
    state_ref: StateRef | None = None
    post_state_ref: StateRef | None = None
    collapsed: bool = True
    meta: MetaInput = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize scalar fields after dataclass construction."""
        if not isinstance(self.outcome, tuple):
            raise TypeError("outcome must be tuple")
        if not isinstance(self.outcome_labels, tuple):
            raise TypeError("outcome_labels must be tuple")
        if len(self.outcome) != len(self.outcome_labels):
            raise ValueError("outcome and outcome_labels must have the same length")
        for bit in self.outcome:
            if type(bit) is not int or bit not in {0, 1}:
                raise ValueError("outcome values must be bits 0 or 1")
        for label in self.outcome_labels:
            if not isinstance(label, str):
                raise TypeError("outcome_labels entries must be str")
            if not label:
                raise ValueError("outcome_labels entries must be non-empty")

        if type(self.probability) not in {int, float}:
            raise TypeError("probability must be int or float")
        probability = float(self.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

        state_ref = _check_state_ref(self.state_ref, name="state_ref")
        post_state_ref = _check_state_ref(self.post_state_ref, name="post_state_ref")
        validate_bool(self.collapsed, field_name="collapsed")

        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "state_ref", state_ref)
        object.__setattr__(self, "post_state_ref", post_state_ref)
        object.__setattr__(self, "meta", coerce_meta(self.meta))

    @property
    def label(self) -> str | tuple[str, ...]:
        """Return the single label or label tuple for the sampled outcome."""
        if len(self.outcome_labels) == 1:
            return self.outcome_labels[0]
        return self.outcome_labels

    def _with_refs(
        self,
        *,
        state_ref: StateRef | None,
        post_state_ref: StateRef | None,
    ) -> "MeasurementResult":
        """Return an internally annotated result without revalidating fields."""
        result = object.__new__(type(self))
        object.__setattr__(result, "outcome", self.outcome)
        object.__setattr__(result, "outcome_labels", self.outcome_labels)
        object.__setattr__(result, "probability", self.probability)
        object.__setattr__(result, "probabilities", self.probabilities)
        object.__setattr__(result, "post_state", self.post_state)
        object.__setattr__(result, "state_ref", state_ref)
        object.__setattr__(result, "post_state_ref", post_state_ref)
        object.__setattr__(result, "collapsed", self.collapsed)
        object.__setattr__(result, "meta", self.meta)
        return result


@dataclass(frozen=True, slots=True)
class BellResult:
    """Result of a Bell-state measurement.

    Parameters
    ----------
    label : str
        Bell outcome label.  The stored value is stripped and lower-cased.
    outcome : tuple of int
        Two-bit Bell outcome convention, for example ``(0, 0)`` for ``phi+``.
        Entries must be exactly integer ``0`` or ``1``.
    probability : float
        Probability assigned to ``label``.  Integers and floats are accepted and
        stored as ``float``.
    probabilities : BellProbabilityTable
        Tuple of ``(label, probability)`` entries.  Labels are stripped and
        lower-cased; probabilities are stored as floats.  The record does not
        require the table to sum to one or to include ``label``.
    post_state : object, optional
        Collapsed post-measurement payload, or ``None``.
    state_ref : StateRef, optional
        State reference measured by a manager-level call.
    post_state_ref : StateRef, optional
        State reference containing ``post_state`` after a manager-level
        collapse.
    collapsed : bool, default=True
        Whether the creating routine performed state collapse.
    meta : mapping or iterable, optional
        Extra metadata coerced with ``coerce_meta``.

    Raises
    ------
    TypeError
        If label, outcome, probability table entries, references,
        ``collapsed``, or metadata have invalid types.
    ValueError
        If labels are empty, outcome shape or bit values are invalid, or a
        probability is outside ``[0, 1]``.
    """

    label: str
    outcome: tuple[int, int]
    probability: float
    probabilities: BellProbabilityTable
    post_state: object | None = None
    state_ref: StateRef | None = None
    post_state_ref: StateRef | None = None
    collapsed: bool = True
    meta: MetaInput = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize labels, scalar probabilities, references, and metadata."""
        if not isinstance(self.label, str):
            raise TypeError("label must be str")
        label = self.label.strip().lower()
        if not label:
            raise ValueError("label must be non-empty")

        if not isinstance(self.outcome, tuple):
            raise TypeError("outcome must be tuple")
        if len(self.outcome) != 2:
            raise ValueError("outcome must contain exactly two bits")
        for bit in self.outcome:
            if type(bit) is not int or bit not in {0, 1}:
                raise ValueError("outcome values must be bits 0 or 1")

        if type(self.probability) not in {int, float}:
            raise TypeError("probability must be int or float")
        probability = float(self.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

        if not isinstance(self.probabilities, tuple):
            raise TypeError("probabilities must be tuple")
        probabilities: list[tuple[str, float]] = []
        for item in self.probabilities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("probabilities entries must be 2-tuples")
            raw_label, raw_probability = item
            if not isinstance(raw_label, str):
                raise TypeError("probability labels must be str")
            probability_label = raw_label.strip().lower()
            if not probability_label:
                raise ValueError("probability labels must be non-empty")
            if type(raw_probability) not in {int, float}:
                raise TypeError("probability values must be int or float")
            table_probability = float(raw_probability)
            if not 0.0 <= table_probability <= 1.0:
                raise ValueError("probability values must be in [0, 1]")
            probabilities.append((probability_label, table_probability))

        state_ref = _check_state_ref(self.state_ref, name="state_ref")
        post_state_ref = _check_state_ref(self.post_state_ref, name="post_state_ref")
        validate_bool(self.collapsed, field_name="collapsed")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "probabilities", tuple(probabilities))
        object.__setattr__(self, "state_ref", state_ref)
        object.__setattr__(self, "post_state_ref", post_state_ref)
        object.__setattr__(self, "meta", coerce_meta(self.meta))

    @property
    def outcome_label(self) -> str:
        """Return the normalized Bell label for the sampled outcome."""
        return self.label

    def _with_refs(
        self,
        *,
        state_ref: StateRef | None,
        post_state_ref: StateRef | None,
    ) -> "BellResult":
        """Return an internally annotated result without revalidating fields."""
        result = object.__new__(type(self))
        object.__setattr__(result, "label", self.label)
        object.__setattr__(result, "outcome", self.outcome)
        object.__setattr__(result, "probability", self.probability)
        object.__setattr__(result, "probabilities", self.probabilities)
        object.__setattr__(result, "post_state", self.post_state)
        object.__setattr__(result, "state_ref", state_ref)
        object.__setattr__(result, "post_state_ref", post_state_ref)
        object.__setattr__(result, "collapsed", self.collapsed)
        object.__setattr__(result, "meta", self.meta)
        return result


@dataclass(frozen=True, slots=True)
class POVMResult:
    """Result of a POVM measurement.

    Parameters
    ----------
    outcome : int
        Zero-based index of the sampled POVM element.
    label : str
        Label associated with ``outcome``.  The stored value is stripped but
        keeps its original case.
    probability : float
        Probability assigned to the sampled element.  Integers and floats are
        accepted and stored as ``float``.
    probabilities : POVMProbabilityTable
        Tuple of ``(label, probability)`` entries.  Labels are stripped and
        probabilities are stored as floats.  The record does not require the
        table to sum to one or to contain the sampled label.
    post_state : object, optional
        Collapsed post-measurement payload, or ``None``.
    state_ref : StateRef, optional
        State reference measured by a manager-level call.
    post_state_ref : StateRef, optional
        State reference containing ``post_state`` after a manager-level
        collapse.
    collapsed : bool, default=True
        Whether the creating routine performed state collapse.
    meta : mapping or iterable, optional
        Extra metadata coerced with ``coerce_meta``.

    Raises
    ------
    TypeError
        If outcome, label, probability table entries, references, ``collapsed``,
        or metadata have invalid types.
    ValueError
        If ``outcome`` is negative, labels are empty, or a probability is
        outside ``[0, 1]``.
    """

    outcome: int
    label: str
    probability: float
    probabilities: POVMProbabilityTable
    post_state: object | None = None
    state_ref: StateRef | None = None
    post_state_ref: StateRef | None = None
    collapsed: bool = True
    meta: MetaInput = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize labels, scalar probabilities, references, and metadata."""
        if type(self.outcome) is not int:
            raise TypeError("outcome must be int")
        if self.outcome < 0:
            raise ValueError("outcome must be non-negative")

        if not isinstance(self.label, str):
            raise TypeError("label must be str")
        label = self.label.strip()
        if not label:
            raise ValueError("label must be non-empty")

        if type(self.probability) not in {int, float}:
            raise TypeError("probability must be int or float")
        probability = float(self.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")

        if not isinstance(self.probabilities, tuple):
            raise TypeError("probabilities must be tuple")
        probabilities: list[tuple[str, float]] = []
        for item in self.probabilities:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("probabilities entries must be 2-tuples")
            raw_label, raw_probability = item
            if not isinstance(raw_label, str):
                raise TypeError("probability labels must be str")
            probability_label = raw_label.strip()
            if not probability_label:
                raise ValueError("probability labels must be non-empty")
            if type(raw_probability) not in {int, float}:
                raise TypeError("probability values must be int or float")
            table_probability = float(raw_probability)
            if not 0.0 <= table_probability <= 1.0:
                raise ValueError("probability values must be in [0, 1]")
            probabilities.append((probability_label, table_probability))

        state_ref = _check_state_ref(self.state_ref, name="state_ref")
        post_state_ref = _check_state_ref(self.post_state_ref, name="post_state_ref")
        validate_bool(self.collapsed, field_name="collapsed")

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "probabilities", tuple(probabilities))
        object.__setattr__(self, "state_ref", state_ref)
        object.__setattr__(self, "post_state_ref", post_state_ref)
        object.__setattr__(self, "meta", coerce_meta(self.meta))

    def _with_refs(
        self,
        *,
        state_ref: StateRef | None,
        post_state_ref: StateRef | None,
    ) -> "POVMResult":
        """Return an internally annotated result without revalidating fields."""
        result = object.__new__(type(self))
        object.__setattr__(result, "outcome", self.outcome)
        object.__setattr__(result, "label", self.label)
        object.__setattr__(result, "probability", self.probability)
        object.__setattr__(result, "probabilities", self.probabilities)
        object.__setattr__(result, "post_state", self.post_state)
        object.__setattr__(result, "state_ref", state_ref)
        object.__setattr__(result, "post_state_ref", post_state_ref)
        object.__setattr__(result, "collapsed", self.collapsed)
        object.__setattr__(result, "meta", self.meta)
        return result


__all__ = [
    "BellProbabilityTable",
    "BellResult",
    "MeasurementResult",
    "POVMProbabilityTable",
    "POVMResult",
    "ProbabilityTable",
]
