from __future__ import annotations

"""Positive-operator-valued measurements for ket and density payloads.

``POVMElement`` stores an effect ``E`` and a collapse operator ``M`` satisfying
``M†M = E``.  Measurement routines expand each local effect or operator onto the
requested qubit axes and sample from the Born-rule probabilities.
"""

from dataclasses import dataclass
from math import log2
from typing import Any

import numpy as np

from simyuj.primitives.validation import validate_bool

from ..errors import DimensionError, MeasurementError
from ..math.const import ATOL
from ..math.linalg import dagger, is_hermitian, is_psd, readonly, trace
from ..math.tensor import expand_operator
from ..state.convert import ket_to_density
from ..state.density import DensityState
from ..state.ket import KetState
from .result import POVMProbabilityTable, POVMResult
from .sample import normalize_probs, sample_probs


@dataclass(frozen=True, slots=True, init=False)
class POVMElement:
    """Single POVM outcome effect and collapse operator.

    Parameters
    ----------
    label : str
        Outcome label.  The stored value is stripped but keeps its original
        case.
    effect : object
        Square Hermitian positive-semidefinite effect matrix with power-of-two
        dimension.
    op : object, optional
        Collapse operator ``M`` with the same shape as ``effect`` and satisfying
        ``M†M = effect`` within ``ATOL``.  If omitted, the positive square root
        of ``effect`` is used.

    Attributes
    ----------
    label : str
        Non-empty outcome label.
    effect : ndarray of complex
        Read-only effect matrix.
    op : ndarray of complex
        Read-only collapse operator.
    arity : int
        Number of qubit axes implied by the effect dimension.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If ``label`` is empty after stripping.
    DimensionError
        If ``effect`` is not square, its dimension is not a positive power of
        two, or ``op`` shape does not match ``effect``.
    MeasurementError
        If ``effect`` is not Hermitian or positive semidefinite, or if ``op``
        does not satisfy ``M†M = effect``.
    """

    label: str
    effect: np.ndarray
    op: np.ndarray

    def __init__(self, label: str, effect: object, op: object | None = None) -> None:
        """Validate and store one POVM element."""
        if not isinstance(label, str):
            raise TypeError("label must be str")
        checked_label = label.strip()
        if not checked_label:
            raise ValueError("label must be non-empty")

        checked_effect = _check_effect(effect)
        checked_op = (
            _sqrt_psd(checked_effect) if op is None else _check_op(op, checked_effect)
        )

        object.__setattr__(self, "label", checked_label)
        object.__setattr__(self, "effect", readonly(checked_effect))
        object.__setattr__(self, "op", readonly(checked_op))

    @property
    def arity(self) -> int:
        """Number of qubit operands for this effect."""
        return _infer_qubit_arity(int(self.effect.shape[0]))


@dataclass(frozen=True, slots=True, init=False)
class POVM:
    """Validated collection of POVM elements.

    Parameters
    ----------
    elements : tuple of POVMElement
        Non-empty tuple of elements.  All entries must have the same arity and
        unique labels.
    name : str, default="povm"
        Human-readable POVM name.  The stored value is stripped but keeps its
        original case.

    Attributes
    ----------
    elements : tuple of POVMElement
        Outcome elements in sampling order.
    name : str
        Non-empty POVM name.
    arity : int
        Number of measured qubit axes required by this POVM.
    labels : tuple of str
        Element labels in outcome-index order.

    Raises
    ------
    TypeError
        If ``elements`` is not a tuple, any entry is not a ``POVMElement``, or
        ``name`` is not a string.
    ValueError
        If ``elements`` is empty, labels are duplicated, or ``name`` is empty.
    DimensionError
        If element arities do not match.
    MeasurementError
        If the element effects do not sum to identity within ``ATOL``.

    Examples
    --------
    >>> import numpy as np
    >>> from simyuj.qstate.measure.povm import POVM, POVMElement
    >>> from simyuj.qstate.measure.povm import measure_povm_ket
    >>> from simyuj.qstate.state import basis
    >>> z_povm = POVM((
    ...     POVMElement("zero", np.diag([1.0, 0.0])),
    ...     POVMElement("one", np.diag([0.0, 1.0])),
    ... ), name="z")
    >>> z_povm.labels
    ('zero', 'one')
    >>> measure_povm_ket(basis("0"), z_povm, axes=(0,)).label
    'zero'
    """

    elements: tuple[POVMElement, ...]
    name: str
    arity: int

    def __init__(
        self,
        elements: tuple[POVMElement, ...],
        name: str = "povm",
    ) -> None:
        """Validate and store a POVM definition."""
        if not isinstance(elements, tuple):
            raise TypeError("elements must be tuple")
        if not elements:
            raise ValueError("elements must be non-empty")
        for element in elements:
            if not isinstance(element, POVMElement):
                raise TypeError("elements entries must be POVMElement")

        if not isinstance(name, str):
            raise TypeError("name must be str")
        checked_name = name.strip()
        if not checked_name:
            raise ValueError("name must be non-empty")

        arity = elements[0].arity
        if any(element.arity != arity for element in elements):
            raise DimensionError("POVM element arities must match")

        labels = tuple(element.label for element in elements)
        if len(set(labels)) != len(labels):
            raise ValueError("POVM labels must be unique")

        size = int(elements[0].effect.shape[0])
        effect_sum = np.zeros((size, size), dtype=np.complex128)
        for element in elements:
            effect_sum = effect_sum + element.effect
        if not np.allclose(effect_sum, np.eye(size), atol=ATOL, rtol=ATOL):
            raise MeasurementError("POVM effects must sum to identity")

        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "name", checked_name)
        object.__setattr__(self, "arity", arity)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return element labels in outcome-index order."""
        return tuple(element.label for element in self.elements)


def measure_povm_ket(
    state: KetState,
    povm: POVM,
    *,
    axes: tuple[int, ...],
    rng: Any | None = None,
    collapse: bool = True,
) -> POVMResult:
    """Measure a ket state with a POVM after density conversion.

    Parameters
    ----------
    state : KetState
        Ket payload to measure.
    povm : POVM
        POVM whose arity must match ``len(axes)``.
    axes : tuple of int
        Target axes in POVM operand order.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed density payload in ``post_state``.

    Returns
    -------
    POVMResult
        POVM outcome index, label, probability table, and optional
        post-measurement density state.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState`` or delegated density measurement
        validation fails.
    MeasurementError
        If delegated POVM or axis validation fails.
    ValueError
        If probabilistic sampling is requested without an RNG.

    Notes
    -----
    The implementation converts ``state`` with ``ket_to_density`` and delegates
    to :func:`measure_povm_density`, so the returned post-state representation is
    density, not ket.
    """
    if not isinstance(state, KetState):
        raise TypeError("state must be KetState")
    return measure_povm_density(
        ket_to_density(state),
        povm,
        axes=axes,
        rng=rng,
        collapse=collapse,
    )


def measure_povm_density(
    state: DensityState,
    povm: POVM,
    *,
    axes: tuple[int, ...],
    rng: Any | None = None,
    collapse: bool = True,
) -> POVMResult:
    """Measure a density state with a POVM.

    Parameters
    ----------
    state : DensityState
        Density payload to measure.
    povm : POVM
        POVM whose arity must match ``len(axes)``.
    axes : tuple of int
        Target axes in POVM operand order.  Axis ``0`` is the most significant
        computational-basis bit.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed ``DensityState`` in ``post_state``.

    Returns
    -------
    POVMResult
        POVM outcome index, label, probability table, and optional collapsed
        density state.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``, ``povm`` is not a ``POVM``,
        ``axes`` is not a tuple of ints, or ``collapse`` is not ``bool``.
    MeasurementError
        If ``len(axes)`` does not match ``povm.arity``, or axes are duplicated
        or out of range.
    DimensionError
        If expanding a POVM element onto ``axes`` finds incompatible dimensions.
    ValueError
        If probabilistic sampling is requested without an RNG.

    Notes
    -----
    Collapse applies the selected local operator ``M`` as
    :math:`M\\rho M^\\dagger / p` after expanding it to the full Hilbert space.
    The divisor is the unnormalized Born-rule probability computed from
    :math:`\\operatorname{Tr}(E\\rho)`.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    if not isinstance(povm, POVM):
        raise TypeError("povm must be POVM")
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != povm.arity:
        raise MeasurementError("POVM measurement axes must match POVM arity")
    _check_axes(axes, state.num_qubits)
    validate_bool(collapse, field_name="collapse")

    effects = tuple(
        expand_operator(element.effect, axes=axes, num_qubits=state.num_qubits)
        for element in povm.elements
    )
    raw_probabilities = tuple(
        float(trace(effect @ state.rho).real) for effect in effects
    )
    probabilities = normalize_probs(raw_probabilities)
    outcome = sample_probs(probabilities, rng=rng)
    probability = probabilities[outcome]

    post_state = None
    if collapse:
        op = expand_operator(
            povm.elements[outcome].op,
            axes=axes,
            num_qubits=state.num_qubits,
        )
        post_state = DensityState._from_trusted(
            op @ state.rho @ dagger(op) / raw_probabilities[outcome]
        )

    return POVMResult(
        outcome=outcome,
        label=povm.elements[outcome].label,
        probability=probability,
        probabilities=_probability_table(povm, probabilities),
        post_state=post_state,
        collapsed=collapse,
    )


def _probability_table(
    povm: POVM,
    probabilities: tuple[float, ...],
) -> POVMProbabilityTable:
    """Pair POVM element labels with probabilities."""
    return tuple(zip(povm.labels, probabilities))


def _check_axes(axes: tuple[int, ...], num_qubits: int) -> None:
    """Validate unique in-range axes for a density payload."""
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    for axis in axes:
        if type(axis) is not int:
            raise TypeError("axes entries must be int")
        if axis < 0 or axis >= num_qubits:
            raise MeasurementError("axes entries must be in range")
    if len(set(axes)) != len(axes):
        raise MeasurementError("axes entries must be unique")


def _check_effect(effect: object) -> np.ndarray:
    """Validate a Hermitian positive-semidefinite effect matrix."""
    array = np.asarray(effect, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise DimensionError("effect must be square")
    _infer_qubit_arity(int(array.shape[0]))
    if not is_hermitian(array):
        raise MeasurementError("effect must be Hermitian")
    if not is_psd(array):
        raise MeasurementError("effect must be positive semidefinite")
    return array


def _check_op(op: object, effect: np.ndarray) -> np.ndarray:
    """Validate a collapse operator against an effect matrix."""
    array = np.asarray(op, dtype=np.complex128)
    if array.shape != effect.shape:
        raise DimensionError("op shape must match effect")
    if not np.allclose(dagger(array) @ array, effect, atol=ATOL, rtol=ATOL):
        raise MeasurementError("op must satisfy M†M = E")
    return array


def _sqrt_psd(effect: np.ndarray) -> np.ndarray:
    """Return the positive square root of a PSD effect matrix."""
    eigvals, eigvecs = np.linalg.eigh(effect)
    clipped = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(clipped)) @ dagger(eigvecs)


def _infer_qubit_arity(size: int) -> int:
    """Infer qubit arity from a positive power-of-two dimension."""
    if type(size) is not int or size <= 0:
        raise DimensionError("size must be a positive int")
    arity = int(log2(size))
    if 2**arity != size:
        raise DimensionError("size must be a power of two")
    return arity


__all__ = ["POVM", "POVMElement", "measure_povm_density", "measure_povm_ket"]
