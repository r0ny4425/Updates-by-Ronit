from __future__ import annotations

"""Bell-state vectors, projectors, and Bell-basis measurement routines.

Bell helpers use ``BELL_LABELS`` order from ``state.bell_diag``:
``("phi+", "phi-", "psi+", "psi-")``.  Dense vectors are length four in
computational-basis order :math:`|00\\rangle`, :math:`|01\\rangle`,
:math:`|10\\rangle`, :math:`|11\\rangle`.
"""

from typing import Any, TypeAlias, cast

import numpy as np

from simyuj.primitives.validation import validate_bool

from ..errors import DimensionError, InvalidLayoutError, MeasurementError
from ..math.const import SQRT2
from ..math.linalg import trace
from ..math.projector import vector_projector
from ..math.tensor import expand_operator
from ..space.layout import StateLayout
from ..state.bell_diag import (
    BELL_LABELS,
    BellLabel,
    label_to_bits,
    normalize_bell_label,
)
from ..state.check import _assert_payload_layout_compatible_trusted
from ..state.density import DensityState
from ..state.ket import KetState
from .result import BellResult
from .sample import normalize_probs, sample_probs

BellProbabilityTable: TypeAlias = tuple[tuple[BellLabel, float], ...]


def bell_vector(label: object = "phi+") -> np.ndarray:
    """Return a read-only Bell-state vector.

    Parameters
    ----------
    label : object, default="phi+"
        Bell label or alias accepted by ``normalize_bell_label``.

    Returns
    -------
    ndarray of complex, shape (4,)
        Bell vector in computational-basis order :math:`|00\\rangle`,
        :math:`|01\\rangle`, :math:`|10\\rangle`, :math:`|11\\rangle`.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If ``label`` is empty or unsupported.
    """
    normalized = normalize_bell_label(label)
    vector = np.zeros(4, dtype=np.complex128)
    if normalized == "phi+":
        vector[0] = 1.0 / SQRT2
        vector[3] = 1.0 / SQRT2
    elif normalized == "phi-":
        vector[0] = 1.0 / SQRT2
        vector[3] = -1.0 / SQRT2
    elif normalized == "psi+":
        vector[1] = 1.0 / SQRT2
        vector[2] = 1.0 / SQRT2
    else:
        vector[1] = 1.0 / SQRT2
        vector[2] = -1.0 / SQRT2
    vector.setflags(write=False)
    return vector


def bell_vectors() -> tuple[np.ndarray, ...]:
    """Return all read-only Bell vectors in ``BELL_LABELS`` order."""
    return tuple(bell_vector(label) for label in BELL_LABELS)


def bell_projector(label: object = "phi+") -> np.ndarray:
    """Return the rank-one projector for a Bell state.

    Parameters
    ----------
    label : object, default="phi+"
        Bell label or alias accepted by ``normalize_bell_label``.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Projector :math:`|B\\rangle\\langle B|` for the requested Bell state.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If ``label`` is empty or unsupported.
    """
    return vector_projector(bell_vector(label))


def bell_projectors() -> tuple[np.ndarray, ...]:
    """Return all Bell projectors in ``BELL_LABELS`` order."""
    return tuple(bell_projector(label) for label in BELL_LABELS)


def bell_density_matrix(label: object = "phi+") -> np.ndarray:
    """Return the pure density matrix for a Bell state.

    This is an alias for :func:`bell_projector`.

    Parameters
    ----------
    label : object, default="phi+"
        Bell label or alias accepted by ``normalize_bell_label``.

    Returns
    -------
    ndarray of complex, shape (4, 4)
        Pure Bell-state density matrix.
    """
    return bell_projector(label)


def measure_bell_ket(
    state: KetState,
    *,
    layout: StateLayout,
    axes: tuple[int, int],
    rng: Any | None = None,
    collapse: bool = True,
) -> BellResult:
    """Measure two ket axes in the Bell basis.

    Parameters
    ----------
    state : KetState
        Ket payload to measure.
    layout : StateLayout
        Layout describing ``state``.  It must match the payload and the
        measured axes must be qubit axes.
    axes : tuple of int
        Exactly two unique axes.  Their order defines the two-qubit operand
        order for Bell projectors.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed ``KetState`` in ``post_state``.

    Returns
    -------
    BellResult
        Bell outcome label, two-bit outcome, probability table, and optional
        post-measurement ket.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState``, ``layout`` is not a ``StateLayout``,
        ``axes`` is not a tuple of ints, or ``collapse`` is not ``bool``.
    MeasurementError
        If the Bell measurement axis tuple does not contain exactly two unique
        in-range axes.
    DimensionError
        If ``layout`` is incompatible with ``state`` or a measured axis is not
        a qubit axis.
    ValueError
        If probabilistic sampling is requested without an RNG.
    """
    if not isinstance(state, KetState):
        raise TypeError("state must be KetState")
    _check_bell_inputs(layout, axes)
    _check_payload_layout(state, layout, "ket")

    vector = cast(np.ndarray, state.vector)
    validate_bool(collapse, field_name="collapse")

    projectors = _expanded_bell_projectors(axes=axes, num_qubits=state.num_qubits)
    raw_probabilities = tuple(
        float(np.vdot(vector, projector @ vector).real) for projector in projectors
    )
    probabilities = normalize_probs(raw_probabilities)
    outcome_index = sample_probs(probabilities, rng=rng)
    probability = probabilities[outcome_index]
    label = BELL_LABELS[outcome_index]
    post_state = None
    if collapse:
        raw_probability = raw_probabilities[outcome_index]
        collapsed = projectors[outcome_index] @ vector
        post_state = KetState._from_trusted(collapsed / np.sqrt(raw_probability))

    return _bell_result(
        label=label,
        probability=probability,
        probabilities=probabilities,
        post_state=post_state,
        collapse=collapse,
    )


def measure_bell_density(
    state: DensityState,
    *,
    layout: StateLayout,
    axes: tuple[int, int],
    rng: Any | None = None,
    collapse: bool = True,
) -> BellResult:
    """Measure two density-state axes in the Bell basis.

    Parameters
    ----------
    state : DensityState
        Density payload to measure.
    layout : StateLayout
        Layout describing ``state``.  It must match the payload and the
        measured axes must be qubit axes.
    axes : tuple of int
        Exactly two unique axes.  Their order defines the two-qubit operand
        order for Bell projectors.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed ``DensityState`` in ``post_state``.

    Returns
    -------
    BellResult
        Bell outcome label, two-bit outcome, probability table, and optional
        post-measurement density matrix.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``, ``layout`` is not a
        ``StateLayout``, ``axes`` is not a tuple of ints, or ``collapse`` is not
        ``bool``.
    MeasurementError
        If the Bell measurement axis tuple does not contain exactly two unique
        in-range axes.
    DimensionError
        If ``layout`` is incompatible with ``state`` or a measured axis is not
        a qubit axis.
    ValueError
        If probabilistic sampling is requested without an RNG.

    Notes
    -----
    Collapse uses the unnormalized Born-rule probability computed from
    :math:`\\operatorname{Tr}(P\\rho)` before the probability vector is
    normalized for sampling.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    _check_bell_inputs(layout, axes)
    _check_payload_layout(state, layout, "density")

    rho = cast(np.ndarray, state.rho)
    validate_bool(collapse, field_name="collapse")

    projectors = _expanded_bell_projectors(axes=axes, num_qubits=state.num_qubits)
    raw_probabilities = tuple(
        float(trace(projector @ rho).real) for projector in projectors
    )
    probabilities = normalize_probs(raw_probabilities)
    outcome_index = sample_probs(probabilities, rng=rng)
    probability = probabilities[outcome_index]
    label = BELL_LABELS[outcome_index]
    post_state = None
    if collapse:
        raw_probability = raw_probabilities[outcome_index]
        projector = projectors[outcome_index]
        post_state = DensityState._from_trusted(
            projector @ rho @ projector / raw_probability
        )

    return _bell_result(
        label=label,
        probability=probability,
        probabilities=probabilities,
        post_state=post_state,
        collapse=collapse,
    )


def _bell_result(
    *,
    label: BellLabel,
    probability: float,
    probabilities: tuple[float, ...],
    post_state: object | None,
    collapse: bool,
) -> BellResult:
    """Build a ``BellResult`` from a label and probability vector."""
    return BellResult(
        label=label,
        outcome=label_to_bits(label),
        probability=probability,
        probabilities=tuple(zip(BELL_LABELS, probabilities)),
        post_state=post_state,
        collapsed=collapse,
    )


def _expanded_bell_projectors(
    *,
    axes: tuple[int, int],
    num_qubits: int,
) -> tuple[np.ndarray, ...]:
    """Expand Bell projectors onto selected full-state axes."""
    return tuple(
        expand_operator(projector, axes=axes, num_qubits=num_qubits)
        for projector in bell_projectors()
    )


def _check_bell_inputs(
    layout: StateLayout,
    axes: tuple[int, int],
) -> None:
    """Validate Bell-measurement layout and two-axis selection."""
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != 2:
        raise MeasurementError("Bell measurement requires exactly two axes")
    for axis in axes:
        if type(axis) is not int:
            raise TypeError("axes entries must be int")
    if len(set(axes)) != len(axes):
        raise MeasurementError("Bell measurement axes must be unique")
    for axis in axes:
        if axis < 0 or axis >= layout.size:
            raise MeasurementError("Bell measurement axes must be in layout range")
        if layout.dim_at(axis) != 2:
            raise DimensionError("Bell measurement requires qubit axes")


def _check_payload_layout(
    state: KetState | DensityState,
    layout: StateLayout,
    rep: str,
) -> None:
    """Translate payload-layout compatibility errors for Bell measurement."""
    try:
        _assert_payload_layout_compatible_trusted(state, layout)
    except InvalidLayoutError as exc:
        if "qubit axes" in str(exc):
            raise DimensionError("Bell measurement requires qubit axes") from exc
        raise DimensionError(f"layout does not match {rep} payload") from exc


__all__ = [
    "BellProbabilityTable",
    "bell_density_matrix",
    "bell_projector",
    "bell_projectors",
    "bell_vector",
    "bell_vectors",
    "measure_bell_density",
    "measure_bell_ket",
]
