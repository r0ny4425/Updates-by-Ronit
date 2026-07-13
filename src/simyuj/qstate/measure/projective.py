from __future__ import annotations

"""Projective measurement for ket and density payloads.

Measurements act on qubit axes where axis ``0`` is the most significant
computational-basis bit.  For multi-axis measurements, the order of ``axes``
defines the bit order in the sampled outcome and probability table.
"""

from itertools import product
from typing import Any, cast

import numpy as np

from simyuj.primitives.validation import validate_bool

from ..errors import DimensionError, InvalidLayoutError
from ..math.linalg import trace
from ..math.projector import vector_projector
from ..math.tensor import apply_unitary_to_axes, expand_operator, kron_all
from ..space.layout import StateLayout
from ..state.check import assert_payload_layout_compatible
from ..state.density import DensityState
from ..state.ket import KetState
from .basis import MeasurementBasis, basis_for
from .result import MeasurementResult, ProbabilityTable
from .sample import normalize_probs, sample_probs

_EXPANDED_PROJECTOR_CACHE_MAX_SIZE = 256
_EXPANDED_PROJECTOR_CACHE: dict[tuple[object, ...], tuple[np.ndarray, ...]] = {}


def measure_ket(
    state: KetState,
    *,
    layout: StateLayout,
    axes: tuple[int, ...],
    basis: str | MeasurementBasis = "z",
    rng: Any | None = None,
    collapse: bool = True,
) -> MeasurementResult:
    """Measure selected axes of a ket state in a single-qubit basis.

    Parameters
    ----------
    state : KetState
        Ket payload to measure.
    layout : StateLayout
        Layout describing the payload axes.  It must match ``state`` and all
        measured axes must be qubit axes.
    axes : tuple of int
        Non-empty unique axes to measure.  The tuple order becomes the outcome
        bit order.
    basis : str or MeasurementBasis, default="z"
        Measurement basis name or explicit single-qubit basis.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed ``KetState`` in ``post_state``.

    Returns
    -------
    MeasurementResult
        Projective measurement result with outcome bits, labels, probabilities,
        and optional post-measurement ket.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState``, ``collapse`` is not ``bool``, basis
        or layout resolution fails by type, or delegated axis checks fail by
        type.
    ValueError
        If axes are empty, duplicated, or out of range, or if probabilistic
        sampling is requested without an RNG.
    DimensionError
        If ``layout`` is incompatible with ``state`` or a measured axis is not
        a qubit axis.
    MeasurementError
        If the requested basis name is unsupported.
    """
    if not isinstance(state, KetState):
        raise TypeError("state must be KetState")
    _check_payload_layout(state, layout, "ket")
    validate_bool(collapse, field_name="collapse")
    _check_axes(layout, axes)

    return _measure_ket_checked(
        state,
        axes=axes,
        basis=basis,
        rng=rng,
        collapse=collapse,
    )


def _measure_ket_checked(
    state: KetState,
    *,
    axes: tuple[int, ...],
    basis: str | MeasurementBasis = "z",
    rng: Any | None = None,
    collapse: bool = True,
) -> MeasurementResult:
    """Measure a ket after payload, layout, and axis checks."""
    validate_bool(collapse, field_name="collapse")
    vector = cast(np.ndarray, state.vector)
    measurement_basis = basis_for(basis)
    transformed, basis_matrix = _to_measurement_coordinates(
        vector,
        measurement_basis,
        axes=axes,
        num_qubits=state.num_qubits,
    )
    probabilities = _measurement_probabilities(
        transformed,
        axes=axes,
        num_qubits=state.num_qubits,
    )
    outcome_index = sample_probs(probabilities, rng=rng)
    outcome = _index_to_bits(outcome_index, len(axes))
    probability = probabilities[outcome_index]
    post_state = None
    if collapse:
        post_state = KetState._from_trusted(
            _collapse_and_return_to_computational(
                transformed,
                basis_matrix,
                outcome_index=outcome_index,
                probability=probability,
                axes=axes,
                num_qubits=state.num_qubits,
            )
        )

    return MeasurementResult(
        outcome=outcome,
        outcome_labels=tuple(measurement_basis.labels[bit] for bit in outcome),
        probability=probability,
        probabilities=_probability_table(probabilities, measurement_basis, len(axes)),
        post_state=post_state,
        collapsed=collapse,
    )


def measure_density(
    state: DensityState,
    *,
    layout: StateLayout,
    axes: tuple[int, ...],
    basis: str | MeasurementBasis = "z",
    rng: Any | None = None,
    collapse: bool = True,
) -> MeasurementResult:
    """Measure selected axes of a density state in a single-qubit basis.

    Parameters
    ----------
    state : DensityState
        Density payload to measure.
    layout : StateLayout
        Layout describing the payload axes.  It must match ``state`` and all
        measured axes must be qubit axes.
    axes : tuple of int
        Non-empty unique axes to measure.  The tuple order becomes the outcome
        bit order.
    basis : str or MeasurementBasis, default="z"
        Measurement basis name or explicit single-qubit basis.
    rng : object, optional
        Random source passed to ``sample_probs`` for probabilistic outcomes.
        Deterministic outcomes do not require an RNG.
    collapse : bool, default=True
        Whether to return a collapsed ``DensityState`` in ``post_state``.

    Returns
    -------
    MeasurementResult
        Projective measurement result with outcome bits, labels, probabilities,
        and optional post-measurement density matrix.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``, ``collapse`` is not ``bool``,
        basis or layout resolution fails by type, or delegated axis checks fail
        by type.
    ValueError
        If axes are empty, duplicated, or out of range, or if probabilistic
        sampling is requested without an RNG.
    DimensionError
        If ``layout`` is incompatible with ``state`` or a measured axis is not
        a qubit axis.
    MeasurementError
        If the requested basis name is unsupported.

    Notes
    -----
    Collapse uses the unnormalized Born-rule probability computed from
    :math:`\\operatorname{Tr}(P\\rho)` before the probability vector is
    normalized for sampling.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    _check_payload_layout(state, layout, "density")
    validate_bool(collapse, field_name="collapse")
    _check_axes(layout, axes)

    return _measure_density_checked(
        state,
        axes=axes,
        basis=basis,
        rng=rng,
        collapse=collapse,
    )


def _measure_density_checked(
    state: DensityState,
    *,
    axes: tuple[int, ...],
    basis: str | MeasurementBasis = "z",
    rng: Any | None = None,
    collapse: bool = True,
) -> MeasurementResult:
    """Measure density state after payload, layout, and axis checks."""
    validate_bool(collapse, field_name="collapse")
    measurement_basis = basis_for(basis)

    if state.num_qubits == 1 and axes == (0,):
        return _measure_one_qubit_density_checked(
            state,
            basis=measurement_basis,
            rng=rng,
            collapse=collapse,
        )

    projectors = _expanded_projectors(
        measurement_basis,
        axes=axes,
        num_qubits=state.num_qubits,
    )
    raw_probabilities = tuple(
        float(trace(projector @ state.rho).real) for projector in projectors
    )
    probabilities = normalize_probs(raw_probabilities)
    outcome_index = sample_probs(probabilities, rng=rng)
    outcome = _index_to_bits(outcome_index, len(axes))
    probability = probabilities[outcome_index]
    post_state = None
    if collapse:
        projector = projectors[outcome_index]
        collapse_probability = raw_probabilities[outcome_index]
        post_state = DensityState._from_trusted(
            projector @ state.rho @ projector / collapse_probability
        )

    return MeasurementResult(
        outcome=outcome,
        outcome_labels=tuple(measurement_basis.labels[bit] for bit in outcome),
        probability=probability,
        probabilities=_probability_table(probabilities, measurement_basis, len(axes)),
        post_state=post_state,
        collapsed=collapse,
    )


def _measure_one_qubit_density_checked(
    state: DensityState,
    *,
    basis: MeasurementBasis,
    rng: Any | None = None,
    collapse: bool = True,
) -> MeasurementResult:
    """Fast path for one-qubit density measurement in any valid basis."""
    raw_probabilities = tuple(
        float(np.vdot(vector, state.rho @ vector).real) for vector in basis.vectors
    )
    probabilities = normalize_probs(raw_probabilities)
    outcome_index = sample_probs(probabilities, rng=rng)
    outcome = (outcome_index,)
    probability = probabilities[outcome_index]
    post_state = None
    if collapse:
        vector = basis.vectors[outcome_index]
        projector = np.outer(vector, np.conjugate(vector))
        collapse_probability = raw_probabilities[outcome_index]
        post_state = DensityState._from_trusted(
            projector @ state.rho @ projector / collapse_probability
        )

    return MeasurementResult(
        outcome=outcome,
        outcome_labels=(basis.labels[outcome_index],),
        probability=probability,
        probabilities=_probability_table(probabilities, basis, 1),
        post_state=post_state,
        collapsed=collapse,
    )


def _to_measurement_coordinates(
    vector: np.ndarray,
    basis: MeasurementBasis,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate selected ket axes into the requested measurement basis."""
    basis_matrix = kron_all([basis.matrix] * len(axes))
    inverse = kron_all([basis.inverse_matrix] * len(axes))
    transformed = apply_unitary_to_axes(
        vector,
        inverse,
        axes=axes,
        num_qubits=num_qubits,
    )
    return transformed, basis_matrix


def _measurement_probabilities(
    transformed: np.ndarray,
    axes: tuple[int, ...],
    num_qubits: int,
) -> tuple[float, ...]:
    """Return normalized outcome probabilities from basis-coordinate amplitudes."""
    rest_axes = tuple(axis for axis in range(num_qubits) if axis not in axes)
    permutation = axes + rest_axes
    target_dim = 2 ** len(axes)
    tensor = transformed.reshape((2,) * num_qubits)
    moved = np.transpose(tensor, permutation).reshape(target_dim, -1)
    probabilities = tuple(float(np.sum(np.abs(row) ** 2)) for row in moved)
    return normalize_probs(probabilities)


def _collapse_and_return_to_computational(
    transformed: np.ndarray,
    basis_matrix: np.ndarray,
    *,
    outcome_index: int,
    probability: float,
    axes: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Collapse a ket in measurement coordinates and rotate it back."""
    rest_axes = tuple(axis for axis in range(num_qubits) if axis not in axes)
    permutation = axes + rest_axes
    inverse = np.argsort(permutation)
    target_dim = 2 ** len(axes)

    tensor = transformed.reshape((2,) * num_qubits)
    moved = np.transpose(tensor, permutation).reshape(target_dim, -1)
    collapsed = np.zeros_like(moved)
    collapsed[outcome_index] = moved[outcome_index] / np.sqrt(probability)
    restored = collapsed.reshape((2,) * num_qubits)
    measurement_coords = np.transpose(restored, inverse).reshape(-1)
    return apply_unitary_to_axes(
        measurement_coords,
        basis_matrix,
        axes=axes,
        num_qubits=num_qubits,
    )


def _expanded_projectors(
    basis: MeasurementBasis,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> tuple[np.ndarray, ...]:
    """Build full-space projectors for all outcomes in ``axes`` order."""
    key = (
        id(expand_operator),
        axes,
        num_qubits,
        tuple(_array_cache_key(vector) for vector in basis.vectors),
    )
    cached = _EXPANDED_PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    projectors = []
    for bits in product((0, 1), repeat=len(axes)):
        local_projectors = []
        for bit in bits:
            local_projectors.append(vector_projector(basis.vectors[bit]))
        local_projector = kron_all(local_projectors)
        projector = expand_operator(local_projector, axes=axes, num_qubits=num_qubits)
        projector.setflags(write=False)
        projectors.append(projector)

    cached = tuple(projectors)
    if len(_EXPANDED_PROJECTOR_CACHE) >= _EXPANDED_PROJECTOR_CACHE_MAX_SIZE:
        _EXPANDED_PROJECTOR_CACHE.clear()
    _EXPANDED_PROJECTOR_CACHE[key] = cached
    return cached


def _array_cache_key(array: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    """Return a stable content key for small dense basis vectors."""
    checked = np.asarray(array, dtype=np.complex128)
    return checked.shape, checked.dtype.str, checked.tobytes()


def _index_to_bits(index: int, width: int) -> tuple[int, ...]:
    """Convert an outcome index to big-endian bits of a fixed width."""
    return tuple((index >> shift) & 1 for shift in range(width - 1, -1, -1))


def _probability_table(
    probabilities: tuple[float, ...],
    basis: MeasurementBasis,
    width: int,
) -> ProbabilityTable:
    """Pair labeled outcome tuples with probabilities."""
    return tuple(
        (tuple(basis.labels[bit] for bit in bits), probabilities[index])
        for index, bits in enumerate(product((0, 1), repeat=width))
    )


def _check_axes(layout: StateLayout, axes: tuple[int, ...]) -> None:
    """Validate non-empty unique qubit axes against a layout."""
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if not axes:
        raise ValueError("axes must be non-empty")
    for axis in axes:
        if type(axis) is not int:
            raise TypeError("axes must be ints")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must be unique")
    for axis in axes:
        if axis < 0 or axis >= layout.size:
            raise ValueError("axes must be in range")
        if layout.dim_at(axis) != 2:
            raise DimensionError("ket/density operations support qubit axes only")


def _check_payload_layout(
    state: KetState | DensityState,
    layout: StateLayout,
    rep: str,
) -> None:
    """Translate payload-layout compatibility errors for measurement callers."""
    try:
        assert_payload_layout_compatible(state, layout)
    except InvalidLayoutError as exc:
        if "qubit axes" in str(exc):
            raise DimensionError(
                "ket/density operations support qubit axes only"
            ) from exc
        raise DimensionError(f"layout does not match {rep} payload") from exc


__all__ = ["measure_density", "measure_ket"]
