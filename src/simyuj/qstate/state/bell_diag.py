from __future__ import annotations

"""Compact two-qubit Bell-diagonal state representation.

Bell-diagonal payloads store four probabilities in ``BELL_LABELS`` order:
``("phi+", "phi-", "psi+", "psi-")``.  They are useful for Bell-pair workflows
that can stay in a probability representation until an operation requires a
dense ket or density matrix.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from simyuj.primitives.validation import validate_bool

from ..check import check_prob_vector, check_probability
from ..errors import InvalidOperationError, MeasurementError, NoiseError
from ..math import matrix as mat
from ..math.const import PROB_ATOL
from ..measure.basis import MeasurementBasis
from ..measure.result import BellResult, MeasurementResult
from ..measure.sample import sample_probs
from ..space.layout import StateLayout

BellLabel: TypeAlias = str
BELL_LABELS: tuple[BellLabel, ...] = ("phi+", "phi-", "psi+", "psi-")

_LABEL_ALIASES: dict[str, BellLabel] = {
    "phi+": "phi+",
    "phi_plus": "phi+",
    "phi-plus": "phi+",
    "phiplus": "phi+",
    "φ+": "phi+",
    "phi-": "phi-",
    "phi_minus": "phi-",
    "phi-minus": "phi-",
    "phiminus": "phi-",
    "φ-": "phi-",
    "psi+": "psi+",
    "psi_plus": "psi+",
    "psi-plus": "psi+",
    "psiplus": "psi+",
    "ψ+": "psi+",
    "psi-": "psi-",
    "psi_minus": "psi-",
    "psi-minus": "psi-",
    "psiminus": "psi-",
    "ψ-": "psi-",
}

_LABEL_BITS = {
    "phi+": (0, 0),
    "phi-": (0, 1),
    "psi+": (1, 0),
    "psi-": (1, 1),
}

_BITS_LABEL = {bits: label for label, bits in _LABEL_BITS.items()}

_PAULI_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}

_PAULI_MATRICES = {
    "I": mat.I2,
    "X": mat.X,
    "Y": mat.Y,
    "Z": mat.Z,
}

_TWO_QUBIT_PAULI_MATRICES = {
    left + right: np.kron(_PAULI_MATRICES[left], _PAULI_MATRICES[right])
    for left in _PAULI_MATRICES
    for right in _PAULI_MATRICES
}


def normalize_bell_label(label: object) -> BellLabel:
    """Normalize a Bell-state label or alias.

    Parameters
    ----------
    label : object
        Bell label string.  Supported labels include ``phi+``, ``phi-``,
        ``psi+``, ``psi-`` and underscore, hyphen, compact, and Greek-letter
        aliases.

    Returns
    -------
    str
        Normalized Bell label in ``BELL_LABELS``.

    Raises
    ------
    TypeError
        If ``label`` is not a string.
    ValueError
        If the stripped label is empty or unsupported.
    """
    if not isinstance(label, str):
        raise TypeError("Bell label must be str")
    normalized = label.strip().lower()
    if not normalized:
        raise ValueError("Bell label must be non-empty")
    try:
        return _LABEL_ALIASES[normalized]
    except KeyError:
        raise ValueError(f"unsupported Bell label: {label!r}") from None


def bell_index(label: object) -> int:
    """Return the index of a Bell label in ``BELL_LABELS``.

    Parameters
    ----------
    label : object
        Label accepted by :func:`normalize_bell_label`.

    Returns
    -------
    int
        Index in ``("phi+", "phi-", "psi+", "psi-")`` order.
    """
    return BELL_LABELS.index(normalize_bell_label(label))


def label_to_bits(label: object) -> tuple[int, int]:
    """Return the two-bit outcome convention for a Bell label.

    Parameters
    ----------
    label : object
        Label accepted by :func:`normalize_bell_label`.

    Returns
    -------
    tuple of int
        Outcome bits: ``phi+ -> (0, 0)``, ``phi- -> (0, 1)``,
        ``psi+ -> (1, 0)``, and ``psi- -> (1, 1)``.
    """
    return _LABEL_BITS[normalize_bell_label(label)]


def bits_to_label(bits: object) -> BellLabel:
    """Return the Bell label associated with two outcome bits.

    Parameters
    ----------
    bits : object
        Tuple of exactly two integer bits.  Boolean values are rejected because
        the implementation requires entries whose type is exactly ``int``.

    Returns
    -------
    str
        Bell label in ``BELL_LABELS``.

    Raises
    ------
    TypeError
        If ``bits`` is not a tuple.
    ValueError
        If ``bits`` does not contain exactly two entries or any entry is not
        ``0`` or ``1``.
    """
    if not isinstance(bits, tuple):
        raise TypeError("bits must be tuple")
    if len(bits) != 2:
        raise ValueError("bits must contain exactly two entries")
    for bit in bits:
        if type(bit) is not int or bit not in {0, 1}:
            raise ValueError("bits entries must be 0 or 1")
    return _BITS_LABEL[bits]


def _bell_label_after_pauli(label: BellLabel, pauli: str) -> BellLabel:
    bell_x, bell_z = _LABEL_BITS[label]
    pauli_x, pauli_z = _PAULI_BITS[pauli]
    return _BITS_LABEL[(bell_x ^ pauli_x, bell_z ^ pauli_z)]


def _combined_pauli_label(labels: str) -> str:
    x_bit = 0
    z_bit = 0
    for label in labels:
        pauli_x, pauli_z = _PAULI_BITS[label]
        x_bit ^= pauli_x
        z_bit ^= pauli_z
    for pauli, bits in _PAULI_BITS.items():
        if bits == (x_bit, z_bit):
            return pauli
    raise RuntimeError("unreachable Pauli bit combination")


def _apply_pauli_permutation(state: BellDiagState, pauli: str) -> BellDiagState:
    probs = [0.0, 0.0, 0.0, 0.0]
    for label, probability in zip(BELL_LABELS, state.probs):
        mapped = _bell_label_after_pauli(label, pauli)
        probs[bell_index(mapped)] += probability
    return BellDiagState(tuple(probs))


def _apply_pauli_mixture(
    state: BellDiagState,
    probabilities: Mapping[str, float],
) -> BellDiagState:
    probs = [0.0, 0.0, 0.0, 0.0]
    for pauli, branch_probability in probabilities.items():
        branch = _apply_pauli_permutation(state, pauli)
        for index, probability in enumerate(branch.probs):
            probs[index] += branch_probability * probability
    return BellDiagState(tuple(probs))


def _pauli_label_from_unitary(operation: object) -> str | None:
    from ..ops.unitary import Unitary

    if not isinstance(operation, Unitary):
        return None
    if operation.arity == 1:
        return _matrix_pauli_label(operation.matrix, _PAULI_MATRICES)
    if operation.arity == 2:
        label = _matrix_pauli_label(operation.matrix, _TWO_QUBIT_PAULI_MATRICES)
        if label is not None:
            return _combined_pauli_label(label)
    return None


def _pauli_mixture_from_kraus(channel: object) -> dict[str, float] | None:
    from ..noise.base import KrausChannel

    if not isinstance(channel, KrausChannel):
        return None
    basis = _PAULI_MATRICES if channel.arity == 1 else _TWO_QUBIT_PAULI_MATRICES
    probabilities = {pauli: 0.0 for pauli in _PAULI_MATRICES}
    for op in channel.ops:
        parsed = _scaled_pauli_label(op, basis)
        if parsed is None:
            return None
        raw_label, probability = parsed
        label = raw_label if channel.arity == 1 else _combined_pauli_label(raw_label)
        probabilities[label] += probability
    return probabilities


def _matrix_pauli_label(
    matrix: np.ndarray,
    basis: Mapping[str, np.ndarray],
) -> str | None:
    for label, pauli in basis.items():
        if _allclose_up_to_global_phase(matrix, pauli):
            return label
    return None


def _scaled_pauli_label(
    matrix: np.ndarray,
    basis: Mapping[str, np.ndarray],
) -> tuple[str, float] | None:
    if np.allclose(matrix, 0.0):
        return "I", 0.0

    for label, pauli in basis.items():
        scale = _global_scale(matrix, pauli)
        if scale is None:
            continue
        if np.allclose(matrix, scale * pauli):
            return label, float(abs(scale) ** 2)
    return None


def _allclose_up_to_global_phase(matrix: np.ndarray, target: np.ndarray) -> bool:
    scale = _global_scale(matrix, target)
    return scale is not None and np.allclose(matrix, scale * target)


def _global_scale(matrix: np.ndarray, target: np.ndarray) -> complex | None:
    candidate = np.asarray(matrix, dtype=np.complex128)
    reference = np.asarray(target, dtype=np.complex128)
    if candidate.shape != reference.shape:
        return None

    nonzero = np.argwhere(np.abs(reference) > 0.0)
    if nonzero.size == 0:
        return None
    index = tuple(int(part) for part in nonzero[0])
    scale = candidate[index] / reference[index]
    return complex(scale)


def _check_two_qubit_bell_layout(layout: StateLayout) -> None:
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")
    if layout.size != 2:
        raise InvalidOperationError("bell_diag operations require a two-axis layout")
    if layout.dims != (2, 2):
        raise InvalidOperationError("bell_diag operations require two qubit axes")


def _check_operation_axes(
    axes: tuple[int, ...],
    *,
    arity: int,
) -> None:
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != arity:
        raise InvalidOperationError("axes count must match operation arity")
    for axis in axes:
        if type(axis) is not int:
            raise TypeError("axes entries must be int")
        if axis not in {0, 1}:
            raise InvalidOperationError("bell_diag axes must refer to stored qubits")
    if len(set(axes)) != len(axes):
        raise InvalidOperationError("axes must be unique")


@dataclass(frozen=True, slots=True, init=False)
class BellDiagState:
    """Compact Bell-diagonal two-qubit probability payload.

    Parameters
    ----------
    probs : object
        Four probabilities in ``BELL_LABELS`` order.  The values must form a
        unit-sum probability vector under ``check_prob_vector``.

    Attributes
    ----------
    probs : tuple of float
        Probabilities for ``phi+``, ``phi-``, ``psi+``, and ``psi-``.

    Raises
    ------
    TypeError
        If ``probs`` is not a supported probability sequence.
    ValueError
        If probabilities are invalid or the sequence length is not four.
    """

    probs: tuple[float, float, float, float]

    def __init__(self, probs: object) -> None:
        """Validate and store Bell-state probabilities."""
        checked = check_prob_vector(probs, name="probs")
        if len(checked) != 4:
            raise ValueError("Bell-diagonal probabilities must have length four")
        object.__setattr__(self, "probs", checked)

    @property
    def probabilities(self) -> tuple[tuple[BellLabel, float], ...]:
        """Return labeled probabilities in ``BELL_LABELS`` order."""
        return tuple(zip(BELL_LABELS, self.probs))

    @property
    def num_qubits(self) -> int:
        """Number of qubits represented by the Bell-diagonal payload."""
        return 2

    def probability(self, label: object) -> float:
        """Return the probability assigned to a Bell label.

        Parameters
        ----------
        label : object
            Label accepted by :func:`normalize_bell_label`.

        Returns
        -------
        float
            Stored probability for ``label``.
        """
        return self.probs[bell_index(label)]

    def fidelity(self, label: object = "phi+") -> float:
        """Return Bell-state fidelity with a target label.

        Parameters
        ----------
        label : object, default="phi+"
            Target Bell label.

        Returns
        -------
        float
            Probability assigned to the target label.
        """
        return self.probability(label)

    @classmethod
    def from_label(cls, label: object) -> BellDiagState:
        """Construct a pure Bell-diagonal state for one Bell label.

        Parameters
        ----------
        label : object
            Label accepted by :func:`normalize_bell_label`.

        Returns
        -------
        BellDiagState
            Probability vector with mass one on ``label``.
        """
        index = bell_index(label)
        probs = [0.0, 0.0, 0.0, 0.0]
        probs[index] = 1.0
        return cls(tuple(probs))


def make_bell_diag(state: object = "phi+") -> BellDiagState:
    """Coerce a value into a Bell-diagonal state.

    Parameters
    ----------
    state : object, default="phi+"
        Existing ``BellDiagState``, Bell label string, mapping from labels to
        probabilities, or probability sequence in ``BELL_LABELS`` order.

    Returns
    -------
    BellDiagState
        Bell-diagonal payload.

    Raises
    ------
    TypeError
        If ``state`` is not a supported input type.
    ValueError
        If labels or probabilities are invalid, or if a mapping provides the
        same normalized label more than once.
    """
    if isinstance(state, BellDiagState):
        return state
    if isinstance(state, str):
        return BellDiagState.from_label(state)
    if isinstance(state, Mapping):
        probs = [0.0, 0.0, 0.0, 0.0]
        seen: set[BellLabel] = set()
        for raw_label, probability in state.items():
            label = normalize_bell_label(raw_label)
            if label in seen:
                raise ValueError(f"duplicate Bell label: {label}")
            seen.add(label)
            probs[bell_index(label)] = probability
        return BellDiagState(tuple(probs))
    if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
        return BellDiagState(state)
    raise TypeError("state must be BellDiagState, str, mapping, or sequence")


def werner(fidelity: object, label: object = "phi+") -> BellDiagState:
    """Construct a Werner-like Bell-diagonal state by target fidelity.

    Parameters
    ----------
    fidelity : object
        Target Bell-state probability assigned to ``label``.  Must be a scalar
        probability accepted by ``check_probability``.
    label : object, default="phi+"
        Target Bell label.

    Returns
    -------
    BellDiagState
        Bell-diagonal probability vector with the target probability set to
        ``fidelity`` and the remaining mass spread uniformly across the other
        three Bell labels.

    Notes
    -----
    For the white-noise convention
    :math:`\\rho = p|B\\rangle\\langle B| + (1 - p)I / 4`, this helper uses
    :math:`F = (1 + 3p) / 4`.  It accepts any probability ``F`` in
    ``[0, 1]`` and does not restrict to the physical Werner-mixture range
    :math:`F \\ge 0.25`.
    """

    target = normalize_bell_label(label)
    f = check_probability(fidelity, name="fidelity")
    other = (1.0 - f) / 3.0

    probs = [other] * len(BELL_LABELS)
    probs[bell_index(target)] = f
    return BellDiagState(tuple(probs))


class BellDiagHandler:
    """Representation handler for compact two-qubit Bell-diagonal states.

    The compact representation supports construction, Bell-basis measurement,
    Pauli operations, and Pauli noise. Tensor products, non-Pauli unitary
    operations, projective measurement, and non-Pauli noise require conversion
    to a dense representation first.
    """

    rep = "bell_diag"

    def make(self, state: object) -> object:
        """Coerce ``state`` with :func:`make_bell_diag`."""
        return make_bell_diag(state)

    def tensor(self, left: object, right: object) -> object:
        """Reject tensoring in the compact Bell-diagonal representation."""
        raise InvalidOperationError("Bell-diagonal tensoring is not implemented")

    def apply(
        self,
        payload: object,
        operation: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply Bell-diagonal-preserving Pauli operations."""
        from ..ops.unitary import Unitary

        if not isinstance(payload, BellDiagState):
            raise TypeError("payload must be BellDiagState")
        if not isinstance(operation, Unitary):
            raise TypeError("operation must be Unitary")
        _check_two_qubit_bell_layout(layout)
        _check_operation_axes(axes, arity=operation.arity)

        pauli = _pauli_label_from_unitary(operation)
        if pauli is None:
            raise InvalidOperationError(
                "unitary application on bell_diag supports Pauli operations only; "
                "convert to density for general unitaries"
            )
        return _apply_pauli_permutation(payload, pauli)

    def channel(
        self,
        payload: object,
        channel: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply Bell-diagonal-preserving Pauli noise channels."""
        from ..noise.base import KrausChannel

        if not isinstance(payload, BellDiagState):
            raise TypeError("payload must be BellDiagState")
        if not isinstance(channel, KrausChannel):
            raise TypeError("channel must be KrausChannel")
        _check_two_qubit_bell_layout(layout)
        _check_operation_axes(axes, arity=channel.arity)

        probabilities = _pauli_mixture_from_kraus(channel)
        if probabilities is None:
            raise NoiseError(
                "noise application on bell_diag supports Pauli channels only; "
                "convert to density for general channels"
            )
        return _apply_pauli_mixture(payload, probabilities)

    def measure(
        self,
        payload: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
        basis: str | MeasurementBasis = "z",
        rng: Any | None = None,
        collapse: bool = True,
    ) -> MeasurementResult:
        """Reject projective measurement without density conversion."""
        raise MeasurementError(
            "projective measurement on bell_diag requires density conversion"
        )

    def measure_bell(
        self,
        payload: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
        rng: Any | None = None,
        collapse: bool = True,
    ) -> BellResult:
        """Sample or collapse a compact Bell-diagonal state in the Bell basis.

        Parameters
        ----------
        payload : object
            Bell-diagonal payload to measure.
        layout : StateLayout
            Two-qubit layout with dimensions ``(2, 2)``.
        axes : tuple of int
            Axes to measure.  The set must be ``{0, 1}``; either order is
            accepted.
        rng : object, optional
            Random source passed to ``sample_probs`` for probabilistic outcomes.
        collapse : bool, default=True
            Whether to return a pure Bell-diagonal post-state.

        Returns
        -------
        BellResult
            Bell-basis measurement result.
        """
        if not isinstance(payload, BellDiagState):
            raise TypeError("payload must be BellDiagState")
        if not isinstance(layout, StateLayout):
            raise TypeError("layout must be StateLayout")
        if layout.size != 2:
            raise MeasurementError("Bell measurement requires a two-axis layout")
        if layout.dims != (2, 2):
            raise MeasurementError("Bell measurement requires two qubit axes")
        if not isinstance(axes, tuple):
            raise TypeError("axes must be tuple")
        for axis in axes:
            if type(axis) is not int:
                raise TypeError("axes entries must be int")
        if set(axes) != {0, 1}:
            raise MeasurementError("Bell measurement must cover both stored qubit axes")
        for axis in axes:
            layout.dim_at(axis)
        validate_bool(collapse, field_name="collapse")

        outcome_index = sample_probs(payload.probs, rng=rng)
        label = BELL_LABELS[outcome_index]
        post_state = BellDiagState.from_label(label) if collapse else None

        return BellResult(
            label=label,
            outcome=label_to_bits(label),
            probability=payload.probs[outcome_index],
            probabilities=payload.probabilities,
            post_state=post_state,
            collapsed=collapse,
        )


def is_pure_bell_diag(state: BellDiagState) -> bool:
    """Return whether a Bell-diagonal state has one probability near one.

    Parameters
    ----------
    state : BellDiagState
        Bell-diagonal state to inspect.

    Returns
    -------
    bool
        ``True`` if any stored probability is within ``PROB_ATOL`` of one.
    """
    return any(abs(prob - 1.0) <= PROB_ATOL for prob in state.probs)


__all__ = [
    "BELL_LABELS",
    "BellDiagHandler",
    "BellDiagState",
    "BellLabel",
    "bell_index",
    "bits_to_label",
    "is_pure_bell_diag",
    "label_to_bits",
    "make_bell_diag",
    "normalize_bell_label",
    "werner",
]
