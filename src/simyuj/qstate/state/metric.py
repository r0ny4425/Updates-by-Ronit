from __future__ import annotations

"""State-overlap, entropy, and entanglement metrics for qstate payloads."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..errors import InvalidOperationError
from ..math import matrix as _matrix
from ..math.linalg import trace
from .bell_diag import BellDiagState, normalize_bell_label
from .density import DensityState
from .ket import KetState


def purity(state: object) -> float:
    """Return the purity of a supported payload.

    Parameters
    ----------
    state : object
        ``KetState``, ``DensityState``, or ``BellDiagState``.

    Returns
    -------
    float
        ``1`` for kets, :math:`\\operatorname{Tr}(\\rho^2)` for density
        matrices, or :math:`\\sum_i p_i^2` for Bell-diagonal probabilities.

    Raises
    ------
    InvalidOperationError
        If ``state`` has an unsupported representation.
    """
    if isinstance(state, KetState):
        return 1.0
    if isinstance(state, DensityState):
        return float(trace(state.rho @ state.rho).real)
    if isinstance(state, BellDiagState):
        return sum(probability * probability for probability in state.probs)
    raise InvalidOperationError("purity currently supports ket, density, and bell_diag")


def entropy(state: object, *, base: float = 2.0) -> float:
    """Return the von Neumann entropy of a supported payload.

    Parameters
    ----------
    state : object
        ``KetState``, ``DensityState``, or ``BellDiagState``.
    base : float, default=2.0
        Logarithm base. The default returns entropy in bits.

    Returns
    -------
    float
        ``0`` for kets, density-matrix von Neumann entropy, or the Shannon
        entropy of Bell probabilities for Bell-diagonal states.

    Raises
    ------
    TypeError
        If ``base`` is not a real scalar.
    ValueError
        If ``base`` is not positive or equals one.
    InvalidOperationError
        If ``state`` has an unsupported representation.
    """
    log_base = _check_entropy_base(base)
    if isinstance(state, KetState):
        return 0.0
    if isinstance(state, DensityState):
        eigenvalues = np.linalg.eigvalsh(state.rho)
        return _shannon_entropy(eigenvalues, log_base=log_base)
    if isinstance(state, BellDiagState):
        return _shannon_entropy(state.probs, log_base=log_base)
    raise InvalidOperationError(
        "entropy currently supports ket, density, and bell_diag"
    )


def bell_fidelity(state: object, label: object = "phi+") -> float:
    """Return fidelity with a target Bell state.

    Parameters
    ----------
    state : object
        ``BellDiagState`` or two-qubit ``KetState``/``DensityState``.
    label : object, default="phi+"
        Target Bell label.

    Returns
    -------
    float
        Overlap with the target Bell state.

    Raises
    ------
    InvalidOperationError
        If the payload representation is unsupported or a dense payload is not
        two-qubit.
    """
    normalized = normalize_bell_label(label)
    if isinstance(state, BellDiagState):
        return state.fidelity(normalized)
    if isinstance(state, KetState):
        if state.num_qubits != 2:
            raise InvalidOperationError("Bell fidelity requires exactly two qubits")

        from ..measure.bell import bell_vector

        return float(abs(np.vdot(bell_vector(normalized), state.vector)) ** 2)
    if isinstance(state, DensityState):
        if state.num_qubits != 2:
            raise InvalidOperationError("Bell fidelity requires exactly two qubits")

        from ..measure.bell import bell_projector

        return float(trace(bell_projector(normalized) @ state.rho).real)
    raise InvalidOperationError(
        "Bell fidelity currently supports ket, density, and bell_diag"
    )


def concurrence(state: object) -> float:
    """Return the two-qubit concurrence of a supported payload.

    Parameters
    ----------
    state : object
        Two-qubit ``KetState``, ``DensityState``, or ``BellDiagState``.

    Returns
    -------
    float
        Wootters concurrence. Bell-diagonal states use the closed form
        ``max(0, 2 * max(p_i) - 1)``.

    Raises
    ------
    InvalidOperationError
        If the payload representation is unsupported or does not represent
        exactly two qubits.
    """
    if isinstance(state, BellDiagState):
        return max(0.0, 2.0 * max(state.probs) - 1.0)

    density = _two_qubit_density_for_metric(state, metric_name="concurrence")
    spin_flip = np.kron(_matrix.Y, _matrix.Y)
    rho_tilde = spin_flip @ np.conjugate(density.rho) @ spin_flip
    eigenvalues = np.linalg.eigvals(density.rho @ rho_tilde)
    roots = sorted(
        (float(np.sqrt(max(0.0, value.real))) for value in eigenvalues),
        reverse=True,
    )
    return max(0.0, roots[0] - roots[1] - roots[2] - roots[3])


def negativity(state: object, *, subsystem: int = 1) -> float:
    """Return the two-qubit entanglement negativity.

    Parameters
    ----------
    state : object
        Two-qubit ``KetState``, ``DensityState``, or ``BellDiagState``.
    subsystem : int, default=1
        Subsystem to partially transpose for dense payloads. For two-qubit
        states this may be ``0`` or ``1``; Bell-diagonal states use the same
        closed form for either subsystem.

    Returns
    -------
    float
        :math:`(||\\rho^{T_B}||_1 - 1) / 2`.

    Raises
    ------
    TypeError
        If ``subsystem`` is not exactly an ``int``.
    ValueError
        If ``subsystem`` is not ``0`` or ``1``.
    InvalidOperationError
        If the payload representation is unsupported or does not represent
        exactly two qubits.
    """
    _check_bipartite_subsystem(subsystem)
    if isinstance(state, BellDiagState):
        return max(0.0, max(state.probs) - 0.5)

    trace_norm = _partial_transpose_trace_norm(
        state,
        subsystem=subsystem,
        metric_name="negativity",
    )
    return max(0.0, 0.5 * (trace_norm - 1.0))


def log_negativity(
    state: object,
    *,
    subsystem: int = 1,
    base: float = 2.0,
) -> float:
    """Return the logarithmic negativity of a two-qubit state.

    Parameters
    ----------
    state : object
        Two-qubit ``KetState``, ``DensityState``, or ``BellDiagState``.
    subsystem : int, default=1
        Subsystem to partially transpose for dense payloads.
    base : float, default=2.0
        Logarithm base. The default returns the value in bits.

    Returns
    -------
    float
        :math:`\\log(||\\rho^{T_B}||_1)` in the requested logarithm base.

    Raises
    ------
    TypeError
        If ``subsystem`` or ``base`` has the wrong type.
    ValueError
        If ``subsystem`` or ``base`` is outside the supported range.
    InvalidOperationError
        If the payload representation is unsupported or does not represent
        exactly two qubits.
    """
    _check_bipartite_subsystem(subsystem)
    log_base = _check_entropy_base(base)
    if isinstance(state, BellDiagState):
        trace_norm = 2.0 * negativity(state, subsystem=subsystem) + 1.0
    else:
        trace_norm = _partial_transpose_trace_norm(
            state,
            subsystem=subsystem,
            metric_name="log negativity",
        )
    return float(np.log(trace_norm) / log_base)


def max_chsh_value(state: object) -> float:
    """Return the maximal CHSH value for a two-qubit state.

    Parameters
    ----------
    state : object
        Two-qubit ``KetState``, ``DensityState``, or ``BellDiagState``.

    Returns
    -------
    float
        Maximal CHSH S value over projective measurement settings.

    Raises
    ------
    InvalidOperationError
        If the payload representation is unsupported or the payload does not
        represent exactly two qubits.

    Notes
    -----
    The implementation uses the Horodecki two-qubit criterion.  It builds the
    real correlation matrix
    :math:`T_{ij} = \\operatorname{Tr}(\\rho(\\sigma_i \\otimes \\sigma_j))`
    for :math:`\\sigma_i` in ``(X, Y, Z)``, then returns
    :math:`2\\sqrt{u_1 + u_2}` where :math:`u_1` and :math:`u_2` are the two
    largest eigenvalues of :math:`T^T T`.
    """
    density = _two_qubit_density_for_metric(
        state,
        metric_name="max CHSH value",
    )
    paulis = (_matrix.X, _matrix.Y, _matrix.Z)
    correlations = np.empty((3, 3), dtype=np.float64)
    for left_index, left_pauli in enumerate(paulis):
        for right_index, right_pauli in enumerate(paulis):
            observable = np.kron(left_pauli, right_pauli)
            correlations[left_index, right_index] = float(
                trace(density.rho @ observable).real
            )

    eigenvalues = np.linalg.eigvalsh(correlations.T @ correlations)
    top_two_sum = max(0.0, float(eigenvalues[-1] + eigenvalues[-2]))
    return float(2.0 * np.sqrt(top_two_sum))


def fidelity(left: object, right: object) -> float:
    """Return fidelity for supported payload pairs.

    Parameters
    ----------
    left, right : object
        Supported pairs are ket/ket, ket/density, density/ket, and
        BellDiagState/BellDiagState.

    Returns
    -------
    float
        Representation-specific fidelity value.

    Raises
    ------
    InvalidOperationError
        If the pair is unsupported.

    Notes
    -----
    Density/density fidelity is not implemented here.  Bell-diagonal fidelity
    uses the squared classical Bhattacharyya overlap of the stored probability
    vectors.
    """
    if isinstance(left, KetState) and isinstance(right, KetState):
        return float(abs(np.vdot(left.vector, right.vector)) ** 2)
    if isinstance(left, KetState) and isinstance(right, DensityState):
        vector = left.vector.reshape(-1, 1)
        return float((np.conjugate(vector.T) @ right.rho @ vector)[0, 0].real)
    if isinstance(left, DensityState) and isinstance(right, KetState):
        return fidelity(right, left)
    if isinstance(left, BellDiagState) and isinstance(right, BellDiagState):
        overlap = sum(
            np.sqrt(left_prob * right_prob)
            for left_prob, right_prob in zip(left.probs, right.probs)
        )
        return float(overlap * overlap)
    raise InvalidOperationError(
        "fidelity currently supports ket/ket, ket/density, and bell_diag/bell_diag"
    )


def _check_entropy_base(base: object) -> float:
    if not isinstance(base, (int, float)) or isinstance(base, bool):
        raise TypeError("base must be int or float")
    checked = float(base)
    if checked <= 0.0 or not np.isfinite(checked):
        raise ValueError("base must be positive and finite")
    if checked == 1.0:
        raise ValueError("base must not be one")
    return float(np.log(checked))


def _shannon_entropy(values: Iterable[Any], *, log_base: float) -> float:
    entropy_value = 0.0
    for raw_value in values:
        probability = max(0.0, float(np.real(raw_value)))
        if probability > 0.0:
            entropy_value -= probability * float(np.log(probability)) / log_base
    return float(entropy_value)


def _check_bipartite_subsystem(subsystem: object) -> int:
    if type(subsystem) is not int:
        raise TypeError("subsystem must be int")
    if subsystem not in {0, 1}:
        raise ValueError("subsystem must be 0 or 1")
    return subsystem


def _partial_transpose_trace_norm(
    state: object,
    *,
    subsystem: int,
    metric_name: str,
) -> float:
    density = _two_qubit_density_for_metric(state, metric_name=metric_name)
    partial = _partial_transpose_two_qubit(density.rho, subsystem=subsystem)
    eigenvalues = np.linalg.eigvalsh(partial)
    return float(np.sum(np.abs(eigenvalues)))


def _partial_transpose_two_qubit(rho: np.ndarray, *, subsystem: int) -> np.ndarray:
    tensor = np.asarray(rho, dtype=np.complex128).reshape(2, 2, 2, 2)
    if subsystem == 0:
        transposed = np.transpose(tensor, (2, 1, 0, 3))
    else:
        transposed = np.transpose(tensor, (0, 3, 2, 1))
    return transposed.reshape(4, 4)


def _two_qubit_density_for_metric(state: object, *, metric_name: str) -> DensityState:
    if isinstance(state, DensityState):
        density = state
    elif isinstance(state, KetState):
        from .convert import ket_to_density

        density = ket_to_density(state)
    elif isinstance(state, BellDiagState):
        from .convert import bell_diag_to_density

        density = bell_diag_to_density(state)
    else:
        raise InvalidOperationError(
            f"{metric_name} currently supports ket, density, and bell_diag"
        )

    if density.num_qubits != 2:
        raise InvalidOperationError(f"{metric_name} requires exactly two qubits")
    return density


__all__ = [
    "bell_fidelity",
    "concurrence",
    "entropy",
    "fidelity",
    "log_negativity",
    "max_chsh_value",
    "negativity",
    "purity",
]
