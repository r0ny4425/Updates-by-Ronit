from __future__ import annotations

"""Representation conversion helpers for qstate payloads.

Conversions are intentionally explicit and loss-aware.  Pure-state conversions
between ket and density are supported, Bell-diagonal states can convert to dense
density matrices, and conversions back to ket or Bell-diagonal form require the
state to be exactly representable within the package tolerance.
"""

import numpy as np

from ..errors import DimensionError, InvalidReprError, InvalidStateError
from ..math.const import ATOL
from ..math.linalg import trace
from ..math.projector import vector_projector
from .bell_diag import BELL_LABELS, BellDiagState
from .density import DensityState
from .ket import KetState


def ket_to_density(state: KetState) -> DensityState:
    """Convert a ket payload to :math:`\\rho = |\\psi\\rangle\\langle\\psi|`.

    Parameters
    ----------
    state : KetState
        Normalized ket payload.

    Returns
    -------
    DensityState
        Pure density-matrix payload.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState``.

    Examples
    --------
    >>> from simyuj.qstate.state import basis, ket_to_density
    >>> ket_to_density(basis("0")).rho.shape
    (2, 2)
    """
    if not isinstance(state, KetState):
        raise TypeError("state must be KetState")
    vector = np.asarray(state.vector, dtype=np.complex128)
    return DensityState._from_trusted(vector_projector(vector))


def density_to_ket_if_pure(state: DensityState) -> KetState:
    """Convert a pure density state to a ket.

    Parameters
    ----------
    state : DensityState
        Density-matrix payload.

    Returns
    -------
    KetState
        Principal eigenvector of ``state.rho`` with deterministic global phase.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``.
    InvalidStateError
        If the density matrix is not pure within ``ATOL``.

    Notes
    -----
    The first non-zero amplitude of the returned vector is made real and
    positive.  This fixes global phase for reproducible conversions.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")

    eigvals, eigvecs = np.linalg.eigh(state.rho)
    index = int(np.argmax(eigvals))

    if abs(float(eigvals[index]) - 1.0) > ATOL:
        raise InvalidStateError("density state is not pure")

    if np.sum(np.abs(eigvals) > ATOL) != 1:
        raise InvalidStateError("density state is not pure")

    vector = np.asarray(eigvecs[:, index], dtype=np.complex128)
    for amplitude in vector:
        if abs(amplitude) > ATOL:
            vector = vector / (amplitude / abs(amplitude))
            break
    return KetState._from_trusted(vector)


def bell_diag_to_density(state: BellDiagState) -> DensityState:
    """Expand a Bell-diagonal probability record to a density matrix.

    Parameters
    ----------
    state : BellDiagState
        Bell-diagonal payload in ``BELL_LABELS`` order.

    Returns
    -------
    DensityState
        Two-qubit density matrix
        :math:`\\sum_i p_i |B_i\\rangle\\langle B_i|`.

    Raises
    ------
    TypeError
        If ``state`` is not a ``BellDiagState``.

    Examples
    --------
    >>> from simyuj.qstate.state import BellDiagState, bell_diag_to_density
    >>> state = BellDiagState.from_label("psi+")
    >>> bell_diag_to_density(state).rho.shape
    (4, 4)
    """
    if not isinstance(state, BellDiagState):
        raise TypeError("state must be BellDiagState")

    from ..measure.bell import bell_density_matrix

    rho = np.zeros((4, 4), dtype=np.complex128)
    for label, probability in zip(BELL_LABELS, state.probs):
        rho = rho + probability * bell_density_matrix(label)
    return DensityState._from_trusted(rho)


def bell_diag_to_ket_if_pure(state: BellDiagState) -> KetState:
    """Convert a pure Bell-diagonal state to a Bell ket.

    Parameters
    ----------
    state : BellDiagState
        Bell-diagonal payload.

    Returns
    -------
    KetState
        Bell-state ket for the single probability equal to one within ``ATOL``.

    Raises
    ------
    TypeError
        If ``state`` is not a ``BellDiagState``.
    InvalidStateError
        If exactly one Bell probability is not near one.
    """
    if not isinstance(state, BellDiagState):
        raise TypeError("state must be BellDiagState")

    pure_indices = [
        index
        for index, probability in enumerate(state.probs)
        if abs(probability - 1.0) <= ATOL
    ]
    if len(pure_indices) != 1:
        raise InvalidStateError("Bell-diagonal state is not pure")

    from ..measure.bell import bell_vector

    return KetState._from_trusted(bell_vector(BELL_LABELS[pure_indices[0]]))


def density_to_bell_diag_if_exact(state: DensityState) -> BellDiagState:
    """Convert an exactly Bell-diagonal density matrix to compact form.

    Parameters
    ----------
    state : DensityState
        Two-qubit density-matrix payload.

    Returns
    -------
    BellDiagState
        Bell probabilities obtained by projection onto the Bell basis.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``.
    DimensionError
        If ``state`` does not represent exactly two qubits.
    InvalidStateError
        If the Bell-projector reconstruction is not close to ``state.rho``.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    if state.num_qubits != 2:
        raise DimensionError("Bell-diagonal conversion requires exactly two qubits")

    from ..measure.bell import bell_projectors

    projectors = bell_projectors()
    probabilities = tuple(
        float(trace(projector @ state.rho).real) for projector in projectors
    )
    reconstructed = sum(
        probability * projector
        for probability, projector in zip(probabilities, projectors)
    )

    if not np.allclose(reconstructed, state.rho, atol=ATOL, rtol=ATOL):
        raise InvalidStateError("density state is not exactly Bell diagonal")
    return BellDiagState(probabilities)


def ket_to_bell_diag_if_exact(state: KetState) -> BellDiagState:
    """Convert a ket to compact Bell-diagonal form when exactly representable.

    Parameters
    ----------
    state : KetState
        Ket payload.

    Returns
    -------
    BellDiagState
        Bell-diagonal state obtained through density conversion.

    Raises
    ------
    TypeError
        If ``state`` is not a ``KetState``.
    DimensionError
        If ``state`` is not two-qubit.
    InvalidStateError
        If the state is not exactly Bell diagonal as a density matrix.
    """
    return density_to_bell_diag_if_exact(ket_to_density(state))


def as_rep(payload: object, to_rep: str) -> object:
    """Convert a payload to a requested representation.

    Parameters
    ----------
    payload : object
        ``KetState``, ``DensityState``, or ``BellDiagState`` payload.
    to_rep : str
        Target representation: ``"ket"``, ``"density"``, or ``"bell_diag"``.

    Returns
    -------
    object
        Payload in the requested representation, or the original object when it
        already has that representation.

    Raises
    ------
    InvalidReprError
        If the requested conversion is unsupported.
    InvalidStateError
        If conversion would require a mixed density state or non-exact
        Bell-diagonal representation.

    Examples
    --------
    >>> from simyuj.qstate.state import as_rep, basis
    >>> as_rep(basis("0"), "density").num_qubits
    1
    """
    if to_rep == "ket":
        if isinstance(payload, KetState):
            return payload
        if isinstance(payload, DensityState):
            return density_to_ket_if_pure(payload)
        if isinstance(payload, BellDiagState):
            return bell_diag_to_ket_if_pure(payload)

    if to_rep == "density":
        if isinstance(payload, DensityState):
            return payload
        if isinstance(payload, KetState):
            return ket_to_density(payload)
        if isinstance(payload, BellDiagState):
            return bell_diag_to_density(payload)

    if to_rep == "bell_diag":
        if isinstance(payload, BellDiagState):
            return payload
        if isinstance(payload, DensityState):
            return density_to_bell_diag_if_exact(payload)
        if isinstance(payload, KetState):
            return ket_to_bell_diag_if_exact(payload)

    raise InvalidReprError(f"unsupported representation conversion: {to_rep!r}")


__all__ = [
    "as_rep",
    "bell_diag_to_density",
    "bell_diag_to_ket_if_pure",
    "density_to_bell_diag_if_exact",
    "density_to_ket_if_pure",
    "ket_to_bell_diag_if_exact",
    "ket_to_density",
]
