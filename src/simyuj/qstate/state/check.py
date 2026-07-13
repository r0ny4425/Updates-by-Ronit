from __future__ import annotations

"""Validation and introspection helpers for state payload records.

These routines check representation-specific invariants for dense ket,
density-matrix, and Bell-diagonal payloads.  They are deliberately separate from
constructors so tests and manager code can re-check externally supplied or
manually restored payload objects.
"""

import math

import numpy as np

from ..errors import DimensionError, InvalidLayoutError, InvalidStateError
from ..math.const import ATOL, RTOL
from ..math.linalg import is_hermitian, is_psd, trace
from ..space.layout import StateLayout


def is_ket(payload: object) -> bool:
    """Return whether ``payload`` is a ``KetState`` instance."""
    from .ket import KetState

    return isinstance(payload, KetState)


def is_density(payload: object) -> bool:
    """Return whether ``payload`` is a ``DensityState`` instance."""
    from .density import DensityState

    return isinstance(payload, DensityState)


def is_bell_diag(payload: object) -> bool:
    """Return whether ``payload`` is a ``BellDiagState`` instance."""
    try:
        from .bell_diag import BellDiagState
    except ImportError:
        return False

    return isinstance(payload, BellDiagState)


def check_ket(payload: object) -> object:
    """Validate a dense ket payload.

    Parameters
    ----------
    payload : object
        Candidate ``KetState``.  The state vector must be one-dimensional,
        finite, normalized, and have power-of-two length.

    Returns
    -------
    object
        The original ``payload`` when all checks pass.

    Raises
    ------
    TypeError
        If ``payload`` is not a ``KetState``.
    InvalidStateError
        If the vector has invalid rank, non-finite entries, or is not
        normalized.
    DimensionError
        If the vector is empty, not power-of-two length, or ``num_qubits`` does
        not match the vector length.
    """
    from .ket import KetState

    if not isinstance(payload, KetState):
        raise TypeError("payload must be KetState")

    vector = np.asarray(payload.vector, dtype=np.complex128)
    if vector.ndim != 1:
        raise InvalidStateError("ket vector must be one-dimensional")
    if vector.size <= 0:
        raise DimensionError("ket vector must be non-empty")
    if not _is_power_of_two(int(vector.size)):
        raise DimensionError("ket vector length must be a power of two")
    if not np.all(np.isfinite(vector)):
        raise InvalidStateError("ket vector entries must be finite")

    norm = float(np.linalg.norm(vector))
    if not np.isclose(norm, 1.0, atol=ATOL, rtol=RTOL):
        raise InvalidStateError("ket vector must be normalized")

    expected_qubits = int(math.log2(vector.size))
    if getattr(payload, "num_qubits", expected_qubits) != expected_qubits:
        raise DimensionError("ket num_qubits does not match vector length")
    return payload


def check_density(payload: object) -> object:
    """Validate a dense density-matrix payload.

    Parameters
    ----------
    payload : object
        Candidate ``DensityState``.  The matrix must be square, non-empty,
        finite, Hermitian, trace-one, positive semidefinite, and have
        power-of-two dimension.

    Returns
    -------
    object
        The original ``payload`` when all checks pass.

    Raises
    ------
    TypeError
        If ``payload`` is not a ``DensityState``.
    InvalidStateError
        If matrix rank, entries, Hermiticity, trace, or positivity are invalid.
    DimensionError
        If shape, dimension, or ``num_qubits`` metadata is inconsistent.
    """
    from .density import DensityState

    if not isinstance(payload, DensityState):
        raise TypeError("payload must be DensityState")

    rho = np.asarray(payload.rho, dtype=np.complex128)
    if rho.ndim != 2:
        raise InvalidStateError("density matrix must be two-dimensional")
    if rho.shape[0] != rho.shape[1]:
        raise DimensionError("density matrix must be square")
    if rho.shape[0] <= 0:
        raise DimensionError("density matrix must be non-empty")
    if not _is_power_of_two(int(rho.shape[0])):
        raise DimensionError("density matrix size must be a power of two")
    if not np.all(np.isfinite(rho)):
        raise InvalidStateError("density matrix entries must be finite")
    if not is_hermitian(rho):
        raise InvalidStateError("density matrix must be Hermitian")

    tr = trace(rho)
    if abs(tr.imag) > ATOL:
        raise InvalidStateError("density matrix trace must be real")
    if not np.isclose(float(tr.real), 1.0, atol=ATOL, rtol=RTOL):
        raise InvalidStateError("density matrix trace must be one")
    if not is_psd(rho):
        raise InvalidStateError("density matrix must be positive semidefinite")

    expected_qubits = int(math.log2(rho.shape[0]))
    if getattr(payload, "num_qubits", expected_qubits) != expected_qubits:
        raise DimensionError("density num_qubits does not match matrix shape")
    return payload


def check_bell_diag(payload: object) -> object:
    """Validate a compact Bell-diagonal payload.

    Parameters
    ----------
    payload : object
        Candidate ``BellDiagState`` with four finite probabilities.

    Returns
    -------
    object
        The original ``payload`` when all checks pass.

    Raises
    ------
    TypeError
        If ``payload`` is not a ``BellDiagState``.
    InvalidStateError
        If the probability tuple length, range, finiteness, or sum is invalid.
    DimensionError
        If ``num_qubits`` metadata does not report two qubits.
    """
    from .bell_diag import BellDiagState

    if not isinstance(payload, BellDiagState):
        raise TypeError("payload must be BellDiagState")

    probs = tuple(float(value) for value in payload.probs)
    if len(probs) != 4:
        raise InvalidStateError("Bell-diagonal state must have four probabilities")
    for probability in probs:
        if not np.isfinite(probability):
            raise InvalidStateError("Bell-diagonal probabilities must be finite")
        if probability < -ATOL or probability > 1.0 + ATOL:
            raise InvalidStateError("Bell-diagonal probabilities must be in [0, 1]")
    if not np.isclose(sum(probs), 1.0, atol=ATOL, rtol=RTOL):
        raise InvalidStateError("Bell-diagonal probabilities must sum to one")
    if getattr(payload, "num_qubits", 2) != 2:
        raise DimensionError("Bell-diagonal state must represent two qubits")
    return payload


def check_payload(payload: object, *, rep: str | None = None) -> object:
    """Validate a payload, optionally forcing a representation.

    Parameters
    ----------
    payload : object
        Candidate state payload.
    rep : str, optional
        Required representation name.  Supported values are ``"ket"``,
        ``"density"``, and ``"bell_diag"``.

    Returns
    -------
    object
        The original ``payload`` when validation succeeds.

    Raises
    ------
    TypeError
        If ``payload`` has the wrong type for the requested representation.
    InvalidStateError
        If ``rep`` is unsupported or ``payload`` is not a supported payload
        type.
    DimensionError
        If representation-specific dimensions are invalid.
    """
    if rep == "ket":
        return check_ket(payload)
    if rep == "density":
        return check_density(payload)
    if rep == "bell_diag":
        return check_bell_diag(payload)
    if rep is not None:
        raise InvalidStateError(f"unsupported payload representation: {rep!r}")

    if is_ket(payload):
        return check_ket(payload)
    if is_density(payload):
        return check_density(payload)
    if is_bell_diag(payload):
        return check_bell_diag(payload)
    raise InvalidStateError(f"unsupported payload type: {type(payload).__name__}")


def payload_num_qubits(payload: object) -> int:
    """Return the validated qubit count for a payload.

    Parameters
    ----------
    payload : object
        State payload accepted by :func:`check_payload`.

    Returns
    -------
    int
        Non-negative number of qubits reported by ``payload.num_qubits``.

    Raises
    ------
    InvalidStateError
        If ``payload`` is not a supported payload.
    DimensionError
        If the payload does not expose an integer non-negative ``num_qubits``.
    """
    check_payload(payload)

    value = getattr(payload, "num_qubits", None)
    if type(value) is not int:
        raise DimensionError("payload must expose integer num_qubits")
    if value < 0:
        raise DimensionError("payload num_qubits must be non-negative")
    return value


def _payload_num_qubits_trusted(payload: object) -> int:
    """Return qubit count for a package-created payload without array checks."""
    value = getattr(payload, "num_qubits", None)
    if type(value) is not int:
        raise DimensionError("payload must expose integer num_qubits")
    if value < 0:
        raise DimensionError("payload num_qubits must be non-negative")
    return value


def payload_hilbert_dim(payload: object) -> int:
    """Return ``2 ** payload_num_qubits(payload)``.

    Parameters
    ----------
    payload : object
        State payload accepted by :func:`payload_num_qubits`.

    Returns
    -------
    int
        Hilbert-space dimension for the qubit payload.
    """
    return 2 ** payload_num_qubits(payload)


def assert_payload_layout_compatible(
    payload: object,
    layout: StateLayout,
) -> None:
    """Raise if a payload and layout disagree.

    Parameters
    ----------
    payload : object
        State payload accepted by :func:`payload_num_qubits`.
    layout : StateLayout
        Layout that should describe the payload axes.

    Raises
    ------
    TypeError
        If ``layout`` is not a ``StateLayout``.
    InvalidLayoutError
        If layout size, Hilbert-space dimension, or per-axis dimensions do not
        match the qubit payload.

    Notes
    -----
    The current qstate payload implementations are qubit-only.  Every layout
    axis must therefore have local dimension ``2``.
    """
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")

    num_qubits = payload_num_qubits(payload)
    if layout.size != num_qubits:
        raise InvalidLayoutError(
            f"layout size {layout.size} does not match payload qubits {num_qubits}"
        )
    if layout.hilbert_dim != 2**num_qubits:
        raise InvalidLayoutError(
            f"layout Hilbert dimension {layout.hilbert_dim} does not match payload"
        )
    for axis in range(layout.size):
        if layout.dim_at(axis) != 2:
            raise InvalidLayoutError("qstate currently supports qubit axes only")


def _assert_payload_layout_compatible_trusted(
    payload: object,
    layout: StateLayout,
) -> None:
    """Check layout compatibility without revalidating payload arrays."""
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")

    num_qubits = _payload_num_qubits_trusted(payload)
    if layout.size != num_qubits:
        raise InvalidLayoutError(
            f"layout size {layout.size} does not match payload qubits {num_qubits}"
        )
    if layout.hilbert_dim != 2**num_qubits:
        raise InvalidLayoutError(
            f"layout Hilbert dimension {layout.hilbert_dim} does not match payload"
        )
    for axis in range(layout.size):
        if layout.dim_at(axis) != 2:
            raise InvalidLayoutError("qstate currently supports qubit axes only")


def density_purity(payload: object) -> float:
    """Return :math:`\\operatorname{Tr}(\\rho^2)` for a validated density payload.

    Parameters
    ----------
    payload : object
        Density payload accepted by :func:`check_density`.

    Returns
    -------
    float
        Purity of the density matrix.
    """
    checked = check_density(payload)
    rho = np.asarray(getattr(checked, "rho"), dtype=np.complex128)
    return float(trace(rho @ rho).real)


def ket_global_phase_aligned_vector(payload: object) -> np.ndarray:
    """Return a ket vector with deterministic global phase.

    Parameters
    ----------
    payload : object
        Ket payload accepted by :func:`check_ket`.

    Returns
    -------
    ndarray of complex, shape ``(2**num_qubits,)``
        Vector divided by the phase of the first non-zero amplitude, making that
        amplitude real and positive.

    Raises
    ------
    InvalidStateError
        If the ket validation fails or no non-zero amplitude can be found after
        validation.
    """
    checked = check_ket(payload)
    vector = np.asarray(getattr(checked, "vector"), dtype=np.complex128)
    for amplitude in vector:
        if abs(amplitude) > ATOL:
            return vector / (amplitude / abs(amplitude))
    raise InvalidStateError("ket vector has no non-zero amplitude")


def density_is_pure(payload: object) -> bool:
    """Return whether a density payload has purity one within tolerance."""
    return bool(np.isclose(density_purity(payload), 1.0, atol=ATOL, rtol=RTOL))


def _is_power_of_two(value: int) -> bool:
    return type(value) is int and value > 0 and (value & (value - 1)) == 0


__all__ = [
    "assert_payload_layout_compatible",
    "check_bell_diag",
    "check_density",
    "check_ket",
    "check_payload",
    "density_is_pure",
    "density_purity",
    "is_bell_diag",
    "is_density",
    "is_ket",
    "ket_global_phase_aligned_vector",
    "payload_hilbert_dim",
    "payload_num_qubits",
]
