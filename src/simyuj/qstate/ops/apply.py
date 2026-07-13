from __future__ import annotations

"""Apply dense unitary operations to ket payloads and full Hilbert spaces.

Axes use the package qubit-axis convention: axis ``0`` is the most significant
computational-basis bit.  The order of the ``axes`` tuple defines the operand
order seen by the local operation matrix.
"""

import numpy as np

from ..math.tensor import apply_unitary_to_axes
from ..state.ket import KetState
from .unitary import Unitary, check_unitary


def apply_unitary_ket(
    state: KetState,
    operation: Unitary,
    axes: tuple[int, ...],
) -> KetState:
    """Apply a unitary operation to selected axes of a ket payload.

    Parameters
    ----------
    state : KetState
        Input ket payload with vector shape ``(2**num_qubits,)``.
    operation : Unitary
        Dense operation whose matrix shape must match ``len(axes)``.
    axes : tuple of int
        Target axes in operation-operand order.

    Returns
    -------
    KetState
        New ket payload containing the updated state vector.

    Raises
    ------
    TypeError
        If ``operation`` is not a ``Unitary`` or if delegated axis checks fail.
    ValueError
        If delegated axis uniqueness/range checks fail.
    DimensionError
        If vector or matrix dimensions do not match ``state.num_qubits`` and
        ``len(axes)``.

    Notes
    -----
    This function validates the operation object but otherwise relies on
    ``apply_unitary_to_axes`` and ``KetState`` for shape and state validation.
    """
    operation = check_unitary(operation)
    return KetState._from_trusted(
        apply_unitary_to_axes(
            state.vector,
            operation.matrix,
            axes=axes,
            num_qubits=state.num_qubits,
        )
    )


def expand_unitary(
    operation: Unitary,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> np.ndarray:
    """Expand a local unitary into a dense full-space matrix.

    Parameters
    ----------
    operation : Unitary
        Local unitary to apply.
    axes : tuple of int
        Target axes in operation-operand order.
    num_qubits : int
        Number of qubits in the full Hilbert space.

    Returns
    -------
    ndarray of complex, shape ``(2**num_qubits, 2**num_qubits)``
        Dense matrix whose column action matches ``apply_unitary_ket`` for the
        same axes.

    Raises
    ------
    TypeError
        If ``operation`` is not a ``Unitary`` or delegated axis/type checks
        fail.
    ValueError
        If delegated axis checks reject ``axes``.
    DimensionError
        If ``num_qubits`` is not positive or the operation shape is
        incompatible with ``axes``.

    Notes
    -----
    The implementation constructs the full matrix column by column.  It is
    intended for small qubit counts where a dense ``2**num_qubits`` operator is
    acceptable.
    """
    operation = check_unitary(operation)
    size = 2**num_qubits
    columns = []
    for basis_index in range(size):
        vector = np.zeros(size, dtype=np.complex128)
        vector[basis_index] = 1.0
        columns.append(
            apply_unitary_to_axes(
                vector,
                operation.matrix,
                axes=axes,
                num_qubits=num_qubits,
            )
        )
    return np.column_stack(columns)


__all__ = ["apply_unitary_ket", "expand_unitary"]
