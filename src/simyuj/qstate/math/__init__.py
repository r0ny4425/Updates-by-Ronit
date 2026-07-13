from __future__ import annotations

"""Public numerical helpers for ``qstate``.

This package collects small dense-array routines used by the new quantum-state
backend.  Arrays are represented with NumPy and use ``complex128`` by default.
Tensor products follow the operand order supplied by the caller; qubit-specific
helpers use computational-basis vectors ordered by integer index.
"""

from .const import ATOL, COMPLEX_DTYPE, PROB_ATOL, RTOL, SQRT2
from .linalg import (
    dagger,
    is_hermitian,
    is_psd,
    is_square_matrix,
    is_unitary,
    normalize_density,
    normalize_vector,
    trace,
)
from .prob import (
    argmax_prob,
    check_prob_vector,
    clip_prob,
    is_deterministic,
    normalize_prob_vector,
    normalize_weights,
    safe_real,
)
from .projector import (
    basis_projectors,
    computational_projectors,
    is_projector,
    outer,
    tensor_projectors,
    vector_projector,
)
from .tensor import (
    apply_operator_to_axes,
    apply_unitary_to_axes,
    expand_operator,
    kron,
    kron_all,
)

# Public numerical helper surface for ``simyuj.qstate.math``.
__all__ = [
    "ATOL",
    "COMPLEX_DTYPE",
    "PROB_ATOL",
    "RTOL",
    "SQRT2",
    "dagger",
    "is_hermitian",
    "is_psd",
    "is_square_matrix",
    "is_unitary",
    "normalize_density",
    "normalize_vector",
    "trace",
    "argmax_prob",
    "check_prob_vector",
    "clip_prob",
    "is_deterministic",
    "normalize_prob_vector",
    "normalize_weights",
    "safe_real",
    "basis_projectors",
    "computational_projectors",
    "is_projector",
    "outer",
    "tensor_projectors",
    "vector_projector",
    "apply_operator_to_axes",
    "apply_unitary_to_axes",
    "expand_operator",
    "kron",
    "kron_all",
]
