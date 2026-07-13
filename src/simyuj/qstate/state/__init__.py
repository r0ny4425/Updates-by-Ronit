from __future__ import annotations

"""Public state payloads, constructors, conversions, checks, and metrics.

The ``state`` package exposes the representation layer used by
``QuantumStateManager``.  Current concrete payloads are dense normalized kets,
dense density matrices, and compact two-qubit Bell-diagonal probability records.
Dense qubit states use computational-basis order with axis ``0`` as the most
significant tensor axis.
"""

from .bell_diag import (
    BELL_LABELS,
    BellDiagHandler,
    BellDiagState,
    bell_index,
    bits_to_label,
    label_to_bits,
    make_bell_diag,
    normalize_bell_label,
    werner,
)
from .check import (
    assert_payload_layout_compatible,
    check_bell_diag,
    check_density,
    check_ket,
    check_payload,
    is_bell_diag,
    is_density,
    is_ket,
    payload_hilbert_dim,
    payload_num_qubits,
)
from .convert import (
    as_rep,
    bell_diag_to_density,
    bell_diag_to_ket_if_pure,
    density_to_bell_diag_if_exact,
    density_to_ket_if_pure,
    ket_to_bell_diag_if_exact,
    ket_to_density,
)
from .density import DensityHandler, DensityState
from .ket import KetHandler, KetState
from .make import basis, bell, ghz, make_ket, minus, minus_i, one, plus, plus_i, zero
from .metric import (
    bell_fidelity,
    concurrence,
    entropy,
    fidelity,
    log_negativity,
    max_chsh_value,
    negativity,
    purity,
)
from .reduce import discard_density, drop_axes, keep_axes, partial_trace, reset_density
from .registry import StateRegistry, normalize_rep

# Public state representation surface for ``simyuj.qstate.state``.
__all__ = [
    "BELL_LABELS",
    "BellDiagHandler",
    "BellDiagState",
    "DensityHandler",
    "DensityState",
    "KetHandler",
    "KetState",
    "StateRegistry",
    "as_rep",
    "assert_payload_layout_compatible",
    "bell",
    "bell_diag_to_density",
    "bell_diag_to_ket_if_pure",
    "bell_fidelity",
    "bell_index",
    "basis",
    "bits_to_label",
    "check_bell_diag",
    "check_density",
    "check_ket",
    "check_payload",
    "concurrence",
    "density_to_bell_diag_if_exact",
    "density_to_ket_if_pure",
    "discard_density",
    "drop_axes",
    "entropy",
    "fidelity",
    "ghz",
    "is_bell_diag",
    "is_density",
    "is_ket",
    "keep_axes",
    "ket_to_bell_diag_if_exact",
    "ket_to_density",
    "label_to_bits",
    "log_negativity",
    "make_bell_diag",
    "make_ket",
    "max_chsh_value",
    "minus",
    "minus_i",
    "negativity",
    "normalize_bell_label",
    "normalize_rep",
    "one",
    "partial_trace",
    "payload_hilbert_dim",
    "payload_num_qubits",
    "plus",
    "plus_i",
    "purity",
    "reset_density",
    "werner",
    "zero",
]
