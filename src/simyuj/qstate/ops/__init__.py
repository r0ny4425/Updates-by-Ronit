from __future__ import annotations

"""Public operation constructors and helpers for ``qstate``.

This package namespace gathers dense ``Unitary`` objects, parameterized gate
factories, density-reset wrappers, and classical Pauli-frame bookkeeping
helpers.  Multi-qubit operation operands follow the same order as the backing
matrices: the first operand is the most significant computational-basis bit.
"""

from .frame import (
    PauliFrame,
    correction_for_bell,
    correction_for_entanglement_swap,
    identity_frame,
    update_after_bsm,
    update_after_swap,
    update_after_teleport,
)
from .gates import (
    CCX,
    CCZ,
    CNOT,
    CSWAP,
    CZ,
    FREDKIN,
    MCX,
    MCZ,
    SWAP,
    TOFFOLI,
    H,
    I,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
)
from .reset import discard_and_prepare, reset_one, reset_plus, reset_zero
from .rotations import CRX, CRY, CRZ, RX, RY, RZ, CPhase, Phase
from .unitary import Unitary, check_unitary, identity, unitary

# Public operation surface for ``simyuj.qstate.ops``.
__all__ = [
    "CNOT",
    "CPhase",
    "CRX",
    "CRY",
    "CRZ",
    "CCX",
    "CCZ",
    "CSWAP",
    "CZ",
    "FREDKIN",
    "H",
    "I",
    "MCX",
    "MCZ",
    "S",
    "Sdg",
    "SWAP",
    "T",
    "Tdg",
    "TOFFOLI",
    "X",
    "Y",
    "Z",
    "Phase",
    "RX",
    "RY",
    "RZ",
    "PauliFrame",
    "Unitary",
    "check_unitary",
    "correction_for_bell",
    "correction_for_entanglement_swap",
    "discard_and_prepare",
    "identity",
    "identity_frame",
    "reset_one",
    "reset_plus",
    "reset_zero",
    "unitary",
    "update_after_bsm",
    "update_after_swap",
    "update_after_teleport",
]
