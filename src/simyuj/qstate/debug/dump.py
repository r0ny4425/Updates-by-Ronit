from __future__ import annotations

"""String dump helpers for qstate layouts, payloads, records, and stores.

The functions in this module are side-effect free: they return compact summaries
for debugging and tests instead of printing to stdout or mutating the store.
"""

import numpy as np

from ..ids import StateRef
from ..math.linalg import trace
from ..record import QuantumStateRecord
from ..space.layout import StateLayout
from .invariant import iter_store_records


def dump_layout(layout: StateLayout) -> str:
    """Return a single-line summary of a state layout.

    Parameters
    ----------
    layout : StateLayout
        Layout whose subsystem order and dimensions should be rendered.

    Returns
    -------
    str
        Summary containing layout size, total Hilbert-space dimension, and one
        entry per tensor axis.

    Raises
    ------
    TypeError
        If ``layout`` is not a ``StateLayout``.
    """
    if not isinstance(layout, StateLayout):
        raise TypeError("layout must be StateLayout")

    axes = ", ".join(
        f"axis={axis}:{subsystem}[dim={layout.dims[axis]}]"
        for axis, subsystem in enumerate(layout.subsystems)
    )
    return (
        f"StateLayout(size={layout.size}, hilbert_dim={layout.hilbert_dim}, "
        f"axes=[{axes}])"
    )


def dump_payload_summary(payload: object) -> str:
    """Return a compact summary for a supported qstate payload.

    The helper recognizes payloads by common attributes: ``vector`` for ket-like
    records, ``rho`` for density-like records, and ``probs`` for Bell-diagonal
    records. Unknown objects are summarized by type name only.

    Parameters
    ----------
    payload : object
        Payload object to summarize.

    Returns
    -------
    str
        Shape and scalar summary text for recognized payloads, or the type name
        for unknown payloads.

    Notes
    -----
    Density summaries compute trace and purity when ``rho`` is a square matrix.
    The function reports real parts only and does not validate density-matrix
    semantics.
    """
    payload_type = type(payload).__name__
    num_qubits = getattr(payload, "num_qubits", None)
    prefix = (
        f"{payload_type}(num_qubits={num_qubits})"
        if type(num_qubits) is int
        else payload_type
    )

    vector = getattr(payload, "vector", None)
    if vector is not None and hasattr(vector, "shape"):
        return f"{prefix} vector_shape={tuple(vector.shape)}"

    rho = getattr(payload, "rho", None)
    if rho is not None and hasattr(rho, "shape"):
        matrix = np.asarray(rho)
        rho_text = f"{prefix} rho_shape={tuple(matrix.shape)}"
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            tr = trace(matrix)
            purity = trace(matrix @ matrix)
            rho_text += f" trace={float(tr.real):.12g} purity={float(purity.real):.12g}"
        return rho_text

    probs = getattr(payload, "probs", None)
    if probs is not None:
        checked = tuple(probs)
        text = f"{prefix} probs={checked}"
        if checked:
            text += f" max_prob={max(checked)}"
        return text

    return prefix


def dump_record(
    record: QuantumStateRecord,
    *,
    state_ref: StateRef | None = None,
) -> str:
    """Return a single-line summary of a quantum state record.

    Parameters
    ----------
    record : QuantumStateRecord
        Record to summarize.
    state_ref : StateRef or None, optional
        Optional store reference to include in the rendered text.

    Returns
    -------
    str
        Summary containing reference, representation, payload summary, and
        layout summary.

    Raises
    ------
    TypeError
        If ``record`` is not a ``QuantumStateRecord`` or ``state_ref`` is not an
        ``int`` when provided.
    ValueError
        If ``state_ref`` is negative.
    """
    if not isinstance(record, QuantumStateRecord):
        raise TypeError("record must be QuantumStateRecord")
    if state_ref is not None:
        if type(state_ref) is not int:
            raise TypeError("state_ref must be int")
        if state_ref < 0:
            raise ValueError("state_ref must be non-negative")

    ref_text = "unbound" if state_ref is None else str(state_ref)
    return (
        f"QuantumStateRecord(state_ref={ref_text}, rep={record.rep!r}, "
        f"payload={dump_payload_summary(record.payload)}, "
        f"layout={dump_layout(record.layout)})"
    )


def dump_store(store: object) -> str:
    """Return a multiline summary of a store-like object.

    Parameters
    ----------
    store : object
        Store-like object accepted by
        :func:`simyuj.qstate.debug.invariant.iter_store_records`.

    Returns
    -------
    str
        Header line with store size followed by one line per live record.

    Raises
    ------
    TypeError
        If ``store`` does not expose a supported record-iteration shape.
    """
    records = iter_store_records(store)
    size = len(records)
    if hasattr(store, "size") and callable(store.size):
        size = store.size()

    lines = [f"QuantumStateStore(size={size}, records={len(records)})"]
    lines.extend(
        f"  {dump_record(record, state_ref=state_ref)}" for state_ref, record in records
    )
    return "\n".join(lines)


__all__ = ["dump_layout", "dump_payload_summary", "dump_record", "dump_store"]
