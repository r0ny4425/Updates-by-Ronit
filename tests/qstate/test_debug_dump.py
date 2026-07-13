from __future__ import annotations

import pytest

from simyuj.qstate import QuantumStateManager, QuantumStateRecord, StateLayout
from simyuj.qstate.debug import (
    dump_layout,
    dump_payload_summary,
    dump_record,
    dump_store,
)
from simyuj.qstate.space import SubsystemId
from simyuj.qstate.state.bell_diag import BellDiagState
from simyuj.qstate.state.convert import ket_to_density
from simyuj.qstate.state.make import basis


def q(name: str) -> SubsystemId:
    return SubsystemId(name)


def _layout(*names: str) -> StateLayout:
    return StateLayout(tuple(q(name) for name in names), (2,) * len(names))


def test_dump_layout_returns_single_line_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = dump_layout(_layout("q0", "q1"))

    assert "\n" not in text
    assert text.startswith("StateLayout(")
    assert "size=2" in text
    assert "hilbert_dim=4" in text
    assert "axis=0:q0[dim=2]" in text
    assert "axis=1:q1[dim=2]" in text
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_dump_payload_summary_summarizes_supported_payload_shapes() -> None:
    ket = basis("0")
    density = ket_to_density(ket)
    bell_diag = BellDiagState.from_label("psi-")

    ket_summary = dump_payload_summary(ket)
    density_summary = dump_payload_summary(density)
    bell_diag_summary = dump_payload_summary(bell_diag)

    assert "KetState" in ket_summary
    assert "num_qubits=1" in ket_summary
    assert "vector_shape=(2,)" in ket_summary
    assert "DensityState" in density_summary
    assert "rho_shape=(2, 2)" in density_summary
    assert "trace=1" in density_summary
    assert "purity=1" in density_summary
    assert "BellDiagState" in bell_diag_summary
    assert "num_qubits=2" in bell_diag_summary
    assert "max_prob=1.0" in bell_diag_summary
    assert dump_payload_summary({"raw": "payload"}) == "dict"


def test_dump_record_includes_ref_rep_payload_and_layout() -> None:
    record = QuantumStateRecord(basis("0"), "ket", _layout("q0"))
    text = dump_record(record)

    assert text.startswith("QuantumStateRecord(")
    assert "state_ref=unbound" in text
    assert "rep='ket'" in text
    assert "KetState" in text
    assert "StateLayout" in text
    assert "axis=0:q0[dim=2]" in text
    assert dump_record(record, state_ref=3).startswith(
        "QuantumStateRecord(state_ref=3, rep='ket'"
    )


def test_dump_store_orders_records_and_returns_multiline_string(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = QuantumStateManager()
    manager.prepare("|0>", subsystems=(q("q0"),))
    manager.prepare("|1>", subsystems=(q("q1"),))

    text = dump_store(manager.store)
    lines = text.splitlines()

    assert lines[0] == "QuantumStateStore(size=2, records=2)"
    assert len(lines) == 3
    assert lines[1].startswith("  QuantumStateRecord(state_ref=0")
    assert lines[2].startswith("  QuantumStateRecord(state_ref=1")
    assert "axis=0:q0[dim=2]" in lines[1]
    assert "axis=0:q1[dim=2]" in lines[2]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_dump_helpers_validate_debug_inputs() -> None:
    record = QuantumStateRecord(basis("0"), "ket", _layout("q0"))

    with pytest.raises(TypeError, match="StateLayout"):
        dump_layout(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="QuantumStateRecord"):
        dump_record(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="state_ref"):
        dump_record(record, state_ref="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        dump_record(record, state_ref=-1)
    with pytest.raises(TypeError, match="does not expose"):
        dump_store(object())


def test_dump_record_for_density_state() -> None:
    manager = QuantumStateManager()
    q0 = q("q0")

    state_ref = manager.prepare("|0>", rep="density", subsystems=(q0,))
    record = manager.record(state_ref)

    text = dump_record(record)

    assert "QuantumStateRecord" in text
    assert "rep='density'" in text
    assert "DensityState" in text
    assert "trace=" in text
    assert "purity=" in text
