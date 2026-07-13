from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from simyuj.components.detectors.primitives.reports import DetectionReport
from simyuj.components.memories import (
    MemoryAbsorbReport,
    MemoryDiscardReport,
    MemoryEmitReport,
    MemoryExpireReport,
    MemoryMeasurementReport,
    MemoryMetaUpdateReport,
    MemoryOperatorReport,
    memory_subsystem_id,
)
from simyuj.qstate import SubsystemId


def _memory_subsystem(position: int = 0) -> SubsystemId:
    return memory_subsystem_id("nodeA.mem0", position)


def _detection_report() -> DetectionReport:
    return DetectionReport(
        report_id="det-report-1",
        device_id="nodeA.mem0.readout",
        time=14,
        success=True,
        outcome="0",
        raw_clicks=(),
        measurement_method="projective",
        measurement_label="z",
    )


def _field_names(record_type: type[Any]) -> set[str]:
    return {field.name for field in fields(record_type)}


def test_absorb_report_is_classical_position_notice() -> None:
    report = MemoryAbsorbReport(
        report_id="absorb-report-1",
        memory_id="nodeA.mem0",
        time=10,
        success=True,
        position=0,
        input_signal_id="signal-1",
        memory_subsystem=_memory_subsystem(),
        status="occupied",
        session_id="session-1",
        meta=(("port", "qin"),),
    )

    assert report.position == 0
    assert report.input_signal_id == "signal-1"
    assert report.memory_subsystem == _memory_subsystem()
    assert report.status == "occupied"
    assert "slot_key" not in _field_names(MemoryAbsorbReport)


def test_emit_report_carries_memory_and_output_subsystems() -> None:
    output_subsystem = SubsystemId("photon:nodeA.mem0:position:0:emit:7")

    report = MemoryEmitReport(
        report_id="emit-report-1",
        memory_id="nodeA.mem0",
        time=20,
        success=True,
        position=0,
        memory_subsystem=_memory_subsystem(),
        output_signal_id="signal-out-1",
        output_subsystem=output_subsystem,
        status="emitted",
    )

    assert report.memory_subsystem == _memory_subsystem()
    assert report.output_signal_id == "signal-out-1"
    assert report.output_subsystem == output_subsystem


def test_operator_report_preserves_ordered_positions_and_subsystems() -> None:
    q2 = _memory_subsystem(2)
    q0 = _memory_subsystem(0)

    report = MemoryOperatorReport(
        report_id="operator-report-1",
        memory_id="nodeA.mem0",
        time=30,
        success=True,
        positions=(2, 0),
        memory_subsystems=(q2, q0),
        status="applied",
        meta=(("operator_name", "CNOT"),),
    )

    assert report.positions == (2, 0)
    assert report.memory_subsystems == (q2, q0)
    assert "operator" not in _field_names(MemoryOperatorReport)


def test_measurement_report_reuses_detection_report_shape() -> None:
    detection_report = _detection_report()

    report = MemoryMeasurementReport(
        report_id="measure-report-1",
        memory_id="nodeA.mem0",
        time=40,
        success=True,
        positions=(0,),
        memory_subsystems=(_memory_subsystem(),),
        detection_report=detection_report,
        destructive=True,
        cleared_positions=(0,),
        status="measured",
    )

    assert report.detection_report is detection_report
    assert report.destructive is True
    assert report.cleared_positions == (0,)


def test_measurement_report_can_be_non_destructive() -> None:
    report = MemoryMeasurementReport(
        report_id="measure-report-1",
        memory_id="nodeA.mem0",
        time=40,
        success=True,
        positions=(0,),
        memory_subsystems=(_memory_subsystem(),),
        detection_report=_detection_report(),
        destructive=False,
        cleared_positions=(),
        status="measured",
    )

    assert report.destructive is False
    assert report.cleared_positions == ()


def test_discard_and_expire_reports_carry_status_and_reason() -> None:
    discard = MemoryDiscardReport(
        report_id="discard-report-1",
        memory_id="nodeA.mem0",
        time=50,
        success=True,
        position=0,
        memory_subsystem=_memory_subsystem(),
        status="discarded",
        reason="user_requested",
    )
    expire = MemoryExpireReport(
        report_id="expire-report-1",
        memory_id="nodeA.mem0",
        time=60,
        success=True,
        position=0,
        memory_subsystem=_memory_subsystem(),
        status="expired",
        reason="lifetime_elapsed",
    )

    assert discard.reason == "user_requested"
    assert expire.reason == "lifetime_elapsed"


def test_meta_update_report_carries_updated_removed_keys_and_token() -> None:
    report = MemoryMetaUpdateReport(
        report_id="meta-update-report-1",
        memory_id="nodeA.mem0",
        time=70,
        success=True,
        position=0,
        memory_subsystem=_memory_subsystem(),
        occupancy_token=9,
        status="updated",
        updated_keys=("pair_id",),
        removed_keys=("old_pair",),
        session_id="session-1",
        meta=(("request_id", "meta-1"),),
    )

    assert report.position == 0
    assert report.memory_subsystem == _memory_subsystem()
    assert report.occupancy_token == 9
    assert report.updated_keys == ("pair_id",)
    assert report.removed_keys == ("old_pair",)


def test_failed_meta_update_report_can_omit_subsystem() -> None:
    report = MemoryMetaUpdateReport(
        report_id="meta-update-report-1",
        memory_id="nodeA.mem0",
        time=70,
        success=False,
        position=0,
        memory_subsystem=None,
        occupancy_token=0,
        status="not_occupied:empty",
    )

    assert report.success is False
    assert report.memory_subsystem is None


def test_unsuccessful_reports_can_omit_subsystem_outputs() -> None:
    report = MemoryEmitReport(
        report_id="emit-report-1",
        memory_id="nodeA.mem0",
        time=20,
        success=False,
        position=0,
        memory_subsystem=None,
        output_signal_id=None,
        output_subsystem=None,
        status="empty",
    )

    assert report.success is False
    assert report.output_subsystem is None


def test_successful_reports_require_success_outputs() -> None:
    with pytest.raises(ValueError, match="memory_subsystem"):
        MemoryAbsorbReport(
            report_id="absorb-report-1",
            memory_id="nodeA.mem0",
            time=10,
            success=True,
            position=0,
            input_signal_id="signal-1",
            memory_subsystem=None,
            status="occupied",
        )

    with pytest.raises(ValueError, match="detection_report"):
        MemoryMeasurementReport(
            report_id="measure-report-1",
            memory_id="nodeA.mem0",
            time=40,
            success=True,
            positions=(0,),
            memory_subsystems=(_memory_subsystem(),),
            detection_report=None,
            destructive=True,
            cleared_positions=(0,),
            status="measured",
        )


def test_ordered_report_subsystems_must_match_positions_when_present() -> None:
    with pytest.raises(ValueError, match="match positions"):
        MemoryOperatorReport(
            report_id="operator-report-1",
            memory_id="nodeA.mem0",
            time=30,
            success=True,
            positions=(0, 1),
            memory_subsystems=(_memory_subsystem(),),
            status="applied",
        )


def test_reports_validate_common_fields_and_meta() -> None:
    with pytest.raises(ValueError, match="report_id"):
        MemoryDiscardReport(
            report_id="",
            memory_id="nodeA.mem0",
            time=1,
            success=False,
            position=0,
            memory_subsystem=None,
            status="failed",
            reason="missing",
        )

    with pytest.raises(TypeError, match="meta"):
        MemoryDiscardReport(
            report_id="discard-report-1",
            memory_id="nodeA.mem0",
            time=1,
            success=False,
            position=0,
            memory_subsystem=None,
            status="failed",
            reason="missing",
            meta=(("bad", []),),
        )
