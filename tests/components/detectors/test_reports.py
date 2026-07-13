from __future__ import annotations

from simyuj.components.detectors.primitives.reports import (
    FLAG_SIGNAL_CLICK,
    DetectionReport,
    RawClick,
)


def test_raw_click_and_detection_report_records() -> None:
    click = RawClick(
        detector_id="det-0",
        time=10,
        trigger="signal",
        outcome_label="z0",
        flags=(FLAG_SIGNAL_CLICK,),
        meta=(("basis", "z"),),
    )
    report = DetectionReport(
        report_id="report-0",
        device_id="det-array",
        time=10,
        success=True,
        outcome="z0",
        raw_clicks=(click,),
        measurement_method="threshold",
        signal_id="sig-0",
    )

    assert report.raw_clicks == (click,)
    assert report.success is True
    assert report.qstate_result is None
