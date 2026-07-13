from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components import (
    ACTION_TRANSMIT_QUANTUM,
    Port,
    PortDelivery,
    PortDirection,
    PortKind,
    QuantumChannel,
    SinglePhotonSource,
    connect_ports,
)
from simyuj.components.detectors import (
    ACTION_DETECT_SIGNAL,
    DetectionReport,
    DetectorArray,
    SinglePhotonDetector,
    SinglePhotonDetectorParams,
)
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.qstate import StateNotFoundError
from tests.support.binding import binding_context

ACTION_RECEIVE_REPORT = "receive_report"


@dataclass(slots=True)
class ReportSink(Component):
    device_id: str = "report_sink"
    reports: list[tuple[int, DetectionReport]] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if event.action != ACTION_RECEIVE_REPORT:
            raise ValueError(event.action)
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("delivery arrived on wrong report port")
        report = event.payload_ref.payload
        if not isinstance(report, DetectionReport):
            raise TypeError("payload must be DetectionReport")
        self.reports.append((timeline.current_time, report))


def _perfect_detector(detector_id: str) -> SinglePhotonDetector:
    return SinglePhotonDetector(
        detector_id=detector_id,
        params=SinglePhotonDetectorParams(
            efficiency=1.0,
            dark_count_rate_hz=0.0,
        ),
    )


def _detector_array() -> DetectorArray:
    return DetectorArray(
        device_id="bob_rx",
        detectors=(_perfect_detector("d0"), _perfect_detector("d1")),
        measurement="z",
        readout={"z": {"0": "d0", "1": "d1"}},
        output_latency_ticks=2,
    )


def _source_channel_detector_chain(
    *,
    channel_loss_db: float = 0.0,
) -> tuple[Timeline, SinglePhotonSource, QuantumChannel, DetectorArray, ReportSink]:
    timeline = Timeline(master_seed=1)
    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
    )
    channel = QuantumChannel(
        channel_id="alice_to_bob",
        delay_ticks=5,
        fixed_insertion_loss_db=channel_loss_db,
    )
    detector = _detector_array()
    sink = ReportSink()

    connect_ports(
        source.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
    )
    connect_ports(
        channel.output_port,
        detector.input_port,
        target_action=ACTION_DETECT_SIGNAL,
    )
    connect_ports(
        detector.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    channel.bind(binding_context(timeline))
    detector.bind(binding_context(timeline))
    source.schedule_start(timeline)

    return timeline, source, channel, detector, sink


def test_single_photon_source_channel_detector_public_workflow() -> None:
    timeline, source, channel, detector, sink = _source_channel_detector_chain()

    timeline.run_until_empty()

    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.lost_count == 0
    assert len(detector.reports) == 1
    assert sink.reports == [(7, detector.reports[0])]

    report = sink.reports[0][1]
    assert report.success is True
    assert report.outcome == "0"
    assert report.signal_id == "alice_source:photon:1"
    assert report.measurement_label == "z"
    assert len(report.raw_clicks) == 1
    assert report.raw_clicks[0].detector_id == "d0"

    prepared = source.reports[0]
    assert prepared.signal_ids == ("alice_source:photon:1",)
    with pytest.raises(StateNotFoundError):
        timeline.qstate.state_of(prepared.state_targets[0])


def test_quantum_channel_loss_stops_detector_workflow_deterministically() -> None:
    def run_once() -> tuple[int, int, int, int, int]:
        timeline, source, channel, detector, sink = _source_channel_detector_chain(
            channel_loss_db=1e9,
        )

        timeline.run_until_empty()

        prepared = source.reports[0]
        with pytest.raises(StateNotFoundError):
            timeline.qstate.state_of(prepared.state_targets[0])

        return (
            channel.received_count,
            channel.delivered_count,
            channel.lost_count,
            len(detector.reports),
            len(sink.reports),
        )

    assert run_once() == (1, 0, 1, 0, 0)
    assert run_once() == (1, 0, 1, 0, 0)
