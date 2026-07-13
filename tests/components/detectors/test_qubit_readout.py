from __future__ import annotations

import pytest

from simyuj.components.connections import PortDelivery, connect_ports
from simyuj.components.detectors.primitives.actions import ACTION_RUN_QUBIT_READOUT
from simyuj.components.detectors.primitives.measurement import Measure
from simyuj.components.detectors.primitives.reports import (
    FLAG_NO_CLICK,
    FLAG_NO_OUTCOME,
    DetectionReport,
)
from simyuj.components.detectors.primitives.result_labels import result_label
from simyuj.components.detectors.qubit_readout import (
    QubitReadoutDevice,
    QubitReadoutJob,
)
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.qstate import SubsystemId
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.binding import binding_context

ACTION_RECEIVE_REPORT = "receive_report"


class ReportSink(Component):
    def __init__(self, device_id: str = "sink") -> None:
        self.device_id = device_id
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.INGRESS,
        )
        self.received: list[DetectionReport] = []
        self.received_times: list[int] = []
        self.received_events: list[Event] = []

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        if event.action != ACTION_RECEIVE_REPORT:
            raise ValueError("unexpected sink action")

        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("sink payload_ref must be PortDelivery")

        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("sink received delivery for wrong port")

        payload = event.payload_ref.payload
        if not isinstance(payload, DetectionReport):
            raise TypeError("sink payload must be DetectionReport")

        self.received.append(payload)
        self.received_times.append(timeline.current_time)
        self.received_events.append(event)


def _prepare_qubit(
    timeline: Timeline,
    *,
    subsystem_label: str = "q0",
    state: str = "|0>",
) -> SubsystemId:
    subsystem = SubsystemId(subsystem_label)
    timeline.qstate.prepare(state, subsystems=(subsystem,))
    return subsystem


def _job(
    *,
    job_id: str = "read-0",
    target: SubsystemId | None = None,
    output_latency_ticks: int | None = None,
) -> QubitReadoutJob:
    return QubitReadoutJob(
        job_id=job_id,
        targets=(SubsystemId("q0") if target is None else target,),
        output_latency_ticks=output_latency_ticks,
    )


def _event(
    *,
    device: QubitReadoutDevice,
    action: str = ACTION_RUN_QUBIT_READOUT,
    payload_ref: object | None = None,
    target_ref: object | None = None,
    time: int = 0,
) -> Event:
    return Event(
        time=time,
        target_ref=device if target_ref is None else target_ref,
        action=action,
        payload_ref=_job() if payload_ref is None else payload_ref,
        source=None,
        subsystem_id="components",
    )


def test_qubit_readout_job_rejects_empty_job_id() -> None:
    with pytest.raises(ValueError):
        QubitReadoutJob(job_id="", targets=(SubsystemId("q0"),))


def test_qubit_readout_job_rejects_empty_targets() -> None:
    with pytest.raises(ValueError, match="targets must be non-empty"):
        QubitReadoutJob(job_id="job", targets=())


def test_qubit_readout_job_rejects_duplicate_targets() -> None:
    target = SubsystemId("q0")

    with pytest.raises(ValueError, match="targets must be unique"):
        QubitReadoutJob(job_id="job", targets=(target, target))


def test_qubit_readout_job_rejects_non_subsystem_targets() -> None:
    with pytest.raises(TypeError, match="targets must contain SubsystemId"):
        QubitReadoutJob(job_id="job", targets=("q0",))  # type: ignore[arg-type]


def test_qubit_readout_job_rejects_negative_output_latency() -> None:
    with pytest.raises(ValueError, match="output_latency_ticks must be non-negative"):
        QubitReadoutJob(
            job_id="job",
            targets=(SubsystemId("q0"),),
            output_latency_ticks=-1,
        )


def test_qubit_readout_job_rejects_bad_meta() -> None:
    with pytest.raises(TypeError, match="meta entries must be 2-tuples"):
        QubitReadoutJob(
            job_id="job",
            targets=(SubsystemId("q0"),),
            meta=(("bad",),),  # type: ignore[arg-type]
        )


def test_qubit_readout_bound_device_processes_scheduled_job() -> None:
    timeline = Timeline(master_seed=123)
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")

    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    timeline.schedule(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-0", targets=(q0,)),
        )
    )
    timeline.run_until_empty()

    assert len(device.reports) == 1
    assert device.reports[0].success is True
    assert device.reports[0].outcome == "0"


def test_qubit_readout_logs_ready_on_bind() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.INFO, sinks=[log_sink]))
    device = QubitReadoutDevice(
        device_id="mem_ro",
        output_latency_ticks=3,
        output_priority=4,
    )

    device.bind(binding_context(timeline))

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.qubit_readout.ready"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "device_id": "mem_ro",
        "readout_model": "IdentityQubitReadout",
        "output_latency_ticks": 3,
        "output_priority": 4,
    }


def test_qubit_readout_bind_is_idempotent_for_same_timeline() -> None:
    timeline = Timeline(master_seed=123)
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")

    device.bind(binding_context(timeline))
    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-0", targets=(q0,)),
        ),
        timeline,
    )

    assert len(device.reports) == 1
    assert device.reports[0].success is True


def test_qubit_readout_cannot_rebind_to_different_timeline() -> None:
    first = Timeline(master_seed=1)
    second = Timeline(master_seed=2)
    device = QubitReadoutDevice(device_id="mem_ro")

    device.bind(binding_context(first))

    with pytest.raises(RuntimeError, match="already bound to another timeline"):
        device.bind(binding_context(second))


def test_qubit_readout_handle_event_requires_bind() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro")

    with pytest.raises(RuntimeError, match="must be bound"):
        device.handle_event(_event(device=device), timeline)


def test_qubit_readout_rejects_wrong_action_after_bind() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro")
    device.bind(binding_context(timeline))

    event = _event(device=device, action="anything", payload_ref=None)

    with pytest.raises(ValueError, match="unsupported event action"):
        device.handle_event(event, timeline)


def test_qubit_readout_rejects_event_for_different_target() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro")
    other = QubitReadoutDevice(device_id="other_ro")
    device.bind(binding_context(timeline))

    event = _event(device=device, target_ref=other)

    with pytest.raises(ValueError, match="target_ref must be this QubitReadoutDevice"):
        device.handle_event(event, timeline)


def test_qubit_readout_rejects_non_readout_job_payload() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro")
    device.bind(binding_context(timeline))

    event = _event(device=device, payload_ref=object())

    with pytest.raises(TypeError, match="payload_ref must be QubitReadoutJob"):
        device.handle_event(event, timeline)


def test_qubit_readout_rejects_event_from_different_timeline() -> None:
    first = Timeline()
    second = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro")
    device.bind(binding_context(first))

    with pytest.raises(RuntimeError, match="different timeline"):
        device.handle_event(_event(device=device), second)


def test_qubit_readout_z_measurement_reports_zero() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    event = Event(
        time=0,
        target_ref=device,
        action=ACTION_RUN_QUBIT_READOUT,
        payload_ref=QubitReadoutJob(job_id="read-0", targets=(q0,)),
        source=None,
        subsystem_id="components",
    )

    device.handle_event(event, timeline)

    report = device.reports[0]

    assert report.success is True
    assert report.outcome == "0"
    assert report.raw_clicks == ()
    assert report.signal_id is None
    assert report.measurement_method == "projective"
    assert report.measurement_label == "z"
    assert report.qstate_result is not None


def test_qubit_readout_logs_successful_report_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]))
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")
    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")
    job = QubitReadoutJob(job_id="read-0", targets=(q0,))

    scheduled = timeline.schedule(
        _event(
            device=device,
            payload_ref=job,
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.qubit_readout.readout"
    )
    report = device.reports[0]

    assert record.level is LogLevel.DEBUG
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_RUN_QUBIT_READOUT
    assert dict(record.meta) == {
        "device_id": "mem_ro",
        "job_id": "read-0",
        "report_id": report.report_id,
        "measurement_label": "z",
        "success": True,
        "outcome": "0",
        "flags": (),
    }


def test_qubit_readout_x_measurement_reports_plus() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro", measurement="x")
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|+>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-x", targets=(q0,)),
        ),
        timeline,
    )

    report = device.reports[0]

    assert report.outcome == "+"
    assert report.raw_clicks == ()


def test_qubit_readout_confusion_map_changes_reported_outcome_only() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement="z",
        readout_model={
            "0": {"1": 1.0},
            "1": {"0": 1.0},
        },
    )
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-noisy", targets=(q0,)),
        ),
        timeline,
    )

    report = device.reports[0]

    assert report.outcome == "1"
    assert result_label(report.qstate_result) == "0"
    assert report.raw_clicks == ()


def test_qubit_readout_accepts_callable_readout_model() -> None:
    def append_suffix(
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: object,
        context: object,
        rng: object | None,
    ) -> object | None:
        del qstate_result, measurement_call, context, rng
        return f"{true_outcome}:reported"

    timeline = Timeline()
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement="z",
        readout_model=append_suffix,
    )
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-callable", targets=(q0,)),
        ),
        timeline,
    )

    report = device.reports[0]

    assert report.outcome == "0:reported"


def test_qubit_readout_none_measurement_reports_no_outcome() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement=Measure.none(),
    )
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-none", targets=(q0,)),
        ),
        timeline,
    )

    report = device.reports[0]

    assert report.success is False
    assert report.outcome is None
    assert report.raw_clicks == ()
    assert FLAG_NO_OUTCOME in report.flags
    assert FLAG_NO_CLICK not in report.flags


def test_qubit_readout_logs_no_outcome_report_at_trace() -> None:
    log_sink = MemorySink()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement=Measure.none(),
    )
    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")
    job = QubitReadoutJob(job_id="read-none", targets=(q0,))

    scheduled = timeline.schedule(
        _event(
            device=device,
            payload_ref=job,
        )
    )
    timeline.run_until(0)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.detectors.qubit_readout.readout"
    )
    report = device.reports[0]

    assert record.level is LogLevel.TRACE
    assert record.event_id == scheduled.event_id
    assert record.action == ACTION_RUN_QUBIT_READOUT
    assert dict(record.meta) == {
        "device_id": "mem_ro",
        "job_id": "read-none",
        "report_id": report.report_id,
        "measurement_label": "none",
        "success": False,
        "outcome": None,
        "flags": (FLAG_NO_OUTCOME,),
    }


def test_qubit_readout_report_includes_random_measurement_selection() -> None:
    timeline = Timeline(master_seed=123)
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement=Measure.random({"z": 0.5, "x": 0.5}),
    )
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-random", targets=(q0,)),
        ),
        timeline,
    )

    report = device.reports[0]

    assert report.selection_index in {0, 1}
    assert report.selection_probability == 0.5
    assert report.selection_label in {"z", "x"}


def test_qubit_readout_unconnected_output_only_stores_report() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")
    device.bind(binding_context(timeline))

    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-store-only", targets=(q0,)),
        ),
        timeline,
    )

    assert len(device.reports) == 1
    assert timeline.events_scheduled == 0


def test_qubit_readout_connected_output_transmits_report() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")
    sink = ReportSink()

    connect_ports(
        device.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-output", targets=(q0,)),
        ),
        timeline,
    )

    assert len(device.reports) == 1
    assert timeline.events_scheduled == 1

    timeline.run_one_step()

    assert sink.received == [device.reports[0]]
    assert sink.received[0] is device.reports[0]
    assert sink.received_times == [0]


def test_scheduled_qubit_readout_transmits_report_through_output_port() -> None:
    timeline = Timeline(master_seed=123)
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement="z",
        output_latency_ticks=4,
    )
    sink = ReportSink()

    connect_ports(
        device.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )
    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|1>")

    scheduled = timeline.schedule(
        Event(
            time=3,
            target_ref=device,
            action=ACTION_RUN_QUBIT_READOUT,
            payload_ref=QubitReadoutJob(job_id="read-scheduled-output", targets=(q0,)),
        )
    )

    timeline.run_until(3)

    assert len(device.reports) == 1
    assert device.reports[0].success is True
    assert device.reports[0].outcome == "1"
    assert sink.received == []

    timeline.run_until_empty()

    assert sink.received == [device.reports[0]]
    assert sink.received_times == [7]
    assert sink.received_events[0].event_id != scheduled.event_id
    assert sink.received_events[0].target_ref is sink


def test_qubit_readout_output_latency_is_respected() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement="z",
        output_latency_ticks=7,
    )
    sink = ReportSink()

    connect_ports(
        device.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-latency", targets=(q0,)),
        ),
        timeline,
    )

    assert timeline.current_time == 0
    assert sink.received == []

    timeline.run_one_step()

    assert timeline.current_time == 7
    assert sink.received == [device.reports[0]]
    assert sink.received_times == [7]


def test_qubit_readout_job_output_latency_overrides_device_latency() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(
        device_id="mem_ro",
        measurement="z",
        output_latency_ticks=7,
    )
    sink = ReportSink()

    connect_ports(
        device.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(
                job_id="read-job-latency",
                targets=(q0,),
                output_latency_ticks=2,
            ),
        ),
        timeline,
    )

    timeline.run_one_step()

    assert timeline.current_time == 2
    assert sink.received == [device.reports[0]]
    assert sink.received_times == [2]


def test_qubit_readout_output_event_metadata_identifies_report() -> None:
    timeline = Timeline()
    device = QubitReadoutDevice(device_id="mem_ro", measurement="z")
    sink = ReportSink()

    connect_ports(
        device.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_REPORT,
    )

    device.bind(binding_context(timeline))
    q0 = _prepare_qubit(timeline, state="|0>")

    device.handle_event(
        _event(
            device=device,
            payload_ref=QubitReadoutJob(job_id="read-meta", targets=(q0,)),
        ),
        timeline,
    )

    timeline.run_one_step()

    event = sink.received_events[0]
    report = device.reports[0]

    assert event.meta["device_id"] == "mem_ro"
    assert event.meta["output_port"] == "out"
    assert event.meta["report_id"] == report.report_id
    assert event.meta["job_id"] == "read-meta"
    assert event.meta["qubit_readout_device"] == "mem_ro"
