from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from simyuj.components import Port, PortDelivery, connect_ports
from simyuj.components.ports import PortDirection, PortKind
from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.components.sources import (
    ACTION_EMIT,
    DeltaTiming,
    EntangledPairSource,
    SourcePreparationReport,
)
from simyuj.components.sources.single_photon_source import EmissionAttempt
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.qstate import DensityState, StateSampler, SubsystemId
from simyuj.qstate.noise import two_qubit_depolarizing
from simyuj.signal import Signal, SignalKind
from tests.support.binding import binding_context

ATOL = 1e-12
PSD_ATOL = 1e-10


@dataclass(slots=True)
class QuantumSink(Component):
    device_id: str
    received: list[PortDelivery] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")
        self.received.append(event.payload_ref)


def _received_signal(sink: QuantumSink) -> Signal:
    assert len(sink.received) == 1
    payload = sink.received[0].payload
    assert isinstance(payload, Signal)
    return payload


def _schedule_emit(source: EntangledPairSource, timeline: Timeline) -> None:
    timeline.schedule(
        Event(
            time=0,
            target_ref=source,
            action=ACTION_EMIT,
            payload_ref=EmissionAttempt(
                emission_slot_tick=0,
                emission_delay_ticks=0,
            ),
        )
    )
    timeline.run_until(0)


def _phi_plus_projector() -> np.ndarray:
    vector = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    return np.outer(vector, vector.conj())


def _assert_valid_density(rho: np.ndarray) -> None:
    assert rho == pytest.approx(rho.conj().T, abs=ATOL)
    assert np.trace(rho) == pytest.approx(1.0, abs=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) >= -PSD_ATOL


def test_entangled_pair_source_requires_two_qubit_sampler() -> None:
    sampler = StateSampler(states=("|0>",))

    with pytest.raises(ValueError, match="two-qubit sampler"):
        EntangledPairSource(
            device_id="eps",
            frequency_hz=1e6,
            sampler=sampler,
        )


def test_entangled_pair_source_rejects_timing_delay_reaching_period() -> None:
    with pytest.raises(ValueError, match="strictly smaller"):
        EntangledPairSource(
            device_id="eps",
            frequency_hz=5e11,
            timing_profile=DeltaTiming(emission_delay_ticks=2),
        )


def test_entangled_pair_source_emits_two_one_target_member_signals() -> None:
    timeline = Timeline(master_seed=1)
    source = EntangledPairSource(device_id="eps", frequency_hz=1e6)
    left_sink = QuantumSink(device_id="left")
    right_sink = QuantumSink(device_id="right")
    connect_ports(source.left_output_port, left_sink.input_port, target_action="recv")
    connect_ports(source.right_output_port, right_sink.input_port, target_action="recv")
    source.bind(binding_context(timeline))

    _schedule_emit(source, timeline)

    left_signal = _received_signal(left_sink)
    right_signal = _received_signal(right_sink)

    assert left_signal.signal_kind is SignalKind.ENTANGLED_MEMBER
    assert right_signal.signal_kind is SignalKind.ENTANGLED_MEMBER
    assert left_signal.state_ref == right_signal.state_ref
    assert len(left_signal.state_targets) == 1
    assert len(right_signal.state_targets) == 1
    assert left_signal.state_targets[0].label == "eps:pair:1:left"
    assert right_signal.state_targets[0].label == "eps:pair:1:right"
    assert (
        timeline.qstate.state_of(SubsystemId("eps:pair:1:left"))
        == left_signal.state_ref
    )
    assert (
        timeline.qstate.state_of(SubsystemId("eps:pair:1:right"))
        == right_signal.state_ref
    )


def test_entangled_pair_source_pair_noise_reduces_bell_fidelity() -> None:
    timeline = Timeline(master_seed=1)
    source = EntangledPairSource(
        device_id="eps",
        frequency_hz=1e6,
        pair_noise_models=(two_qubit_depolarizing(1.0),),
    )
    left_sink = QuantumSink(device_id="left")
    right_sink = QuantumSink(device_id="right")
    connect_ports(source.left_output_port, left_sink.input_port, target_action="recv")
    connect_ports(source.right_output_port, right_sink.input_port, target_action="recv")
    source.bind(binding_context(timeline))

    _schedule_emit(source, timeline)

    left_signal = _received_signal(left_sink)
    right_signal = _received_signal(right_sink)
    state_ref = left_signal.state_ref
    assert state_ref is not None
    assert state_ref == right_signal.state_ref
    assert qstate_targets_from_signal(left_signal) == (SubsystemId("eps:pair:1:left"),)
    assert qstate_targets_from_signal(right_signal) == (
        SubsystemId("eps:pair:1:right"),
    )
    record = timeline.qstate.record(state_ref)
    assert record.rep == "density"
    assert isinstance(record.payload, DensityState)

    rho = record.payload.rho
    projector = _phi_plus_projector()
    fidelity = float(np.real(np.trace(projector @ rho)))
    purity = float(np.real(np.trace(rho @ rho)))

    assert rho == pytest.approx(np.eye(4, dtype=np.complex128) / 4.0, abs=ATOL)
    assert fidelity == pytest.approx(0.25, abs=ATOL)
    assert purity == pytest.approx(0.25, abs=ATOL)
    _assert_valid_density(rho)


def test_entangled_pair_source_adds_pairing_metadata() -> None:
    timeline = Timeline(master_seed=1)
    source = EntangledPairSource(device_id="eps", frequency_hz=1e6)
    left_sink = QuantumSink(device_id="left")
    right_sink = QuantumSink(device_id="right")
    connect_ports(source.left_output_port, left_sink.input_port, target_action="recv")
    connect_ports(source.right_output_port, right_sink.input_port, target_action="recv")
    source.bind(binding_context(timeline))

    _schedule_emit(source, timeline)

    left_meta = dict(_received_signal(left_sink).meta)
    right_meta = dict(_received_signal(right_sink).meta)

    assert left_meta["bsa_pair_id"] == 1
    assert right_meta["bsa_pair_id"] == 1
    assert left_meta["member"] == "left"
    assert right_meta["member"] == "right"


def test_entangled_pair_source_stores_pair_preparation_report() -> None:
    timeline = Timeline(master_seed=1)
    source = EntangledPairSource(device_id="eps", frequency_hz=1e6)
    left_sink = QuantumSink(device_id="left")
    right_sink = QuantumSink(device_id="right")
    connect_ports(source.left_output_port, left_sink.input_port, target_action="recv")
    connect_ports(source.right_output_port, right_sink.input_port, target_action="recv")
    source.bind(binding_context(timeline))

    _schedule_emit(source, timeline)

    assert len(source.reports) == 1
    report = source.reports[0]
    assert isinstance(report, SourcePreparationReport)
    assert report.report_id == "eps:prep:1"
    assert report.device_id == "eps"
    assert report.time == 0
    assert report.attempt_index == 1
    assert report.emission_index == 1
    assert report.signal_ids == ("eps:pair:1:left", "eps:pair:1:right")
    assert report.sampler_index == 0
    assert report.sampler_label == "phi+"
    assert report.state_targets == (
        SubsystemId("eps:pair:1:left"),
        SubsystemId("eps:pair:1:right"),
    )
    assert report.emission_slot_tick == 0
    assert report.emission_delay_ticks == 0
