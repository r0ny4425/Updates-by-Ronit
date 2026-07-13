"""QuantumMemory storage-noise timing and ordering tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from simyuj.components import connect_ports
from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_EMIT,
    MEMORY_MEASURE,
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryEmitRequest,
    MemoryMeasureRequest,
    QuantumMemory,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.units import ticks_to_seconds
from simyuj.qstate import DensityState
from simyuj.qstate.noise import T1T2Noise, TimeDependentNoiseModel
from simyuj.qstate.ops import X
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.components.memories._quantum_memory_support import (
    QuantumSink,
    RecordingNoiseModel,
    _absorb_position,
    _signal,
)
from tests.support.binding import binding_context

ATOL = 1e-12
PSD_ATOL = 1e-10


def _assert_valid_density(rho: np.ndarray) -> None:
    assert rho == pytest.approx(rho.conj().T, abs=ATOL)
    assert np.trace(rho) == pytest.approx(1.0, abs=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) >= -PSD_ATOL


def test_quantum_memory_logs_storage_noise_at_trace() -> None:
    log_sink = MemorySink()
    noise_model = RecordingNoiseModel()
    timeline = Timeline(logger=SimulationLogger(level=LogLevel.TRACE, sinks=[log_sink]))
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        noise_models=noise_model,
    )
    memory.bind(binding_context(timeline))
    _absorb_position(memory, timeline, position=0)

    timeline.schedule(
        Event(
            time=5,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id=memory.memory_id,
                positions=(0,),
                destructive=False,
            ),
        )
    )
    timeline.run_until(5)

    record = next(
        record
        for record in log_sink.records
        if record.category == "components.memories.quantum_memory.storage_noise"
    )

    assert record.level is LogLevel.TRACE
    assert dict(record.meta) == {
        "memory_id": "nodeA.mem0",
        "position": 0,
        "memory_subsystem": str(memory_subsystem_id("nodeA.mem0", 0)),
        "duration_ticks": 5,
        "noise_model_count": 1,
    }


def test_emit_failure_settles_storage_noise_without_double_applying() -> None:
    timeline = Timeline(master_seed=1)
    noise_model = RecordingNoiseModel()
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=5,
        emit_success_probability=0.0,
        noise_models=noise_model,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    _absorb_position(memory, timeline)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    timeline.run_until(5)

    assert noise_model.durations == [ticks_to_seconds(5)]
    assert memory.positions[0].last_noise_update_time == 5

    memory.emit_success_probability = 1.0
    memory.handle_event(
        Event(
            time=timeline.current_time,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-2",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )
    timeline.run_until(10)

    assert noise_model.durations == [ticks_to_seconds(5), ticks_to_seconds(5)]
    assert len(sink.received) == 1


def test_real_t1t2_storage_noise_decoheres_emitted_memory_state() -> None:
    T1 = 8.0e-12
    T2 = 8.0e-12
    storage_ticks = 4
    duration_s = ticks_to_seconds(storage_ticks)
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        emit_delay_ticks=0,
        noise_models=cast(TimeDependentNoiseModel, T1T2Noise(T1=T1, T2=T2)),
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    _absorb_position(memory, timeline, state="|+>")

    timeline.schedule(
        Event(
            time=storage_ticks,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id=memory.memory_id,
                position=0,
            ),
        )
    )
    timeline.run_until(storage_ticks)

    assert len(sink.received) == 1
    signal = sink.received[0].payload
    assert isinstance(signal, Signal)
    emitted_target = qstate_targets_from_signal(signal)[0]
    state_ref = timeline.qstate.state_of(emitted_target)
    record = timeline.qstate.record(state_ref)
    assert record.rep == "density"
    assert isinstance(record.payload, DensityState)

    rho = record.payload.rho
    gamma = -np.expm1(-duration_s / T1)
    coherence = np.exp(-duration_s / T2)
    expected = np.array(
        [
            [0.5 * (1.0 + gamma), 0.5 * coherence],
            [0.5 * coherence, 0.5 * (1.0 - gamma)],
        ],
        dtype=np.complex128,
    )
    initial_purity = 1.0
    observed_purity = float(np.real(np.trace(rho @ rho)))

    assert rho == pytest.approx(expected, abs=ATOL)
    assert observed_purity < initial_purity
    _assert_valid_density(rho)


def test_emit_applies_pending_storage_noise_before_relabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        noise_models=noise_model,
        recovery_ticks=9,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")
    signal = _signal(timeline)
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    output_subsystem = emitted_photon_subsystem_id("nodeA.mem0", 0, 0)

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=signal,
            ),
        ),
        timeline,
    )
    qstate_type = type(timeline.qstate)
    original_apply_noise_models = qstate_type.apply_noise_models
    original_relabel_subsystem = qstate_type.relabel_subsystem
    calls: list[tuple[str, object]] = []

    def record_apply_noise_models(self, noise_models, **kwargs):
        calls.append(("noise", kwargs["targets"]))
        assert self.store.contains_subsystem(memory_subsystem)
        assert not self.store.contains_subsystem(output_subsystem)
        return original_apply_noise_models(self, noise_models, **kwargs)

    def record_relabel_subsystem(self, old, new):
        calls.append(("relabel", (old, new)))
        return original_relabel_subsystem(self, old, new)

    monkeypatch.setattr(qstate_type, "apply_noise_models", record_apply_noise_models)
    monkeypatch.setattr(qstate_type, "relabel_subsystem", record_relabel_subsystem)
    timeline.schedule(
        Event(
            time=3,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        )
    )
    timeline.run_until(3)

    assert noise_model.durations == [pytest.approx(3.0e-12)]
    assert calls == [
        ("noise", (memory_subsystem,)),
        ("relabel", (memory_subsystem, output_subsystem)),
    ]


def test_lazy_storage_noise_skips_zero_duration() -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        noise_models=noise_model,
    )
    sink = QuantumSink()
    memory.bind(binding_context(timeline))
    connect_ports(memory.output_port, sink.input_port, target_action="receive_signal")

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_EMIT,
            payload_ref=MemoryEmitRequest(
                request_id="emit-1",
                memory_id="nodeA.mem0",
                position=0,
            ),
        ),
        timeline,
    )

    assert noise_model.durations == []


def test_lazy_storage_noise_updates_timestamp_without_models() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    timeline.schedule(
        Event(
            time=3,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=X,
            ),
        )
    )
    timeline.run_until(3)

    assert memory.positions[0].last_noise_update_time == 3


def test_apply_operator_applies_pending_storage_noise_before_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        noise_models=noise_model,
    )
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    qstate_type = type(timeline.qstate)
    original_apply_noise_models = qstate_type.apply_noise_models
    original_apply = qstate_type.apply
    calls: list[tuple[str, object]] = []

    def record_apply_noise_models(self, noise_models, **kwargs):
        calls.append(("noise", kwargs["targets"]))
        assert self.store.contains_subsystem(memory_subsystem)
        return original_apply_noise_models(self, noise_models, **kwargs)

    def record_apply(self, operation, **kwargs):
        calls.append(("apply", kwargs["targets"]))
        return original_apply(self, operation, **kwargs)

    monkeypatch.setattr(qstate_type, "apply_noise_models", record_apply_noise_models)
    monkeypatch.setattr(qstate_type, "apply", record_apply)
    timeline.schedule(
        Event(
            time=4,
            target_ref=memory,
            action=MEMORY_APPLY_OPERATOR,
            payload_ref=MemoryApplyOperatorRequest(
                request_id="apply-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                operator=X,
            ),
        )
    )
    timeline.run_until(4)

    assert noise_model.durations == [pytest.approx(4.0e-12)]
    assert memory.positions[0].last_noise_update_time == 4
    assert calls == [
        ("noise", (memory_subsystem,)),
        ("apply", (memory_subsystem,)),
    ]


def test_measure_applies_pending_storage_noise_before_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise_model = RecordingNoiseModel()
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=1,
        noise_models=noise_model,
    )
    memory.bind(binding_context(timeline))

    memory.handle_event(
        Event(
            time=0,
            target_ref=memory,
            action=MEMORY_ABSORB,
            payload_ref=MemoryAbsorbRequest(
                request_id="absorb-1",
                memory_id="nodeA.mem0",
                signal=_signal(timeline),
            ),
        ),
        timeline,
    )
    memory_subsystem = memory_subsystem_id("nodeA.mem0", 0)
    qstate_type = type(timeline.qstate)
    original_apply_noise_models = qstate_type.apply_noise_models
    original_measure = qstate_type.measure
    calls: list[tuple[str, object]] = []

    def record_apply_noise_models(self, noise_models, **kwargs):
        calls.append(("noise", kwargs["targets"]))
        assert self.store.contains_subsystem(memory_subsystem)
        return original_apply_noise_models(self, noise_models, **kwargs)

    def record_measure(self, **kwargs):
        calls.append(("measure", kwargs["targets"]))
        return original_measure(self, **kwargs)

    monkeypatch.setattr(qstate_type, "apply_noise_models", record_apply_noise_models)
    monkeypatch.setattr(qstate_type, "measure", record_measure)
    timeline.schedule(
        Event(
            time=5,
            target_ref=memory,
            action=MEMORY_MEASURE,
            payload_ref=MemoryMeasureRequest(
                request_id="measure-1",
                memory_id="nodeA.mem0",
                positions=(0,),
                destructive=False,
            ),
        )
    )
    timeline.run_until(5)

    assert noise_model.durations == [pytest.approx(5.0e-12)]
    assert memory.positions[0].last_noise_update_time == 5
    assert memory.positions[0].ready_at == 0
    assert calls == [
        ("noise", (memory_subsystem,)),
        ("measure", (memory_subsystem,)),
    ]
