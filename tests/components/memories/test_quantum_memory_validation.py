"""Public-boundary validation tests for QuantumMemory."""

from __future__ import annotations

from typing import Any, cast

import pytest

from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MemoryAbsorbRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    QuantumMemory,
)
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import EncodingScheme, Signal, SignalKind
from tests.components.memories._quantum_memory_support import _event
from tests.support.binding import binding_context


def test_quantum_memory_rejects_unknown_actions() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    with pytest.raises(ValueError, match="unsupported event action"):
        memory.handle_event(_event(memory, "not_memory"), timeline)


def test_quantum_memory_requires_bind_before_event_handling() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)

    with pytest.raises(RuntimeError, match="must be bound"):
        memory.handle_event(_event(memory, MEMORY_ABSORB), timeline)


def test_quantum_memory_rejects_events_for_other_targets() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    other = QuantumMemory(memory_id="nodeA.mem1", num_positions=1)
    memory.bind(binding_context(timeline))

    event = Event(
        time=0,
        target_ref=other,
        action=MEMORY_ABSORB,
        payload_ref=None,
    )

    with pytest.raises(ValueError, match="target_ref"):
        memory.handle_event(event, timeline)


def _polarized_pulse_signal(timeline: Timeline) -> Signal:
    """A polarized coherent pulse, exactly as WeakCoherentPulseSource builds one."""
    subsystem = SubsystemId("wcp:mode:1")
    state_ref = timeline.qstate.prepare(
        (1 + 0j, 0j),
        rep="ket",
        subsystems=(subsystem,),
    )

    return Signal(
        id="pulse-0",
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=timeline.current_time,
        origin="wcp",
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=str(subsystem),
                kind="mode",
                index=0,
                metadata=(("qstate_subsystem", str(subsystem)),),
            ),
        ),
        coherent_state=CoherentState.from_mean_photon_number(0.1),
        temporal_mode_sigma_s=1e-11,
    )


def test_quantum_memory_rejects_mode_role_signal() -> None:
    """A polarized coherent pulse must not be absorbed as a stored qubit.

    Nothing routes pulses to a memory today, but the record carries a
    ``state_ref``, so only the role distinguishes it from a carrier.
    """
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    event = Event(
        time=0,
        target_ref=memory,
        action=MEMORY_ABSORB,
        payload_ref=MemoryAbsorbRequest(
            request_id="absorb-1",
            memory_id=memory.memory_id,
            signal=_polarized_pulse_signal(timeline),
        ),
    )

    with pytest.raises(ValueError, match="kind='mode'") as excinfo:
        memory.handle_event(event, timeline)

    assert "not implemented yet" in str(excinfo.value)


def test_quantum_memory_validates_shell_config() -> None:
    with pytest.raises(ValueError, match="num_positions"):
        QuantumMemory(memory_id="nodeA.mem0", num_positions=0)

    with pytest.raises(ValueError, match="absorb_success_probability"):
        QuantumMemory(
            memory_id="nodeA.mem0",
            num_positions=1,
            absorb_success_probability=1.1,
        )

    with pytest.raises(TypeError, match="noise_models"):
        QuantumMemory(
            memory_id="nodeA.mem0",
            num_positions=1,
            noise_models=cast(Any, "bad"),
        )

    with pytest.raises(ValueError, match="recovery_ticks"):
        QuantumMemory(
            memory_id="nodeA.mem0",
            num_positions=1,
            recovery_ticks=-1,
        )

    with pytest.raises(TypeError, match="recovery_ticks"):
        QuantumMemory(
            memory_id="nodeA.mem0",
            num_positions=1,
            recovery_ticks=cast(Any, True),
        )


def test_emit_rejects_empty_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    with pytest.raises(ValueError, match="not occupied"):
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


def test_discard_rejects_empty_position() -> None:
    timeline = Timeline(master_seed=1)
    memory = QuantumMemory(memory_id="nodeA.mem0", num_positions=1)
    memory.bind(binding_context(timeline))

    with pytest.raises(ValueError, match="not occupied"):
        memory.handle_event(
            Event(
                time=0,
                target_ref=memory,
                action=MEMORY_DISCARD,
                payload_ref=MemoryDiscardRequest(
                    request_id="discard-1",
                    memory_id="nodeA.mem0",
                    position=0,
                ),
            ),
            timeline,
        )
