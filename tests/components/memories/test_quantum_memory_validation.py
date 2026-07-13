"""Public-boundary validation tests for QuantumMemory."""

from __future__ import annotations

from typing import Any, cast

import pytest

from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    QuantumMemory,
)
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
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
