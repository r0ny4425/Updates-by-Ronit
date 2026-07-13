from __future__ import annotations

from typing import Any, cast

import pytest

from simyuj.components.memories import QuantumMemory, normalize_memory_noise_models
from simyuj.qstate.noise import NoiseChannel, depolarizing


class RecordingNoiseModel:
    name = "recording"
    arity = 1

    def __init__(self) -> None:
        self.durations: list[float] = []

    def resolve(self, *, duration_s: float) -> NoiseChannel:
        self.durations.append(duration_s)
        return depolarizing(0.0)


def test_none_means_no_storage_noise_per_position() -> None:
    assert normalize_memory_noise_models(None, num_positions=3) == ((), (), ())


def test_single_model_applies_to_every_position() -> None:
    model = RecordingNoiseModel()

    normalized = normalize_memory_noise_models(model, num_positions=3)

    assert normalized == ((model,), (model,), (model,))
    assert model.durations == []


def test_position_indexed_sequence_maps_to_positions() -> None:
    model0 = RecordingNoiseModel()
    model2a = RecordingNoiseModel()
    model2b = depolarizing(0.0)

    normalized = normalize_memory_noise_models(
        [model0, None, (model2a, model2b)],
        num_positions=3,
    )

    assert normalized == ((model0,), (), (model2a, model2b))
    assert model0.durations == []
    assert model2a.durations == []


def test_top_level_sequence_must_match_num_positions() -> None:
    with pytest.raises(ValueError, match="num_positions"):
        normalize_memory_noise_models([None], num_positions=2)


def test_invalid_noise_model_entry_is_rejected() -> None:
    with pytest.raises(TypeError, match="NoiseModel"):
        normalize_memory_noise_models(cast(Any, [object()]), num_positions=1)


def test_quantum_memory_stores_normalized_noise_models() -> None:
    model = RecordingNoiseModel()

    memory = QuantumMemory(
        memory_id="nodeA.mem0",
        num_positions=2,
        noise_models=model,
    )

    assert memory._position_noise_models == ((model,), (model,))
    assert model.durations == []
