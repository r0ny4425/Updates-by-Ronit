"""Memory-position storage-noise normalization.

This module converts public storage-noise configuration into one ordered
``NoiseModel`` chain per memory position. ``QuantumMemory`` applies the chain
for elapsed storage time immediately before operations that touch occupied
positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

from simyuj.qstate.noise import NoiseChannel, NoiseModel

MemoryNoiseModels: TypeAlias = tuple[tuple[NoiseModel, ...], ...]
MemoryNoiseModelsInput: TypeAlias = (
    NoiseModel | Sequence[NoiseModel | Sequence[NoiseModel] | None] | None
)


def normalize_memory_noise_models(
    noise_models: MemoryNoiseModelsInput,
    *,
    num_positions: int,
) -> MemoryNoiseModels:
    """
    Normalize memory storage noise to one ordered chain per position.

    Parameters
    ----------
    noise_models : MemoryNoiseModelsInput
        ``None`` for no noise, one noise model shared by all positions, or a
        position-indexed sequence of noise models, sequences, or ``None``. A
        top-level sequence is always interpreted by position index, not as one
        shared chain.
    num_positions : int
        Positive memory position count.

    Returns
    -------
    MemoryNoiseModels
        Tuple indexed by physical position. Each item is the ordered noise chain
        for that position.

    Notes
    -----
    A noise model is accepted structurally as ``NoiseChannel`` or any object
    exposing callable ``resolve``. Strings and bytes are rejected as accidental
    sequences.
    """
    _check_num_positions(num_positions)

    if noise_models is None:
        return tuple(() for _position in range(num_positions))

    if _is_noise_model(noise_models):
        model = cast(NoiseModel, noise_models)
        return tuple((model,) for _position in range(num_positions))

    if _is_bad_sequence(noise_models) or not isinstance(noise_models, Sequence):
        raise TypeError(
            "noise_models must be a NoiseModel, a position-indexed sequence, or None"
        )

    position_entries = cast(
        Sequence[NoiseModel | Sequence[NoiseModel] | None],
        noise_models,
    )

    if len(position_entries) != num_positions:
        raise ValueError("noise_models top-level sequence must match num_positions")

    return tuple(
        _normalize_position_noise(entry, position=position)
        for position, entry in enumerate(position_entries)
    )


def _normalize_position_noise(
    entry: NoiseModel | Sequence[NoiseModel] | None,
    *,
    position: int,
) -> tuple[NoiseModel, ...]:
    if entry is None:
        return ()

    if _is_noise_model(entry):
        return (cast(NoiseModel, entry),)

    if _is_bad_sequence(entry) or not isinstance(entry, Sequence):
        raise TypeError(
            f"noise_models[{position}] must be NoiseModel, sequence, or None"
        )

    chain: list[NoiseModel] = []
    for index, model in enumerate(entry):
        if not _is_noise_model(model):
            raise TypeError(f"noise_models[{position}][{index}] must be NoiseModel")
        chain.append(cast(NoiseModel, model))
    return tuple(chain)


def _is_noise_model(value: object) -> bool:
    if isinstance(value, NoiseChannel):
        return True
    return callable(getattr(value, "resolve", None))


def _is_bad_sequence(value: object) -> bool:
    return isinstance(value, (str, bytes))


def _check_num_positions(num_positions: object) -> None:
    if type(num_positions) is not int:
        raise TypeError("num_positions must be int")
    if num_positions <= 0:
        raise ValueError("num_positions must be positive")


__all__ = [
    "MemoryNoiseModels",
    "MemoryNoiseModelsInput",
    "normalize_memory_noise_models",
]
