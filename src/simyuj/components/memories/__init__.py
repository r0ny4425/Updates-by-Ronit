"""Quantum memory component package.

This package exposes the event-driven ``QuantumMemory`` component, memory
position records, operation request payloads, report payloads, and storage-noise
normalization helpers. Memory components interact with the timeline through
explicit events and with qstate through stable per-position subsystem labels.
"""

from __future__ import annotations

from .noise import (
    MemoryNoiseModels,
    MemoryNoiseModelsInput,
    normalize_memory_noise_models,
)
from .position import (
    MemoryPositionRecord,
    MemoryPositionStatus,
    emitted_photon_subsystem_id,
    memory_subsystem_id,
)
from .quantum_memory import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_EXPIRE,
    MEMORY_MEASURE,
    MEMORY_UPDATE_META,
    QuantumMemory,
)
from .reports import (
    MemoryAbsorbReport,
    MemoryDiscardReport,
    MemoryEmitReport,
    MemoryExpireReport,
    MemoryMeasurementReport,
    MemoryMetaUpdateReport,
    MemoryOperatorReport,
    MemoryReport,
)
from .requests import (
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryExpireRequest,
    MemoryMeasureRequest,
    MemoryUpdateMetaRequest,
)

__all__ = [
    "MEMORY_ABSORB",
    "MEMORY_APPLY_OPERATOR",
    "MEMORY_DISCARD",
    "MEMORY_EMIT",
    "MEMORY_EXPIRE",
    "MEMORY_MEASURE",
    "MEMORY_UPDATE_META",
    "MemoryNoiseModels",
    "MemoryNoiseModelsInput",
    "MemoryAbsorbReport",
    "MemoryAbsorbRequest",
    "MemoryDiscardReport",
    "MemoryApplyOperatorRequest",
    "MemoryDiscardRequest",
    "MemoryEmitReport",
    "MemoryEmitRequest",
    "MemoryExpireReport",
    "MemoryExpireRequest",
    "MemoryMetaUpdateReport",
    "MemoryMeasureRequest",
    "MemoryMeasurementReport",
    "MemoryOperatorReport",
    "MemoryPositionRecord",
    "MemoryPositionStatus",
    "MemoryReport",
    "MemoryUpdateMetaRequest",
    "QuantumMemory",
    "emitted_photon_subsystem_id",
    "memory_subsystem_id",
    "normalize_memory_noise_models",
]
