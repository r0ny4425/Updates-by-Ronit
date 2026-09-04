"""Public component package exports.

This package collects the simulator's event-driven component surfaces:
channels, connections, ports, memories, and sources. Concrete components remain
timeline event targets; ports and connection records provide the routing
structure used to schedule cross-component delivery events.
"""

from __future__ import annotations

from .channels import (
    ACTION_RECEIVE_CLASSICAL,
    ACTION_TRANSMIT_CLASSICAL,
    ACTION_TRANSMIT_QUANTUM,
    DEFAULT_FIBER_LIGHT_SPEED_M_PER_S,
    ClassicalChannel,
    QuantumChannel,
)
from .connections import (
    PortConnection,
    PortDelivery,
    connect_ports,
    default_connection_id,
    require_connection,
)
from .interferometers.delay_interferometer import (
    ACTION_FLUSH_DELAY_ARM,
    ACTION_INTERFERE,
    ACTION_RESOLVE_BS2,
    DelayInterferometer,
    InterferenceReport,
)
from .memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_EXPIRE,
    MEMORY_MEASURE,
    QuantumMemory,
)
from .ports import Port, PortDirection, PortKind
from .sources.coherent_preparation import (
    DPS_PHASES,
    CarrierPhaseSelector,
    EncodingPhaseSelector,
    FixedCarrierPhase,
    FixedIntensity,
    FixedPhase,
    IntensitySelection,
    IntensitySelector,
    PerPulseRandomCarrierPhase,
    PhaseSelection,
    PhaseSequence,
    PolarizationSelection,
    PolarizationSelector,
    RandomPhaseChoice,
)
from .sources.reports import CoherentPulsePreparationReport, SourcePreparationReport
from .sources.single_photon_source import (
    ACTION_EMIT,
    ACTION_START,
    DeltaTiming,
    EmissionTimingProfile,
    ExGaussianTiming,
    GaussianTiming,
    SinglePhotonSource,
)
from .sources.weak_coherent_pulse_source import WeakCoherentPulseSource

__all__ = [
    "ACTION_FLUSH_DELAY_ARM",
    "ACTION_INTERFERE",
    "ACTION_RESOLVE_BS2",
    "DelayInterferometer",
    "InterferenceReport",
    "ACTION_EMIT",
    "ACTION_RECEIVE_CLASSICAL",
    "ACTION_START",
    "ACTION_TRANSMIT_CLASSICAL",
    "ACTION_TRANSMIT_QUANTUM",
    "CarrierPhaseSelector",
    "ClassicalChannel",
    "CoherentPulsePreparationReport",
    "DEFAULT_FIBER_LIGHT_SPEED_M_PER_S",
    "DPS_PHASES",
    "DeltaTiming",
    "EmissionTimingProfile",
    "EncodingPhaseSelector",
    "ExGaussianTiming",
    "FixedCarrierPhase",
    "FixedIntensity",
    "FixedPhase",
    "GaussianTiming",
    "IntensitySelection",
    "IntensitySelector",
    "MEMORY_ABSORB",
    "MEMORY_APPLY_OPERATOR",
    "MEMORY_DISCARD",
    "MEMORY_EMIT",
    "MEMORY_EXPIRE",
    "MEMORY_MEASURE",
    "PerPulseRandomCarrierPhase",
    "PhaseSelection",
    "PhaseSequence",
    "PolarizationSelection",
    "PolarizationSelector",
    "Port",
    "PortConnection",
    "PortDelivery",
    "PortDirection",
    "PortKind",
    "QuantumMemory",
    "QuantumChannel",
    "RandomPhaseChoice",
    "SinglePhotonSource",
    "SourcePreparationReport",
    "WeakCoherentPulseSource",
    "connect_ports",
    "default_connection_id",
    "require_connection",
]
