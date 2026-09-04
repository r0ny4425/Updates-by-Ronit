"""Source component exports.

The source package contains event-driven components that create qstate-backed
quantum signals and inject them into the component port graph. Shared timing
profiles and emission actions are exported here for source configuration.

This package-level surface is for concrete source models, public source
actions, timing profiles, and preparation reports. Internal scheduling and
validation helpers stay in ``_common``.
"""

from __future__ import annotations

from .coherent_preparation import (
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
from .entangled_pair_source import EntangledPairSource
from .reports import CoherentPulsePreparationReport, SourcePreparationReport
from .single_photon_source import (
    ACTION_EMIT,
    ACTION_START,
    DeltaTiming,
    EmissionTimingProfile,
    ExGaussianTiming,
    GaussianTiming,
    SinglePhotonSource,
)
from .weak_coherent_pulse_source import WeakCoherentPulseSource

__all__ = [
    "ACTION_EMIT",
    "ACTION_START",
    "DPS_PHASES",
    "CarrierPhaseSelector",
    "CoherentPulsePreparationReport",
    "DeltaTiming",
    "EmissionTimingProfile",
    "EncodingPhaseSelector",
    "EntangledPairSource",
    "ExGaussianTiming",
    "FixedCarrierPhase",
    "FixedIntensity",
    "FixedPhase",
    "GaussianTiming",
    "IntensitySelection",
    "IntensitySelector",
    "PerPulseRandomCarrierPhase",
    "PhaseSelection",
    "PhaseSequence",
    "PolarizationSelection",
    "PolarizationSelector",
    "RandomPhaseChoice",
    "SinglePhotonSource",
    "SourcePreparationReport",
    "WeakCoherentPulseSource",
]
