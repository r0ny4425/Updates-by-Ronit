"""Source component exports.

The source package contains event-driven components that create qstate-backed
quantum signals and inject them into the component port graph. Shared timing
profiles and emission actions are exported here for source configuration.

This package-level surface is for concrete source models, public source
actions, timing profiles, and preparation reports. Internal scheduling and
validation helpers stay in ``_common``.
"""

from __future__ import annotations

from .entangled_pair_source import EntangledPairSource
from .reports import SourcePreparationReport
from .single_photon_source import (
    ACTION_EMIT,
    ACTION_START,
    DeltaTiming,
    EmissionTimingProfile,
    ExGaussianTiming,
    GaussianTiming,
    SinglePhotonSource,
)

__all__ = [
    "ACTION_EMIT",
    "ACTION_START",
    "DeltaTiming",
    "EmissionTimingProfile",
    "EntangledPairSource",
    "ExGaussianTiming",
    "GaussianTiming",
    "SinglePhotonSource",
    "SourcePreparationReport",
]
