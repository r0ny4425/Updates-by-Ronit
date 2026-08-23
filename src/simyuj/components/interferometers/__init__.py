"""Interferometer components.

This package holds components that recombine optical amplitudes. They are
timeline event targets; the arithmetic they call lives in
``components/coherent_optics.py`` and takes no timeline.

``DelayInterferometer`` is the only member today, which is why its event actions
are defined in its own module rather than in a shared constants module the way
``detectors/primitives/actions.py`` serves several detector components.
"""

from __future__ import annotations

from .delay_interferometer import (
    ACTION_FLUSH_DELAY_ARM,
    ACTION_INTERFERE,
    ACTION_RESOLVE_BS2,
    PORT_OUT_0,
    PORT_OUT_1,
    ArmContribution,
    DelayArmFlush,
    DelayInterferometer,
    HeldLongArm,
    InterferenceReport,
    PendingCombination,
    vacuum_like,
)

__all__ = [
    "ACTION_FLUSH_DELAY_ARM",
    "ACTION_INTERFERE",
    "ACTION_RESOLVE_BS2",
    "PORT_OUT_0",
    "PORT_OUT_1",
    "ArmContribution",
    "DelayArmFlush",
    "DelayInterferometer",
    "HeldLongArm",
    "InterferenceReport",
    "PendingCombination",
    "vacuum_like",
]
