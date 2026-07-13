"""Detector components and primitives.

This package exposes event-driven detector components, single-channel detector
models, readout and measurement primitives, click-resolution helpers, and
immutable report records. Component classes remain timeline event targets;
primitive records and helpers do not schedule events by themselves.
"""

from __future__ import annotations

from .bell_analyzer import (
    BellStateAnalyzer,
    BSMDecision,
    BSMModel,
    BufferedBellInput,
    CoincidenceTimeout,
    bsm_model_from_spec,
)
from .detector_array import DetectorArray
from .primitives.actions import (
    ACTION_COINCIDENCE_TIMEOUT,
    ACTION_DARK_CANDIDATE,
    ACTION_DETECT_SIGNAL,
    ACTION_RUN_BELL_ANALYSIS,
    ACTION_RUN_QUBIT_READOUT,
)
from .primitives.click import ClickPattern, resolve_click_pattern
from .primitives.dark_counts import DarkCountProcess, OnArrivalWindowDarkCounts
from .primitives.gate import (
    AlwaysOpenGate,
    GateModel,
    GateWindow,
    PeriodicGate,
    ScheduledGate,
)
from .primitives.measurement import Measure, MeasurementCall, MeasurementContext
from .primitives.params import SinglePhotonDetectorParams
from .primitives.reports import (
    FLAG_AFTERPULSE,
    FLAG_DARK_COUNT,
    FLAG_DEAD_TIME_BLOCKED,
    FLAG_DOUBLE_CLICK,
    FLAG_INVALID_PAYLOAD,
    FLAG_NO_CLICK,
    FLAG_NO_OUTCOME,
    FLAG_OUTSIDE_GATE,
    FLAG_SIGNAL_CLICK,
    FLAG_TIMEOUT,
    DetectionReport,
    RawClick,
)
from .primitives.rng import DetectorRNGStreams
from .qubit_readout import (
    ConfusionMapQubitReadout,
    IdentityQubitReadout,
    QubitReadoutDevice,
    QubitReadoutJob,
    QubitReadoutModel,
    qubit_readout_model_from_spec,
)
from .single_photon import SinglePhotonDetector

__all__ = [
    "ACTION_COINCIDENCE_TIMEOUT",
    "ACTION_DARK_CANDIDATE",
    "ACTION_DETECT_SIGNAL",
    "ACTION_RUN_BELL_ANALYSIS",
    "ACTION_RUN_QUBIT_READOUT",
    "AlwaysOpenGate",
    "BSMDecision",
    "BSMModel",
    "BellStateAnalyzer",
    "BufferedBellInput",
    "ClickPattern",
    "CoincidenceTimeout",
    "ConfusionMapQubitReadout",
    "DarkCountProcess",
    "DetectionReport",
    "DetectorArray",
    "DetectorRNGStreams",
    "FLAG_AFTERPULSE",
    "FLAG_DARK_COUNT",
    "FLAG_DEAD_TIME_BLOCKED",
    "FLAG_DOUBLE_CLICK",
    "FLAG_INVALID_PAYLOAD",
    "FLAG_NO_CLICK",
    "FLAG_NO_OUTCOME",
    "FLAG_OUTSIDE_GATE",
    "FLAG_SIGNAL_CLICK",
    "FLAG_TIMEOUT",
    "GateModel",
    "GateWindow",
    "IdentityQubitReadout",
    "Measure",
    "MeasurementCall",
    "MeasurementContext",
    "OnArrivalWindowDarkCounts",
    "PeriodicGate",
    "QubitReadoutDevice",
    "QubitReadoutJob",
    "QubitReadoutModel",
    "RawClick",
    "ScheduledGate",
    "SinglePhotonDetector",
    "SinglePhotonDetectorParams",
    "bsm_model_from_spec",
    "qubit_readout_model_from_spec",
    "resolve_click_pattern",
]
