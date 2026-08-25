"""Example pieces for the DPS-QKD pipeline.

Transmitter, fibre and receiver, end to end: Alice's weak coherent pulse source,
the channel's amplitude path, Bob's delay interferometer, and the two detectors
inside it that turn each output amplitude into a click. Bob's bits come from
which port fired.

What is still a later step is the *protocol*: there is no classical channel and
no agents here, so Alice's bits are read from her own reports rather than
learned from a message, and there is no sifting exchange, error correction or
privacy amplification. See ``docs/dev/dps-design.md``.
"""

from .configs import DPS_ENCODING_PHASES, DPSAliceSourceConfig, DPSDetectorConfig
from .helpers import (
    dps_detected_bit,
    dps_differential_bit,
    dps_differential_bits,
    dps_optical_differential_bits,
    dps_phase_histogram,
    dps_slot_arms,
    dps_slot_period_ticks,
    dps_source_duration_s,
)
from .reporting import (
    SUMMARY_KEYS,
    UNMODELLED_PHYSICS,
    summarize_trial,
    write_trial_report,
)
from .trial import (
    ACTION_RECEIVE_DETECTION,
    ACTION_TAP_PULSE,
    DetectionCollector,
    DPSSlotOutcome,
    PulseTap,
    read_detection_slots,
    run_dps_transmitter_trial,
)

__all__ = [
    "ACTION_RECEIVE_DETECTION",
    "ACTION_TAP_PULSE",
    "DPSAliceSourceConfig",
    "DPSDetectorConfig",
    "DPSSlotOutcome",
    "DPS_ENCODING_PHASES",
    "DetectionCollector",
    "PulseTap",
    "SUMMARY_KEYS",
    "UNMODELLED_PHYSICS",
    "dps_detected_bit",
    "dps_differential_bit",
    "dps_differential_bits",
    "dps_optical_differential_bits",
    "dps_phase_histogram",
    "dps_slot_arms",
    "dps_slot_period_ticks",
    "dps_source_duration_s",
    "read_detection_slots",
    "run_dps_transmitter_trial",
    "summarize_trial",
    "write_trial_report",
]
