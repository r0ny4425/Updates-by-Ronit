"""Example pieces for the DPS-QKD pipeline.

Stage 1 only: Alice's weak coherent pulse transmitter. The receiver -- the
channel's amplitude path, the delay interferometer, the optical detector -- and
the protocol agents are later steps in the same package. See
``docs/dev/dps-design.md``.
"""

from .configs import DPS_ENCODING_PHASES, DPSAliceSourceConfig
from .helpers import (
    dps_differential_bit,
    dps_differential_bits,
    dps_phase_histogram,
    dps_slot_period_ticks,
    dps_source_duration_s,
)
from .reporting import (
    SUMMARY_KEYS,
    UNMODELLED_PHYSICS,
    summarize_trial,
    write_trial_report,
)
from .trial import ACTION_TAP_PULSE, PulseTap, run_dps_transmitter_trial

__all__ = [
    "ACTION_TAP_PULSE",
    "DPSAliceSourceConfig",
    "DPS_ENCODING_PHASES",
    "PulseTap",
    "SUMMARY_KEYS",
    "UNMODELLED_PHYSICS",
    "dps_differential_bit",
    "dps_differential_bits",
    "dps_phase_histogram",
    "dps_slot_period_ticks",
    "dps_source_duration_s",
    "run_dps_transmitter_trial",
    "summarize_trial",
    "write_trial_report",
]
