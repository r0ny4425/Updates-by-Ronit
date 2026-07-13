"""Example pieces for the single-photon BB84 tutorial pipeline."""

from .agents import (
    AlicePreparationRecord,
    BB84AliceAgent,
    BB84BobAgent,
    BobDetectionRecord,
)
from .configs import (
    BB84AliceSourceConfig,
    BB84BobDetectorConfig,
    BB84CascadeConfig,
    BB84ClassicalChannelConfig,
    BB84PrivacyAmplificationConfig,
    BB84QBERConfig,
    BB84QuantumChannelConfig,
    BB84VerificationConfig,
)
from .helpers import (
    BB84_LABELS,
    BB84_STATES,
    assign_detection_slot_index,
    bb84_basis_bit,
    bb84_bob_sifting_guard_ticks,
    bb84_final_key_length,
    bb84_slot_assignment_window_ticks,
    bb84_slot_period_ticks,
    bb84_slot_zero_arrival_tick,
    bb84_source_done_delay_ticks,
    bob_report_basis_bit,
    report_detection_time,
    send_json_message,
)
from .reporting import SUMMARY_KEYS, summarize_trial, write_trial_report
from .trial import run_bb84_trial

__all__ = [
    "AlicePreparationRecord",
    "BB84AliceSourceConfig",
    "BB84AliceAgent",
    "BB84BobAgent",
    "BB84BobDetectorConfig",
    "BB84CascadeConfig",
    "BB84ClassicalChannelConfig",
    "BB84_LABELS",
    "BB84PrivacyAmplificationConfig",
    "BB84QBERConfig",
    "BB84QuantumChannelConfig",
    "BB84_STATES",
    "BB84VerificationConfig",
    "assign_detection_slot_index",
    "bb84_basis_bit",
    "bb84_bob_sifting_guard_ticks",
    "bb84_final_key_length",
    "bb84_slot_assignment_window_ticks",
    "bb84_slot_period_ticks",
    "bb84_slot_zero_arrival_tick",
    "bb84_source_done_delay_ticks",
    "bob_report_basis_bit",
    "BobDetectionRecord",
    "report_detection_time",
    "run_bb84_trial",
    "send_json_message",
    "SUMMARY_KEYS",
    "summarize_trial",
    "write_trial_report",
]
