"""Pure helpers for the single-photon BB84 example."""

from __future__ import annotations

import math
from typing import Any

from examples.postprocessing.bb84_event.helpers import binary_entropy
from examples.postprocessing.bb84_event.messages import encode_body
from simyuj.primitives.messages import ClassicalMessage
from simyuj.primitives.units import seconds_to_ticks

BB84_STATES = ("|0>", "|1>", "|+>", "|->")
BB84_LABELS = ("Z0", "Z1", "X0", "X1")


def bb84_basis_bit(label: str) -> tuple[str, int]:
    if label not in BB84_LABELS:
        raise ValueError(f"unknown BB84 label: {label!r}")
    return label[0], int(label[1])


def bob_report_basis_bit(report: Any) -> tuple[str, int] | None:
    if not report.success:
        return None

    basis = report.measurement_label
    outcome = report.outcome

    if basis == "Z" and outcome == "0":
        return "Z", 0
    if basis == "Z" and outcome == "1":
        return "Z", 1
    if basis == "X" and outcome == "+":
        return "X", 0
    if basis == "X" and outcome == "-":
        return "X", 1

    raise ValueError(
        f"unexpected Bob report outcome: basis={basis!r}, outcome={outcome!r}"
    )


def send_json_message(
    agent: Any,
    ctx: Any,
    *,
    receiver_id: str,
    out_port: str,
    message_type: str,
    body: dict[str, Any],
) -> Any:
    message = ClassicalMessage(
        sender_id=agent.agent_id,
        receiver_id=receiver_id,
        body=encode_body(body),
        sent_time=ctx.timeline.current_time,
        session_id=ctx.session_id,
        message_type=message_type,
        message_id=f"{agent.agent_id}-{agent.message_counter}",
    )
    agent.message_counter += 1
    return agent.classical.send(
        message,
        ctx.timeline,
        port_name=out_port,
    )


def bb84_final_key_length(
    *,
    reconciled_len: int,
    estimated_qber: float,
    revealed_bits: int,
    qber_safety_margin: float,
    security_margin_bits: int,
) -> int:
    qber_for_pa = min(0.5, max(0.0, estimated_qber + qber_safety_margin))
    phase_error_cost = math.ceil(reconciled_len * binary_entropy(qber_for_pa))
    final_len = reconciled_len - phase_error_cost - revealed_bits - security_margin_bits
    return max(0, final_len)


def bb84_source_done_delay_ticks(source_config: Any) -> int:
    frame_duration_s = source_config.num_slots / source_config.clock_hz
    source_margin_s = source_config.max_timing_jitter_s
    return seconds_to_ticks(frame_duration_s + source_margin_s)


def bb84_bob_sifting_guard_ticks(detector_config: Any) -> int:
    guard_s = (
        6 * detector_config.jitter_stddev_s
        + detector_config.detection_window_s
        + detector_config.output_latency_s
    )
    return seconds_to_ticks(guard_s)


def bb84_slot_period_ticks(source_config: Any) -> int:
    return seconds_to_ticks(1.0 / source_config.clock_hz)


def bb84_slot_zero_arrival_tick(quantum_config: Any) -> int:
    return seconds_to_ticks(
        quantum_config.length_m / quantum_config.propagation_speed_m_per_s
    )


def bb84_slot_assignment_window_ticks(
    source_config: Any,
    quantum_config: Any,
    detector_config: Any,
) -> int:
    window_s = (
        source_config.max_timing_jitter_s
        + 6 * quantum_config.timing_jitter_stddev_s
        + 6 * detector_config.jitter_stddev_s
        + detector_config.detection_window_s
    )
    return seconds_to_ticks(window_s)


def assign_detection_slot_index(
    detection_time: int,
    *,
    slot_period_ticks: int,
    slot_zero_arrival_tick: int,
    slot_assignment_window_ticks: int,
    num_slots: int,
) -> int | None:
    if slot_period_ticks <= 0:
        raise ValueError("slot_period_ticks must be positive")

    relative_tick = detection_time - slot_zero_arrival_tick
    slot_offset = round(relative_tick / slot_period_ticks)
    slot_index = slot_offset + 1

    if slot_index < 1 or slot_index > num_slots:
        return None

    expected_tick = slot_zero_arrival_tick + slot_offset * slot_period_ticks
    if abs(detection_time - expected_tick) > slot_assignment_window_ticks:
        return None

    return slot_index


def report_detection_time(report: Any) -> int:
    if report.raw_clicks:
        return min(click.time for click in report.raw_clicks)
    return report.time


__all__ = [
    "BB84_LABELS",
    "BB84_STATES",
    "assign_detection_slot_index",
    "bb84_basis_bit",
    "bb84_bob_sifting_guard_ticks",
    "bb84_final_key_length",
    "bb84_slot_assignment_window_ticks",
    "bb84_slot_period_ticks",
    "bb84_slot_zero_arrival_tick",
    "bb84_source_done_delay_ticks",
    "bob_report_basis_bit",
    "report_detection_time",
    "send_json_message",
]
