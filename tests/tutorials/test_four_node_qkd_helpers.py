from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from examples.postprocessing.bb84_event.helpers import random_bits, toeplitz_hash
from examples.postprocessing.bb84_event.messages import encode_body
from simyuj.control.payloads import AgentMessage
from simyuj.engine import Timeline
from simyuj.primitives.messages import ClassicalMessage
from tutorials.four_node_qkd.configs import E91PostProcessingConfig
from tutorials.four_node_qkd.e91_agents import (
    E91BAgent,
    E91DetectionRecord,
    _decode_manifest,
)
from tutorials.four_node_qkd.helpers import (
    bit_from_outcome,
    chsh_value,
    conservative_chsh_value,
    e91_privacy_budget,
    observed_correlation,
    pair_index_from_signal_id,
    photon_basis_from_e91_angle,
    singlet_corrected_bit,
    validate_e91_config,
)


def _message(message_type: str, body: dict[str, object]) -> AgentMessage:
    return AgentMessage(
        message=ClassicalMessage(
            sender_id="d_e91_agent",
            receiver_id="b_e91_agent",
            body=encode_body(body),
            message_type=message_type,
            message_id=f"test-{message_type}",
        ),
        receive_time=0,
    )


def test_e91_basis_pair_and_singlet_helpers() -> None:
    basis = photon_basis_from_e91_angle("B1", math.pi / 4)

    assert basis.name == "b1"
    assert np.isclose(np.vdot(basis.vectors[0], basis.vectors[1]), 0.0)
    assert pair_index_from_signal_id("c_e91_source:pair:217:left") == 217
    assert bit_from_outcome("0") == 0
    assert singlet_corrected_bit("0") == 1
    assert singlet_corrected_bit("1") == 0

    with pytest.raises(ValueError, match="unexpected E91 signal id"):
        pair_index_from_signal_id("not-a-pair")
    with pytest.raises(ValueError, match="unexpected E91 outcome"):
        bit_from_outcome("+")


def test_chsh_sign_convention_reaches_ideal_singlet_value() -> None:
    value = 1 / math.sqrt(2)
    correlations = {
        "E_B0_D0": -value,
        "E_B0_D2": value,
        "E_B2_D0": -value,
        "E_B2_D2": -value,
    }

    assert chsh_value(correlations) == pytest.approx(-2 * math.sqrt(2))
    assert conservative_chsh_value(-2 * math.sqrt(2), 0.05) == pytest.approx(
        2 * math.sqrt(2) - 0.05
    )
    assert observed_correlation([("0", "1"), ("1", "0")]) == -1.0


def test_e91_privacy_budget_exposes_each_subtraction() -> None:
    budget = e91_privacy_budget(
        reconciled_length=1_000,
        s_lower=2.75,
        estimated_qber=0.02,
        qber_safety_margin=0.02,
        cascade_disclosures=120,
        verification_tag_len=64,
        security_margin_bits=64,
    )

    assert budget.qber_upper == pytest.approx(0.04)
    assert budget.error_correction_leak_bits >= budget.cascade_disclosures
    assert budget.final_key_length > 0
    assert budget.final_key_length == (
        budget.reconciled_length
        - budget.eve_information_bits
        - budget.error_correction_leak_bits
        - budget.verification_leak_bits
        - budget.security_margin_bits
    )

    with pytest.raises(ValueError, match="CHSH violation"):
        e91_privacy_budget(
            reconciled_length=100,
            s_lower=2.0,
            estimated_qber=0.0,
            qber_safety_margin=0.0,
            cascade_disclosures=0,
            verification_tag_len=8,
            security_margin_bits=8,
        )


def test_e91_configuration_and_manifest_validation() -> None:
    validate_e91_config(E91PostProcessingConfig())

    with pytest.raises(ValueError, match="duplicate pair index"):
        _decode_manifest(
            {
                "detections": [
                    {"pair_index": 1, "basis": "D0"},
                    {"pair_index": 1, "basis": "D1"},
                ]
            },
            field_name="detections",
            allowed_bases={"D0", "D1", "D2"},
        )
    with pytest.raises(ValueError, match="invalid E91 basis"):
        _decode_manifest(
            {"detections": [{"pair_index": 1, "basis": "X"}]},
            field_name="detections",
            allowed_bases={"D0", "D1", "D2"},
        )


def test_b_agent_high_qber_and_verification_failure_abort(monkeypatch) -> None:
    sent: list[tuple[str, dict[str, object]]] = []

    def record_send(_agent, _ctx, *, message_type, body, **_kwargs):
        sent.append((message_type, body))
        return None

    monkeypatch.setattr(
        "tutorials.four_node_qkd.e91_agents.send_json_message",
        record_send,
    )
    agent = E91BAgent(
        agent_id="b_e91_agent",
        node_id="B",
        postprocessing=E91PostProcessingConfig(),
        readiness_guard_ticks=0,
    )
    agent.key_bits = [0] * 100
    ctx = SimpleNamespace(timeline=Timeline(master_seed=7), session_id="test")
    agent._on_estimate_sample(
        _message(
            "e91.estimate.sample",
            {
                "sample_positions": list(range(20)),
                "sample_bits": [1] * 20,
            },
        ),
        ctx,
    )

    assert agent.estimated_qber == 1.0
    assert agent.qber_accepted is False
    assert agent.aborted_reason == "estimated E91 QBER exceeds abort threshold"
    assert sent[-1][0] == "e91.estimate.result"
    assert sent[-1][1]["accept"] is False

    agent.reconciled_bits = [0, 1, 1, 0]
    seed = random_bits(Timeline(master_seed=8).rng("test", "seed"), 7)
    tag = toeplitz_hash(agent.reconciled_bits, seed, 4)
    tag[0] ^= 1
    agent._on_verify_tag(
        _message(
            "e91.verify.tag",
            {
                "input_len": 4,
                "tag_len": 4,
                "tag_seed": seed,
                "tag": tag,
            },
        ),
        ctx,
    )

    assert agent.verification_complete
    assert agent.verification_accepted is False
    assert agent.aborted_reason == "E91 verification failed"
    assert sent[-1][0] == "e91.verify.result"


def test_b_agent_rejects_inconsistent_accepted_category() -> None:
    agent = E91BAgent(
        agent_id="b_e91_agent",
        node_id="B",
        postprocessing=E91PostProcessingConfig(),
        readiness_guard_ticks=0,
    )
    agent.d_manifest_basis = {1: "D0"}
    agent.detections_by_pair = {
        1: E91DetectionRecord(
            pair_index=1,
            signal_id="source:pair:1:left",
            time=0,
            basis="B0",
            outcome="0",
            report_id="report-1",
            detector_id="D0",
            flags=(),
        )
    }
    message = _message(
        "e91.sift.accepted",
        {
            "key_pair_indices": [1],
            "bell_categories": {
                "E_B0_D0": [],
                "E_B0_D2": [],
                "E_B2_D0": [],
                "E_B2_D2": [],
            },
        },
    )

    with pytest.raises(ValueError, match="inconsistent settings"):
        agent._on_sift_accepted(message, SimpleNamespace())
