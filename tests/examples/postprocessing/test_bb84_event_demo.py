from __future__ import annotations

from typing import Any

from examples.postprocessing.bb84_event import agents
from examples.postprocessing.bb84_event import demo as demo_module
from examples.postprocessing.bb84_event.helpers import toeplitz_hash

TraceEntry = tuple[int, str, str, str, dict[str, Any]]


def run_demo_with_trace(
    monkeypatch,
    *,
    inputs_factory=None,
    bob_cls=None,
) -> tuple[demo_module.DemoResult, list[TraceEntry]]:
    trace: list[TraceEntry] = []
    original_send = agents.BB84PostProcessor._send

    def traced_send(
        self: agents.BB84PostProcessor,
        ctx,
        message_type: str,
        body: dict[str, Any],
    ):
        trace.append(
            (
                ctx.timeline.current_time,
                self.agent_id,
                self.peer_id,
                message_type,
                dict(body),
            )
        )
        return original_send(self, ctx, message_type, body)

    monkeypatch.setattr(agents.BB84PostProcessor, "_send", traced_send)
    if inputs_factory is not None:
        monkeypatch.setattr(demo_module, "make_demo_inputs", inputs_factory)
    if bob_cls is not None:
        monkeypatch.setattr(demo_module, "BobPostProcessor", bob_cls)

    return demo_module.build_demo(), trace


def high_qber_inputs(
    raw_bits: int = 1024,
    *,
    error_sift_positions: tuple[int, ...] = (17,),
    missed_detection_count: int | None = None,
):
    del raw_bits, error_sift_positions, missed_detection_count
    count = 200
    alice_bases = ["Z"] * count
    bob_bases = alice_bases.copy()
    alice_bits = [index & 1 for index in range(count)]
    bob_bits = [1 - bit for bit in alice_bits]
    return alice_bases, alice_bits, bob_bases, bob_bits


def too_few_sifted_inputs(
    raw_bits: int = 1024,
    *,
    error_sift_positions: tuple[int, ...] = (17,),
    missed_detection_count: int | None = None,
):
    del raw_bits, error_sift_positions, missed_detection_count
    count = 40
    alice_bases = ["Z"] * count
    bob_bases = alice_bases.copy()
    bits = [index & 1 for index in range(count)]
    return alice_bases, bits.copy(), bob_bases, bits.copy()


def test_default_demo_completes_with_matching_keys():
    result = demo_module.build_demo()

    assert result.alice.aborted_reason is None
    assert result.bob.aborted_reason is None
    assert result.alice.complete
    assert result.bob.complete
    assert result.alice.final_key == result.bob.final_key
    assert result.alice.final_key is not None
    assert len(result.alice.final_key) == 64
    assert result.alice.reconciled_bits == result.bob.reconciled_bits
    assert result.bob.cascade_corrections == 1
    assert result.bob.cascade_leaked_bits == result.bob.cascade_parity_requests


def test_default_demo_message_order(monkeypatch):
    result, trace = run_demo_with_trace(monkeypatch)
    message_types = [entry[3] for entry in trace]

    assert message_types[:5] == [
        "sift.bases",
        "sift.indices",
        "estimate.sample",
        "estimate.result",
        "cascade.start",
    ]
    assert message_types[-4:] == [
        "verify.tag",
        "verify.result",
        "privacy.seed",
        "finished",
    ]

    cascade_start = message_types.index("cascade.start") + 1
    cascade_end = message_types.index("verify.tag")
    cascade_messages = trace[cascade_start:cascade_end]
    assert len(cascade_messages) == 2 * result.bob.cascade_parity_requests

    for request, response in zip(cascade_messages[::2], cascade_messages[1::2]):
        assert request[3] == "cascade.parity_request"
        assert response[3] == "cascade.parity_response"
        assert response[4]["request_id"] == request[4]["request_id"]

    assert message_types.index("verify.result") < message_types.index("privacy.seed")
    assert message_types.index("privacy.seed") < message_types.index("finished")


def test_opened_qber_sample_bits_are_removed_before_cascade(monkeypatch):
    original_cascade_controller = agents.CascadeController
    cascade_input_bits: list[list[int]] = []

    class RecordingCascadeController(original_cascade_controller):
        def __init__(self, *args, **kwargs) -> None:
            cascade_input_bits.append(list(kwargs["bits"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(agents, "CascadeController", RecordingCascadeController)

    result, trace = run_demo_with_trace(monkeypatch)
    alice = result.alice
    bob = result.bob

    sample_message = next(
        body for _, _, _, typ, body in trace if typ == "estimate.sample"
    )
    opened_positions = set(sample_message["sample_positions"])

    assert (
        sample_message["sample_positions"]
        == alice.sample_positions
        == bob.sample_positions
    )
    assert sample_message["sample_bits"] == [
        alice.sifted_bits[position] for position in alice.sample_positions
    ]

    expected_alice = [
        bit
        for index, bit in enumerate(alice.sifted_bits)
        if index not in opened_positions
    ]
    expected_bob = [
        bit
        for index, bit in enumerate(bob.sifted_bits)
        if index not in opened_positions
    ]
    assert alice.reconciled_bits == expected_alice
    assert cascade_input_bits == [expected_bob]
    assert len(alice.reconciled_bits) == len(alice.sifted_bits) - alice.sample_size

    verify_message = next(body for _, _, _, typ, body in trace if typ == "verify.tag")
    privacy_message = next(
        body for _, _, _, typ, body in trace if typ == "privacy.seed"
    )
    assert verify_message["input_len"] == len(alice.reconciled_bits)
    assert privacy_message["input_len"] == len(alice.reconciled_bits)


def test_matched_missing_detections_are_rejected_before_sifting(monkeypatch):
    result, trace = run_demo_with_trace(monkeypatch)
    sift_indices_body = next(
        body for _, _, _, typ, body in trace if typ == "sift.indices"
    )

    assert sift_indices_body["rejected_missing_detections"] == (
        result.bob.missed_detection_indices
    )
    assert result.bob.missed_detection_indices
    assert all(bit is not None for bit in result.bob.sifted_bits)
    assert set(result.bob.missed_detection_indices).isdisjoint(result.bob.sift_indices)
    assert result.alice.sift_indices == result.bob.sift_indices
    assert len(result.alice.sifted_bits) == len(result.bob.sifted_bits)


def test_high_qber_aborts_before_cascade(monkeypatch):
    result, trace = run_demo_with_trace(
        monkeypatch,
        inputs_factory=high_qber_inputs,
    )
    message_types = [entry[3] for entry in trace]

    assert result.alice.aborted_reason == "estimated qber exceeds abort threshold"
    assert result.bob.aborted_reason == "estimated qber exceeds abort threshold"
    assert not result.alice.complete
    assert not result.bob.complete
    assert result.alice.final_key is None
    assert result.bob.final_key is None
    assert result.bob.estimated_qber is not None
    assert result.bob.estimated_qber > result.bob.qber_abort_threshold
    assert message_types == [
        "sift.bases",
        "sift.indices",
        "estimate.sample",
        "estimate.result",
        "abort",
    ]
    assert "cascade.start" not in message_types
    assert "verify.tag" not in message_types
    assert "privacy.seed" not in message_types


def test_too_few_sifted_bits_aborts_before_estimation(monkeypatch):
    result, trace = run_demo_with_trace(
        monkeypatch,
        inputs_factory=too_few_sifted_inputs,
    )
    message_types = [entry[3] for entry in trace]

    assert result.alice.aborted_reason == (
        "not enough sifted bits for estimation and key generation"
    )
    assert result.bob.aborted_reason == (
        "not enough sifted bits for estimation and key generation"
    )
    assert message_types == ["sift.bases", "sift.indices", "abort"]
    assert "estimate.sample" not in message_types
    assert "cascade.start" not in message_types
    assert "privacy.seed" not in message_types


def test_verification_failure_prevents_privacy_amplification(monkeypatch):
    def send_bad_verification_tag(self: agents.BobPostProcessor, ctx) -> None:
        cascade = self._require_cascade_controller()
        self.cascade_parity_requests = cascade.parity_requests
        self.cascade_corrections = cascade.corrections
        self.cascade_leaked_bits = cascade.leaked_bits

        seed_len = len(self.reconciled_bits) + self.verification_tag_len - 1
        tag_seed = [0] * seed_len
        tag = toeplitz_hash(
            self.reconciled_bits,
            tag_seed,
            self.verification_tag_len,
        )
        tag[0] ^= 1
        self.verification_leaked_bits = self.verification_tag_len
        self._send(
            ctx,
            "verify.tag",
            {
                "tag_seed": tag_seed,
                "tag": tag,
                "tag_len": self.verification_tag_len,
                "input_len": len(self.reconciled_bits),
                "hash_family": "toeplitz",
            },
        )

    monkeypatch.setattr(
        agents.BobPostProcessor,
        "_send_verification_tag",
        send_bad_verification_tag,
    )
    result, trace = run_demo_with_trace(monkeypatch)
    message_types = [entry[3] for entry in trace]

    assert "verify.tag" in message_types
    assert "verify.result" in message_types
    assert "privacy.seed" not in message_types
    assert "finished" not in message_types
    assert result.alice.aborted_reason == "verification failed"
    assert result.bob.aborted_reason == "verification failed"
    assert result.alice.final_key is None
    assert result.bob.final_key is None


def test_privacy_aborts_when_entropy_budget_is_too_small(monkeypatch):
    class TightBudgetBob(agents.BobPostProcessor):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.min_final_key_len = 512
            self.max_final_key_len = 512

    result, trace = run_demo_with_trace(monkeypatch, bob_cls=TightBudgetBob)
    message_types = [entry[3] for entry in trace]

    assert "verify.result" in message_types
    assert "privacy.seed" not in message_types
    assert "finished" not in message_types
    assert result.bob.demo_entropy_budget_bits is not None
    assert result.bob.demo_entropy_budget_bits < result.bob.min_final_key_len
    assert result.bob.aborted_reason == "not enough demo entropy budget for final key"
    assert result.alice.aborted_reason == "not enough demo entropy budget for final key"
    assert result.alice.final_key is None
    assert result.bob.final_key is None
