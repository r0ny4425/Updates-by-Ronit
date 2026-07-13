"""Event-driven Alice/Bob agents for the BB84 post-processing example.

These agents start from already-measured basis/bit arrays. They do not model
the optical link. Their job is to express BB84 post-processing as public
classical messages and local timeline events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from simyuj.control import AGENT_EVENT, Agent, AgentContext
from simyuj.control.payloads import AgentMessage
from simyuj.engine import Event
from simyuj.primitives.messages import ClassicalMessage
from simyuj.runtime.binding import BindingContext

from .cascade import CascadeController, CascadeState
from .helpers import (
    choose_sample_positions,
    demo_entropy_budget,
    measured_bit,
    parity,
    random_bits,
    remove_positions,
    require_bases,
    require_binary_bits,
    require_measured_bits,
    require_optional_binary_bits,
    require_positions,
    toeplitz_hash,
)
from .messages import decode_body, encode_body


@dataclass(slots=True)
class BB84PostProcessor(Agent):
    """Shared state and utilities for Alice and Bob post-processing agents."""

    peer_id: str
    bases: list[str]
    bits: list[int | None]
    out_port: str

    qber_abort_threshold: float = 0.11
    sample_fraction: float = 0.20
    min_sample_bits: int = 8
    min_remaining_bits: int = 64
    cascade_passes: int = 4
    verification_tag_len: int = 64
    statistical_margin: float = 0.02
    security_margin_bits: int = 32
    max_final_key_len: int = 64
    min_final_key_len: int = 16

    sift_indices: list[int] = field(default_factory=list)
    sifted_bits: list[int] = field(default_factory=list)
    reconciled_bits: list[int] = field(default_factory=list)
    final_key: list[int] | None = None
    aborted_reason: str | None = None
    complete: bool = False

    estimated_qber: float | None = None
    sample_positions: list[int] = field(default_factory=list)
    sample_size: int = 0
    sample_errors: int = 0

    cascade_first_block_size: int | None = None
    cascade_parity_requests: int = 0
    cascade_corrections: int = 0
    cascade_leaked_bits: int = 0
    verification_leaked_bits: int = 0
    privacy_seed_bits: int = 0
    demo_phase_error_bound: float | None = None
    demo_eve_info_bound: int | None = None
    demo_entropy_budget_bits: int | None = None
    missed_detection_indices: list[int] = field(default_factory=list)

    _message_counter: int = field(init=False, default=0)
    _sample_rng: Any = field(init=False, default=None, repr=False)
    _cascade_rng: Any = field(init=False, default=None, repr=False)
    _verification_rng: Any = field(init=False, default=None, repr=False)
    _privacy_rng: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        Agent.__post_init__(self)
        if len(self.bases) != len(self.bits):
            raise ValueError("bases and bits must have the same length")
        require_bases("bases", self.bases)
        require_optional_binary_bits("bits", self.bits)
        if not (0.0 <= self.sample_fraction <= 1.0):
            raise ValueError("sample_fraction must be in [0, 1]")
        if self.min_sample_bits <= 0:
            raise ValueError("min_sample_bits must be positive")
        self.enable_classical()

    def bind(self, context: BindingContext) -> None:
        """Bind deterministic RNG streams to this agent and timeline."""
        self._sample_rng = context.timeline.rng(
            self.agent_id, "postprocessing", "sample"
        )
        self._cascade_rng = context.timeline.rng(
            self.agent_id, "postprocessing", "cascade"
        )
        self._verification_rng = context.timeline.rng(
            self.agent_id, "postprocessing", "verification"
        )
        self._privacy_rng = context.timeline.rng(
            self.agent_id, "postprocessing", "privacy"
        )

    def _send(
        self,
        ctx: AgentContext,
        message_type: str,
        body: dict[str, Any],
    ) -> Event:
        """Send one JSON classical message through the configured output port."""
        message = ClassicalMessage(
            sender_id=self.agent_id,
            receiver_id=self.peer_id,
            body=encode_body(body),
            sent_time=ctx.timeline.current_time,
            session_id=ctx.session_id,
            message_type=message_type,
            message_id=f"{self.agent_id}-{self._message_counter}",
        )
        self._message_counter += 1
        return self.classical.send(
            message,
            ctx.timeline,
            port_name=self.out_port,
        )

    def _schedule_local(
        self,
        ctx: AgentContext,
        action: str,
        body: dict[str, Any] | None = None,
    ) -> Event:
        """Schedule an internal protocol step at the current simulation time."""
        return ctx.timeline.schedule(
            Event(
                time=ctx.timeline.current_time,
                target_ref=self,
                action=AGENT_EVENT,
                payload_ref={
                    "postprocessing_action": action,
                    "body": {} if body is None else body,
                },
                source=self,
                subsystem_id="control",
                meta={
                    "session_id": ctx.session_id,
                    "agent_id": self.agent_id,
                    "postprocessing_action": action,
                },
            )
        )

    def _abort(
        self,
        ctx: AgentContext,
        reason: str,
        *,
        notify_peer: bool = True,
    ) -> None:
        if self.aborted_reason is None:
            self.aborted_reason = reason
        if notify_peer:
            self._send(ctx, "abort", {"reason": reason})

    def _record_abort(self, reason: str) -> None:
        if self.aborted_reason is None:
            self.aborted_reason = reason

    def _local_action(self, event: Event) -> tuple[str, dict[str, Any]]:
        payload = event.payload_ref
        if not isinstance(payload, dict):
            raise TypeError("local event payload must be a dict")
        action = payload.get("postprocessing_action")
        body = payload.get("body", {})
        if not isinstance(action, str):
            raise TypeError("postprocessing_action must be str")
        if not isinstance(body, dict):
            raise TypeError("local event body must be a dict")
        return action, body


@dataclass(slots=True)
class AlicePostProcessor(BB84PostProcessor):
    """Alice's side of event-based BB84 post-processing.

    Alice starts sifting by publishing bases, chooses the QBER sample, answers
    Cascade parity requests, checks Bob's verification tag, and applies the
    privacy-amplification seed.
    """

    def __post_init__(self) -> None:
        BB84PostProcessor.__post_init__(self)
        require_measured_bits("Alice bits", self.bits)

    def on_start(self, start: object, ctx: AgentContext) -> None:
        del start
        self._send(ctx, "sift.bases", {"bases": self.bases})

    def on_message(self, message: AgentMessage, ctx: AgentContext) -> None:
        if message.message.message_type == "abort":
            body = decode_body(message)
            self._record_abort(str(body.get("reason", "peer aborted")))
            return
        if self.aborted_reason is not None or self.complete:
            return

        body = decode_body(message)
        if message.message.message_type == "sift.indices":
            self._on_sift_indices(ctx, body)
        elif message.message.message_type == "estimate.result":
            self._on_estimate_result(ctx, body)
        elif message.message.message_type == "cascade.parity_request":
            self._on_parity_request(ctx, body)
        elif message.message.message_type == "verify.tag":
            self._on_verify_tag(ctx, body)
        elif message.message.message_type == "privacy.seed":
            self._on_privacy_seed(ctx, body)
        else:
            raise ValueError(
                f"unsupported Alice message: {message.message.message_type!r}"
            )

    def on_event(self, event: Event, ctx: AgentContext) -> None:
        if self.aborted_reason is not None or self.complete:
            return

        action, _ = self._local_action(event)
        if action == "estimate.choose_sample":
            self._choose_estimate_sample(ctx)
            return
        if action == "abort":
            self._abort(ctx, "local abort")
            return
        raise ValueError(f"unsupported Alice local action: {action!r}")

    def _on_sift_indices(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        indices = [int(index) for index in body["indices"]]
        require_positions("sift_indices", indices, len(self.bits))
        self.sift_indices = indices
        self.sifted_bits = [measured_bit(self.bits, index) for index in indices]
        self.reconciled_bits = self.sifted_bits.copy()
        self._schedule_local(ctx, "estimate.choose_sample")

    def _choose_estimate_sample(self, ctx: AgentContext) -> None:
        """Reveal a public sample and keep enough undisclosed bits for a key."""
        n = len(self.reconciled_bits)
        if n < self.min_sample_bits + self.min_remaining_bits:
            self._abort(
                ctx,
                "not enough sifted bits for estimation and key generation",
            )
            return

        sample_size = min(
            n - self.min_remaining_bits,
            max(self.min_sample_bits, int(self.sample_fraction * n)),
        )
        self.sample_positions = choose_sample_positions(
            self._sample_rng,
            n,
            sample_size,
        )
        self.sample_size = sample_size
        sample_bits = [self.reconciled_bits[pos] for pos in self.sample_positions]
        self._send(
            ctx,
            "estimate.sample",
            {
                "sample_positions": self.sample_positions,
                "sample_bits": sample_bits,
            },
        )

    def _on_estimate_result(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        self.estimated_qber = float(body["qber"])
        self.sample_positions = [int(pos) for pos in body["sample_positions"]]
        require_positions(
            "sample_positions",
            self.sample_positions,
            len(self.reconciled_bits),
        )
        self.sample_size = int(body["sample_size"])
        if self.sample_size != len(self.sample_positions):
            raise ValueError("sample_size must equal number of sample_positions")
        self.sample_errors = int(body["sample_errors"])

        if not bool(body["accept"]):
            self._abort(ctx, "estimated qber exceeds abort threshold")
            return

        # Sample bits have been publicly revealed, so they are removed before
        # Cascade and privacy amplification.
        self.reconciled_bits = remove_positions(
            self.reconciled_bits,
            self.sample_positions,
        )
        n = len(self.reconciled_bits)
        # Original Cascade-style teaching heuristic. It gives large blocks when
        # sampled QBER is low, which is useful for demonstrating the protocol
        # flow but is not a tuned policy for every multi-error pattern.
        qber_floor = max(self.estimated_qber, 1.0 / n)
        first_block_size = min(n, max(2, math.ceil(0.73 / qber_floor)))
        self.cascade_first_block_size = first_block_size
        self._send(
            ctx,
            "cascade.start",
            {
                "qber": self.estimated_qber,
                "passes": self.cascade_passes,
                "first_block_size": first_block_size,
                "pass_indexing": "zero_based",
            },
        )

    def _on_parity_request(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        indices = [int(index) for index in body["indices"]]
        require_positions(
            "cascade request indices",
            indices,
            len(self.reconciled_bits),
        )
        self._send(
            ctx,
            "cascade.parity_response",
            {
                "request_id": body["request_id"],
                "parity": parity(self.reconciled_bits, indices),
            },
        )

    def _on_verify_tag(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        tag_seed = [int(bit) for bit in body["tag_seed"]]
        tag = [int(bit) for bit in body["tag"]]
        tag_len = int(body["tag_len"])
        alice_tag = toeplitz_hash(self.reconciled_bits, tag_seed, tag_len)
        verified = alice_tag == tag
        self.verification_leaked_bits = tag_len
        self._send(ctx, "verify.result", {"verified": verified})
        if not verified:
            self._abort(ctx, "verification failed", notify_peer=False)

    def _on_privacy_seed(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        final_key_len = int(body["final_key_len"])
        toeplitz_seed = [int(bit) for bit in body["toeplitz_seed"]]
        self.privacy_seed_bits = len(toeplitz_seed)
        self.final_key = toeplitz_hash(
            self.reconciled_bits,
            toeplitz_seed,
            final_key_len,
        )
        self.complete = True
        self._send(ctx, "finished", {"final_key_len": final_key_len})


@dataclass(slots=True)
class BobPostProcessor(BB84PostProcessor):
    """Bob's side of event-based BB84 post-processing.

    Bob accepts Alice's bases, rejects missed detections during sifting,
    estimates QBER from Alice's sample, drives Cascade, sends the verification
    tag, and chooses the privacy-amplification seed.
    """

    _cascade_controller: CascadeController | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def on_start(self, start: object, ctx: AgentContext) -> None:
        del start, ctx

    def on_message(self, message: AgentMessage, ctx: AgentContext) -> None:
        if message.message.message_type == "abort":
            body = decode_body(message)
            self._record_abort(str(body.get("reason", "peer aborted")))
            return
        if self.aborted_reason is not None or self.complete:
            return

        body = decode_body(message)
        if message.message.message_type == "sift.bases":
            self._on_sift_bases(ctx, body)
        elif message.message.message_type == "estimate.sample":
            self._on_estimate_sample(ctx, body)
        elif message.message.message_type == "cascade.start":
            self._on_cascade_start(ctx, body)
        elif message.message.message_type == "cascade.parity_response":
            self._on_parity_response(ctx, body)
        elif message.message.message_type == "verify.result":
            self._on_verify_result(ctx, body)
        elif message.message.message_type == "finished":
            self.complete = True
        else:
            raise ValueError(
                f"unsupported Bob message: {message.message.message_type!r}"
            )

    def on_event(self, event: Event, ctx: AgentContext) -> None:
        if self.aborted_reason is not None or self.complete:
            return

        action, _ = self._local_action(event)
        if action == "cascade.next_request":
            self._cascade_next_request(ctx)
            return
        if action == "cascade.completed":
            self._send_verification_tag(ctx)
            return
        if action == "privacy.prepare":
            self._prepare_privacy(ctx)
            return
        if action == "abort":
            self._abort(ctx, "local abort")
            return
        raise ValueError(f"unsupported Bob local action: {action!r}")

    def _on_sift_bases(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        alice_bases = [str(basis) for basis in body["bases"]]
        require_bases("alice_bases", alice_bases)
        if len(alice_bases) != len(self.bases):
            raise ValueError("Alice and Bob basis lists must have the same length")
        matched_missing = [
            i
            for i, (alice_basis, bob_basis) in enumerate(zip(alice_bases, self.bases))
            if alice_basis == bob_basis and self.bits[i] is None
        ]
        matching_indices = [
            i
            for i, (alice_basis, bob_basis) in enumerate(zip(alice_bases, self.bases))
            if alice_basis == bob_basis and self.bits[i] is not None
        ]
        self.missed_detection_indices = matched_missing
        self.sift_indices = matching_indices
        # Missed detections are rejected at sifting. Downstream QBER, Cascade,
        # verification, and privacy amplification only receive measured bits.
        self.sifted_bits = [measured_bit(self.bits, i) for i in matching_indices]
        self.reconciled_bits = self.sifted_bits.copy()
        self._send(
            ctx,
            "sift.indices",
            {
                "indices": matching_indices,
                "rejected_missing_detections": matched_missing,
            },
        )

    def _on_estimate_sample(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        sample_positions = [int(pos) for pos in body["sample_positions"]]
        sample_bits = [int(bit) for bit in body["sample_bits"]]
        require_positions(
            "sample_positions",
            sample_positions,
            len(self.reconciled_bits),
        )
        require_binary_bits("sample_bits", sample_bits)
        if len(sample_positions) != len(sample_bits):
            raise ValueError(
                "sample_positions and sample_bits must have the same length"
            )
        sample_size = len(sample_positions)
        if sample_size == 0:
            raise ValueError("sample_positions must not be empty")
        sample_errors = sum(
            self.reconciled_bits[pos] != alice_bit
            for pos, alice_bit in zip(sample_positions, sample_bits)
        )
        qber = sample_errors / sample_size
        accept = qber <= self.qber_abort_threshold

        self.sample_positions = sample_positions
        self.sample_size = sample_size
        self.sample_errors = sample_errors
        self.estimated_qber = qber
        # The revealed sample is no longer secret key material on Bob's side
        # either. Both agents remove the same positions.
        self.reconciled_bits = remove_positions(self.reconciled_bits, sample_positions)

        self._send(
            ctx,
            "estimate.result",
            {
                "qber": qber,
                "accept": accept,
                "sample_positions": sample_positions,
                "sample_size": sample_size,
                "sample_errors": sample_errors,
            },
        )
        if not accept:
            self._record_abort("estimated qber exceeds abort threshold")

    def _on_cascade_start(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        self.estimated_qber = float(body["qber"])
        first_block_size = int(body["first_block_size"])
        self.cascade_first_block_size = first_block_size
        self._cascade_controller = CascadeController(
            bits=self.reconciled_bits,
            rng=self._cascade_rng,
            passes=int(body["passes"]),
            first_block_size=first_block_size,
        )
        self._schedule_local(ctx, "cascade.next_request")

    def _cascade_next_request(self, ctx: AgentContext) -> None:
        cascade = self._require_cascade_controller()
        request = cascade.next_request()

        if request is None:
            # A None request means either Cascade is complete or the controller
            # advanced internal pass state and needs to be polled again.
            if cascade.complete:
                self._schedule_local(ctx, "cascade.completed")
            else:
                self._schedule_local(ctx, "cascade.next_request")
            return

        self._send(
            ctx,
            "cascade.parity_request",
            request.as_body(),
        )

    def _on_parity_response(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        cascade = self._require_cascade_controller()
        cascade.apply_parity_response(
            request_id=str(body["request_id"]),
            alice_parity=int(body["parity"]),
        )
        self._schedule_local(ctx, "cascade.next_request")

    def _send_verification_tag(self, ctx: AgentContext) -> None:
        cascade = self._require_cascade_controller()
        self.cascade_parity_requests = cascade.parity_requests
        self.cascade_corrections = cascade.corrections
        self.cascade_leaked_bits = cascade.leaked_bits

        seed_len = len(self.reconciled_bits) + self.verification_tag_len - 1
        tag_seed = random_bits(self._verification_rng, seed_len)
        tag = toeplitz_hash(
            self.reconciled_bits,
            tag_seed,
            self.verification_tag_len,
        )
        # The tag is public, so its length is counted as leaked information.
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

    def _on_verify_result(self, ctx: AgentContext, body: dict[str, Any]) -> None:
        if not bool(body["verified"]):
            self._abort(ctx, "verification failed")
            return
        self._schedule_local(ctx, "privacy.prepare")

    def _prepare_privacy(self, ctx: AgentContext) -> None:
        if self.estimated_qber is None:
            raise RuntimeError("estimated_qber is required before privacy")
        # This is a compact teaching budget, not a composable finite-key proof.
        # It keeps the example honest about public leakage and safety margins.
        available, phase_error_bound, eve_info_bound = demo_entropy_budget(
            n=len(self.reconciled_bits),
            estimated_qber=self.estimated_qber,
            statistical_margin=self.statistical_margin,
            cascade_leaked_bits=self.cascade_leaked_bits,
            verification_leaked_bits=self.verification_tag_len,
            security_margin_bits=self.security_margin_bits,
        )
        self.demo_entropy_budget_bits = available
        self.demo_phase_error_bound = phase_error_bound
        self.demo_eve_info_bound = eve_info_bound

        final_key_len = min(self.max_final_key_len, max(0, available))
        if final_key_len < self.min_final_key_len:
            self._abort(ctx, "not enough demo entropy budget for final key")
            return

        seed_len = len(self.reconciled_bits) + final_key_len - 1
        toeplitz_seed = random_bits(self._privacy_rng, seed_len)
        self.privacy_seed_bits = seed_len
        self.final_key = toeplitz_hash(
            self.reconciled_bits,
            toeplitz_seed,
            final_key_len,
        )
        self._send(
            ctx,
            "privacy.seed",
            {
                "toeplitz_seed": toeplitz_seed,
                "final_key_len": final_key_len,
                "input_len": len(self.reconciled_bits),
                "hash_family": "toeplitz",
                "budget_model": "demo_entropy_budget_not_finite_key_proof",
            },
        )

    def _require_cascade_controller(self) -> CascadeController:
        if self._cascade_controller is None:
            raise RuntimeError("Cascade controller has not been initialized")
        return self._cascade_controller

    def _require_cascade_state(self) -> CascadeState:
        return self._require_cascade_controller().state
