"""Event-driven E91 agents for the concurrent four-node tutorial."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from examples.bb84.helpers import send_json_message
from examples.postprocessing.bb84_event.cascade import CascadeController
from examples.postprocessing.bb84_event.helpers import (
    choose_sample_positions,
    parity,
    random_bits,
    remove_positions,
    require_binary_bits,
    require_positions,
    toeplitz_hash,
)
from examples.postprocessing.bb84_event.messages import decode_body
from simyuj.components.detectors import DetectionReport
from simyuj.components.sources import EntangledPairSource, SourcePreparationReport
from simyuj.control import AGENT_EVENT, NodeAgent
from simyuj.engine import Event

from .configs import E91PostProcessingConfig
from .helpers import (
    CHSH_CATEGORIES,
    CHSH_SETTING_PAIRS,
    KEY_SETTING_PAIRS,
    E91PrivacyBudget,
    bit_from_outcome,
    chsh_value,
    conservative_chsh_value,
    e91_privacy_budget,
    observed_correlation,
    pair_index_from_signal_id,
    singlet_corrected_bit,
)


@dataclass(frozen=True)
class E91DetectionRecord:
    """One successful E91 detector outcome indexed by source pair."""

    pair_index: int
    signal_id: str
    time: int
    basis: str
    outcome: str
    report_id: str
    detector_id: str | None
    flags: tuple[str, ...]


def _schedule_agent_action(
    agent: NodeAgent,
    ctx: Any,
    action: str,
    *,
    delay_ticks: int = 0,
) -> None:
    ctx.timeline.schedule(
        Event(
            time=ctx.timeline.current_time + delay_ticks,
            target_ref=agent,
            action=AGENT_EVENT,
            payload_ref={"e91_action": action},
            source=agent,
            subsystem_id="control",
            meta={
                "session_id": ctx.session_id,
                "agent_id": agent.agent_id,
                "e91_action": action,
            },
        )
    )


def _decode_manifest(
    body: dict[str, Any],
    *,
    field_name: str,
    allowed_bases: set[str],
) -> list[tuple[int, str]]:
    raw = body[field_name]
    if not isinstance(raw, list):
        raise TypeError(f"{field_name} must be a list")

    result: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"{field_name} entries must be objects")
        pair_index = int(item["pair_index"])
        basis = str(item["basis"])
        if pair_index <= 0:
            raise ValueError("E91 pair indices must be positive")
        if pair_index in seen:
            raise ValueError(f"duplicate pair index in {field_name}: {pair_index}")
        if basis not in allowed_bases:
            raise ValueError(f"invalid E91 basis in {field_name}: {basis!r}")
        seen.add(pair_index)
        result.append((pair_index, basis))
    return result


@dataclass(slots=True)
class E91SourceAgent(NodeAgent):
    """Start C's pair source and announce completion over physical fibers."""

    source: EntangledPairSource
    frame_done_delay_ticks: int
    b_peer_id: str = "b_e91_agent"
    d_peer_id: str = "d_e91_agent"
    b_out_port: str = "to_b_e91"
    d_out_port: str = "to_d_e91"
    preparations: list[SourcePreparationReport] = field(default_factory=list)
    message_counter: int = 0
    source_start_time: int | None = None
    frame_done_time: int | None = None

    def on_start(self, start: object, ctx: Any) -> None:
        del start
        self.source_start_time = ctx.timeline.current_time
        self.source.schedule_start(ctx.timeline)
        _schedule_agent_action(
            self,
            ctx,
            "frame.done",
            delay_ticks=self.frame_done_delay_ticks,
        )

    def on_report(self, report: object, ctx: Any) -> None:
        del ctx
        if not isinstance(report, SourcePreparationReport):
            raise TypeError(f"C received unsupported report: {type(report)!r}")
        self.preparations.append(report)

    def on_event(self, event: Event, ctx: Any) -> None:
        payload = event.payload_ref
        if not isinstance(payload, dict) or payload.get("e91_action") != "frame.done":
            raise ValueError(f"unsupported C E91 event: {payload!r}")

        self.frame_done_time = ctx.timeline.current_time
        body = {
            "prepared_pairs": len(self.preparations),
            "done_time": self.frame_done_time,
        }
        send_json_message(
            self,
            ctx,
            receiver_id=self.b_peer_id,
            out_port=self.b_out_port,
            message_type="e91.quantum_done",
            body=body,
        )
        send_json_message(
            self,
            ctx,
            receiver_id=self.d_peer_id,
            out_port=self.d_out_port,
            message_type="e91.quantum_done",
            body=body,
        )


@dataclass(slots=True)
class E91ReceiverBase(NodeAgent):
    """Shared detector-report storage for B and D."""

    postprocessing: E91PostProcessingConfig
    readiness_guard_ticks: int
    detector_reports: list[DetectionReport] = field(default_factory=list)
    detections_by_pair: dict[int, E91DetectionRecord] = field(default_factory=dict)
    failed_reports: list[DetectionReport] = field(default_factory=list)
    duplicate_pair_reports: list[DetectionReport] = field(default_factory=list)
    quantum_done_time: int | None = None
    ready_time: int | None = None
    protocol_start_time: int | None = None
    protocol_end_time: int | None = None
    aborted_reason: str | None = None
    message_counter: int = 0

    def on_start(self, start: object, ctx: Any) -> None:
        del start, ctx

    def on_report(self, report: object, ctx: Any) -> None:
        del ctx
        if not isinstance(report, DetectionReport):
            raise TypeError(f"E91 receiver got unsupported report: {type(report)!r}")
        self.detector_reports.append(report)
        if not report.success:
            self.failed_reports.append(report)
            return
        if report.signal_id is None or report.measurement_label is None:
            raise ValueError("successful E91 report lacks signal or basis metadata")

        pair_index = pair_index_from_signal_id(report.signal_id)
        if pair_index in self.detections_by_pair:
            self.duplicate_pair_reports.append(report)
            return
        outcome = str(report.outcome)
        bit_from_outcome(outcome)
        detector_id = report.raw_clicks[0].detector_id if report.raw_clicks else None
        self.detections_by_pair[pair_index] = E91DetectionRecord(
            pair_index=pair_index,
            signal_id=str(report.signal_id),
            time=report.time,
            basis=str(report.measurement_label),
            outcome=outcome,
            report_id=report.report_id,
            detector_id=detector_id,
            flags=report.flags,
        )

    def _on_quantum_done(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        self.quantum_done_time = int(body["done_time"])
        _schedule_agent_action(
            self,
            ctx,
            "receiver.ready",
            delay_ticks=self.readiness_guard_ticks,
        )

    def _set_peer_abort(self, message: Any, ctx: Any) -> None:
        del ctx
        body = decode_body(message)
        self.aborted_reason = str(body.get("reason", "peer aborted"))


@dataclass(slots=True)
class E91BAgent(E91ReceiverBase):
    """B-side E91 reference endpoint and parity responder."""

    peer_id: str = "d_e91_agent"
    out_port: str = "to_d_e91"
    d_manifest_basis: dict[int, str] = field(default_factory=dict)
    key_pair_indices: list[int] = field(default_factory=list)
    bell_pair_indices: dict[str, list[int]] = field(default_factory=dict)
    key_bits: list[int] = field(default_factory=list)
    sample_positions: list[int] = field(default_factory=list)
    estimated_qber: float | None = None
    qber_accepted: bool | None = None
    reconciled_bits: list[int] = field(default_factory=list)
    cascade_parity_requests: int = 0
    cascade_leaked_bits: int = 0
    verification_complete: bool = False
    verification_accepted: bool | None = None
    final_key: list[int] = field(default_factory=list)
    final_key_length: int = 0
    privacy_complete: bool = False
    bell_accepted: bool | None = None
    observed_s: float | None = None
    s_lower: float | None = None

    def on_event(self, event: Event, ctx: Any) -> None:
        payload = event.payload_ref
        if (
            not isinstance(payload, dict)
            or payload.get("e91_action") != "receiver.ready"
        ):
            raise ValueError(f"unsupported B E91 event: {payload!r}")
        self.ready_time = ctx.timeline.current_time
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.receiver_ready",
            body={
                "ready_time": self.ready_time,
                "successful_detections": len(self.detections_by_pair),
            },
        )

    def on_message(self, message: Any, ctx: Any) -> None:
        message_type = message.message.message_type
        if message_type == "e91.quantum_done":
            self._on_quantum_done(message, ctx)
        elif message_type == "e91.sift.d_manifest":
            self._on_d_manifest(message, ctx)
        elif message_type == "e91.sift.accepted":
            self._on_sift_accepted(message, ctx)
        elif message_type == "e91.bell.result":
            self._on_bell_result(message, ctx)
        elif message_type == "e91.estimate.sample":
            self._on_estimate_sample(message, ctx)
        elif message_type == "e91.cascade.parity_request":
            self._on_cascade_request(message, ctx)
        elif message_type == "e91.verify.tag":
            self._on_verify_tag(message, ctx)
        elif message_type == "e91.privacy.seed":
            self._on_privacy_seed(message, ctx)
        elif message_type == "e91.abort":
            self._set_peer_abort(message, ctx)
        else:
            raise ValueError(f"unsupported B E91 message: {message_type!r}")

    def _on_d_manifest(self, message: Any, ctx: Any) -> None:
        manifest = _decode_manifest(
            decode_body(message),
            field_name="detections",
            allowed_bases={"D0", "D1", "D2"},
        )
        self.protocol_start_time = ctx.timeline.current_time
        self.d_manifest_basis = dict(manifest)
        common = [
            {
                "pair_index": pair_index,
                "basis": self.detections_by_pair[pair_index].basis,
            }
            for pair_index, _d_basis in manifest
            if pair_index in self.detections_by_pair
        ]
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.sift.b_manifest",
            body={"detections": common},
        )

    def _on_sift_accepted(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        key_pair_indices = [int(value) for value in body["key_pair_indices"]]
        if len(key_pair_indices) != len(set(key_pair_indices)):
            raise ValueError("duplicate E91 key pair index")

        raw_categories = body["bell_categories"]
        if not isinstance(raw_categories, dict):
            raise TypeError("bell_categories must be an object")
        if set(raw_categories) != set(CHSH_CATEGORIES):
            raise ValueError("bell_categories must contain the four CHSH categories")
        bell_categories = {
            name: [int(value) for value in raw_categories[name]]
            for name in CHSH_CATEGORIES
        }
        all_accepted = key_pair_indices + [
            pair_index
            for name in CHSH_CATEGORIES
            for pair_index in bell_categories[name]
        ]
        if len(all_accepted) != len(set(all_accepted)):
            raise ValueError("E91 accepted pair categories overlap")
        missing = [
            pair_index
            for pair_index in all_accepted
            if pair_index not in self.detections_by_pair
            or pair_index not in self.d_manifest_basis
        ]
        if missing:
            raise ValueError(f"accepted E91 pairs are unknown at B: {missing[:5]}")
        invalid_key_pairs = [
            pair_index
            for pair_index in key_pair_indices
            if (
                self.detections_by_pair[pair_index].basis,
                self.d_manifest_basis[pair_index],
            )
            not in KEY_SETTING_PAIRS
        ]
        if invalid_key_pairs:
            raise ValueError(
                f"E91 key category has inconsistent settings: {invalid_key_pairs[:5]}"
            )
        for category, pair_indices in bell_categories.items():
            expected_settings = CHSH_SETTING_PAIRS[category]
            invalid_bell_pairs = [
                pair_index
                for pair_index in pair_indices
                if (
                    self.detections_by_pair[pair_index].basis,
                    self.d_manifest_basis[pair_index],
                )
                != expected_settings
            ]
            if invalid_bell_pairs:
                raise ValueError(
                    "E91 Bell category has inconsistent settings: "
                    f"{invalid_bell_pairs[:5]}"
                )

        self.key_pair_indices = key_pair_indices
        self.bell_pair_indices = bell_categories
        self.key_bits = [
            bit_from_outcome(self.detections_by_pair[pair_index].outcome)
            for pair_index in key_pair_indices
        ]

        samples = [
            {
                "pair_index": pair_index,
                "outcome": self.detections_by_pair[pair_index].outcome,
            }
            for name in CHSH_CATEGORIES
            for pair_index in bell_categories[name]
        ]
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.bell.sample",
            body={"samples": samples},
        )

    def _on_bell_result(self, message: Any, ctx: Any) -> None:
        del ctx
        body = decode_body(message)
        self.observed_s = float(body["observed_s"])
        self.s_lower = float(body["s_lower"])
        accepted = body["accept"]
        if not isinstance(accepted, bool):
            raise TypeError("E91 Bell acceptance must be boolean")
        self.bell_accepted = accepted
        if not self.bell_accepted:
            self.aborted_reason = str(body.get("reason", "Bell test failed"))

    def _on_estimate_sample(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        positions = [int(value) for value in body["sample_positions"]]
        d_bits = [int(value) for value in body["sample_bits"]]
        require_positions("E91 sample positions", positions, len(self.key_bits))
        require_binary_bits("E91 sample bits", d_bits)
        if len(positions) != len(d_bits) or not positions:
            raise ValueError("invalid E91 QBER sample")

        errors = sum(
            self.key_bits[position] != d_bit
            for position, d_bit in zip(positions, d_bits)
        )
        qber = errors / len(positions)
        accepted = qber <= self.postprocessing.qber_abort_threshold
        self.sample_positions = positions
        self.estimated_qber = qber
        self.qber_accepted = accepted
        self.reconciled_bits = remove_positions(self.key_bits, positions)
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.estimate.result",
            body={
                "sample_positions": positions,
                "sample_errors": errors,
                "qber": qber,
                "accept": accepted,
            },
        )
        if not accepted:
            self.aborted_reason = "estimated E91 QBER exceeds abort threshold"

    def _on_cascade_request(self, message: Any, ctx: Any) -> None:
        if not self.qber_accepted:
            raise RuntimeError("cannot answer E91 Cascade before QBER acceptance")
        body = decode_body(message)
        request_id = str(body["request_id"])
        indices = [int(value) for value in body["indices"]]
        require_positions("E91 Cascade indices", indices, len(self.reconciled_bits))
        self.cascade_parity_requests += 1
        self.cascade_leaked_bits += 1
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.cascade.parity_response",
            body={
                "request_id": request_id,
                "parity": parity(self.reconciled_bits, indices),
            },
        )

    def _on_verify_tag(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        input_len = int(body["input_len"])
        tag_len = int(body["tag_len"])
        seed = [int(value) for value in body["tag_seed"]]
        d_tag = [int(value) for value in body["tag"]]
        require_binary_bits("E91 verification seed", seed)
        require_binary_bits("E91 verification tag", d_tag)
        verified = (
            input_len == len(self.reconciled_bits)
            and toeplitz_hash(
                self.reconciled_bits,
                seed,
                tag_len,
            )
            == d_tag
        )
        self.verification_complete = True
        self.verification_accepted = verified
        if not verified:
            self.aborted_reason = "E91 verification failed"
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.verify.result",
            body={"verified": verified, "tag_len": tag_len},
        )

    def _on_privacy_seed(self, message: Any, ctx: Any) -> None:
        if not self.verification_accepted:
            raise RuntimeError("cannot amplify E91 key before verification")
        body = decode_body(message)
        input_len = int(body["input_len"])
        output_len = int(body["final_key_len"])
        seed = [int(value) for value in body["seed"]]
        require_binary_bits("E91 privacy seed", seed)
        if input_len != len(self.reconciled_bits):
            raise ValueError("E91 privacy-amplification input length mismatch")
        if output_len < self.postprocessing.min_final_key_bits:
            raise ValueError("E91 privacy-amplification output is too short")

        self.final_key = toeplitz_hash(self.reconciled_bits, seed, output_len)
        self.final_key_length = len(self.final_key)
        self.privacy_complete = True
        self.protocol_end_time = ctx.timeline.current_time
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.privacy.done",
            body={"final_key_len": self.final_key_length},
        )


@dataclass(slots=True)
class E91DAgent(E91ReceiverBase):
    """D-side E91 controller for sifting through privacy amplification."""

    peer_id: str = "b_e91_agent"
    out_port: str = "to_b_e91"
    local_ready: bool = False
    b_ready: bool = False
    sifting_started: bool = False
    b_basis_by_pair: dict[int, str] = field(default_factory=dict)
    key_pair_indices: list[int] = field(default_factory=list)
    bell_pair_indices: dict[str, list[int]] = field(default_factory=dict)
    key_bits: list[int] = field(default_factory=list)
    bell_counts: dict[str, int] = field(default_factory=dict)
    bell_correlations: dict[str, float] = field(default_factory=dict)
    observed_s: float | None = None
    s_lower: float | None = None
    bell_accepted: bool | None = None
    sample_positions: list[int] = field(default_factory=list)
    sample_errors: int = 0
    estimated_qber: float | None = None
    qber_accepted: bool | None = None
    reconciled_bits: list[int] = field(default_factory=list)
    cascade_complete: bool = False
    cascade_first_block_size: int | None = None
    cascade_parity_requests: int = 0
    cascade_corrections: int = 0
    cascade_leaked_bits: int = 0
    verification_complete: bool = False
    verification_accepted: bool | None = None
    verification_leaked_bits: int = 0
    privacy_budget: E91PrivacyBudget | None = None
    final_key: list[int] = field(default_factory=list)
    final_key_length: int = 0
    privacy_complete: bool = False
    _sample_rng: object | None = field(default=None, repr=False)
    _cascade_rng: object | None = field(default=None, repr=False)
    _verification_rng: object | None = field(default=None, repr=False)
    _privacy_rng: object | None = field(default=None, repr=False)
    _cascade_controller: CascadeController | None = field(default=None, repr=False)

    def bind(self, context: Any) -> None:
        self._sample_rng = context.timeline.rng(self.agent_id, "e91", "qber_sample")
        self._cascade_rng = context.timeline.rng(self.agent_id, "e91", "cascade")
        self._verification_rng = context.timeline.rng(
            self.agent_id, "e91", "verification"
        )
        self._privacy_rng = context.timeline.rng(
            self.agent_id, "e91", "privacy_amplification"
        )

    def on_event(self, event: Event, ctx: Any) -> None:
        payload = event.payload_ref
        if not isinstance(payload, dict):
            raise TypeError("D E91 local event payload must be an object")
        action = payload.get("e91_action")
        if action == "receiver.ready":
            self.ready_time = ctx.timeline.current_time
            self.local_ready = True
            self._maybe_start_sifting(ctx)
        elif action == "cascade.next_request":
            self._cascade_next_request(ctx)
        else:
            raise ValueError(f"unsupported D E91 event: {payload!r}")

    def on_message(self, message: Any, ctx: Any) -> None:
        message_type = message.message.message_type
        if message_type == "e91.quantum_done":
            self._on_quantum_done(message, ctx)
        elif message_type == "e91.receiver_ready":
            self.b_ready = True
            self._maybe_start_sifting(ctx)
        elif message_type == "e91.sift.b_manifest":
            self._on_b_manifest(message, ctx)
        elif message_type == "e91.bell.sample":
            self._on_bell_sample(message, ctx)
        elif message_type == "e91.estimate.result":
            self._on_estimate_result(message, ctx)
        elif message_type == "e91.cascade.parity_response":
            self._on_cascade_response(message, ctx)
        elif message_type == "e91.verify.result":
            self._on_verify_result(message, ctx)
        elif message_type == "e91.privacy.done":
            self._on_privacy_done(message, ctx)
        elif message_type == "e91.abort":
            self._set_peer_abort(message, ctx)
        else:
            raise ValueError(f"unsupported D E91 message: {message_type!r}")

    def _maybe_start_sifting(self, ctx: Any) -> None:
        if not self.local_ready or not self.b_ready or self.sifting_started:
            return
        self.sifting_started = True
        self.protocol_start_time = ctx.timeline.current_time
        manifest = [
            {"pair_index": pair_index, "basis": record.basis}
            for pair_index, record in sorted(self.detections_by_pair.items())
        ]
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.sift.d_manifest",
            body={"detections": manifest},
        )

    def _on_b_manifest(self, message: Any, ctx: Any) -> None:
        manifest = _decode_manifest(
            decode_body(message),
            field_name="detections",
            allowed_bases={"B0", "B1", "B2"},
        )
        unknown = [
            pair_index
            for pair_index, _basis in manifest
            if pair_index not in self.detections_by_pair
        ]
        if unknown:
            raise ValueError(f"B announced unknown D pair indices: {unknown[:5]}")
        self.b_basis_by_pair = dict(manifest)

        key_pairs: list[int] = []
        bell_categories: dict[str, list[int]] = {name: [] for name in CHSH_CATEGORIES}
        category_for_settings = {
            settings: name for name, settings in CHSH_SETTING_PAIRS.items()
        }
        for pair_index, b_basis in manifest:
            d_basis = self.detections_by_pair[pair_index].basis
            settings = (b_basis, d_basis)
            if settings in KEY_SETTING_PAIRS:
                key_pairs.append(pair_index)
            elif settings in category_for_settings:
                bell_categories[category_for_settings[settings]].append(pair_index)

        self.key_pair_indices = key_pairs
        self.bell_pair_indices = bell_categories
        self.key_bits = [
            singlet_corrected_bit(self.detections_by_pair[pair_index].outcome)
            for pair_index in key_pairs
        ]
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.sift.accepted",
            body={
                "key_pair_indices": key_pairs,
                "bell_categories": bell_categories,
            },
        )

    def _on_bell_sample(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        samples = body["samples"]
        if not isinstance(samples, list):
            raise TypeError("E91 Bell samples must be a list")
        b_outcomes: dict[int, str] = {}
        for item in samples:
            if not isinstance(item, dict):
                raise TypeError("E91 Bell sample entries must be objects")
            pair_index = int(item["pair_index"])
            if pair_index in b_outcomes:
                raise ValueError(f"duplicate E91 Bell sample: {pair_index}")
            outcome = str(item["outcome"])
            bit_from_outcome(outcome)
            b_outcomes[pair_index] = outcome

        expected = {
            pair_index
            for category in CHSH_CATEGORIES
            for pair_index in self.bell_pair_indices[category]
        }
        if set(b_outcomes) != expected:
            raise ValueError("E91 Bell sample IDs do not match accepted categories")

        correlations: dict[str, float] = {}
        counts: dict[str, int] = {}
        for category in CHSH_CATEGORIES:
            pair_indices = self.bell_pair_indices[category]
            counts[category] = len(pair_indices)
            if pair_indices:
                correlations[category] = observed_correlation(
                    (
                        b_outcomes[pair_index],
                        self.detections_by_pair[pair_index].outcome,
                    )
                    for pair_index in pair_indices
                )
            else:
                correlations[category] = 0.0

        observed_s = chsh_value(correlations)
        s_lower = conservative_chsh_value(
            observed_s,
            self.postprocessing.chsh_safety_margin,
        )
        enough_samples = all(
            count >= self.postprocessing.min_chsh_samples_per_category
            for count in counts.values()
        )
        accepted = enough_samples and s_lower > 2.0
        reason = None
        if not enough_samples:
            reason = "insufficient E91 CHSH samples"
        elif s_lower <= 2.0:
            reason = "E91 Bell test did not exceed the conservative threshold"

        self.bell_counts = counts
        self.bell_correlations = correlations
        self.observed_s = observed_s
        self.s_lower = s_lower
        self.bell_accepted = accepted
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.bell.result",
            body={
                "counts": counts,
                "correlations": correlations,
                "observed_s": observed_s,
                "s_lower": s_lower,
                "accept": accepted,
                "reason": reason,
            },
        )
        if not accepted:
            self.aborted_reason = reason
            return
        self._start_qber_estimation(ctx)

    def _start_qber_estimation(self, ctx: Any) -> None:
        n = len(self.key_bits)
        needed = (
            self.postprocessing.min_qber_sample_bits
            + self.postprocessing.min_remaining_bits
        )
        if n < needed:
            self._abort(
                ctx,
                f"not enough E91 key bits for QBER estimation: have {n}, need {needed}",
            )
            return
        if self._sample_rng is None:
            raise RuntimeError("D E91 sample RNG is not bound")

        sample_size = min(
            n - self.postprocessing.min_remaining_bits,
            max(
                self.postprocessing.min_qber_sample_bits,
                int(self.postprocessing.qber_sample_fraction * n),
            ),
        )
        positions = choose_sample_positions(self._sample_rng, n, sample_size)
        self.sample_positions = positions
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.estimate.sample",
            body={
                "sample_positions": positions,
                "sample_bits": [self.key_bits[position] for position in positions],
            },
        )

    def _on_estimate_result(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        positions = [int(value) for value in body["sample_positions"]]
        if positions != self.sample_positions:
            raise ValueError("B returned different E91 sample positions")
        self.sample_errors = int(body["sample_errors"])
        self.estimated_qber = float(body["qber"])
        self.qber_accepted = bool(body["accept"])
        self.reconciled_bits = remove_positions(self.key_bits, positions)
        if not self.qber_accepted:
            self.aborted_reason = "estimated E91 QBER exceeds abort threshold"
            return
        self._start_cascade(ctx)

    def _start_cascade(self, ctx: Any) -> None:
        if not self.reconciled_bits:
            self._abort(ctx, "E91 reconciled input is empty")
            return
        if self._cascade_rng is None:
            raise RuntimeError("D E91 Cascade RNG is not bound")
        n = len(self.reconciled_bits)
        block_qber = max(
            float(self.estimated_qber or 0.0),
            self.postprocessing.cascade_block_qber_floor,
            1.0 / n,
        )
        first_block_size = min(
            n,
            self.postprocessing.cascade_max_first_block_size,
            max(
                self.postprocessing.cascade_min_block_size,
                math.ceil(0.73 / block_qber),
            ),
        )
        self.cascade_first_block_size = first_block_size
        self._cascade_controller = CascadeController(
            bits=self.reconciled_bits,
            rng=self._cascade_rng,
            passes=self.postprocessing.cascade_passes,
            first_block_size=first_block_size,
        )
        _schedule_agent_action(self, ctx, "cascade.next_request")

    def _cascade_next_request(self, ctx: Any) -> None:
        if self._cascade_controller is None:
            raise RuntimeError("D E91 Cascade controller is not initialized")
        request = self._cascade_controller.next_request()
        if request is None:
            if self._cascade_controller.complete:
                cascade = self._cascade_controller
                self.cascade_complete = True
                self.cascade_parity_requests = cascade.parity_requests
                self.cascade_corrections = cascade.corrections
                self.cascade_leaked_bits = cascade.leaked_bits
                self._start_verification(ctx)
            else:
                _schedule_agent_action(self, ctx, "cascade.next_request")
            return
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.cascade.parity_request",
            body=request.as_body(),
        )

    def _on_cascade_response(self, message: Any, ctx: Any) -> None:
        if self._cascade_controller is None:
            raise RuntimeError("D E91 Cascade controller is not initialized")
        body = decode_body(message)
        self._cascade_controller.apply_parity_response(
            request_id=str(body["request_id"]),
            alice_parity=int(body["parity"]),
        )
        _schedule_agent_action(self, ctx, "cascade.next_request")

    def _start_verification(self, ctx: Any) -> None:
        if self._verification_rng is None:
            raise RuntimeError("D E91 verification RNG is not bound")
        tag_len = self.postprocessing.verification_tag_len
        seed = random_bits(
            self._verification_rng,
            len(self.reconciled_bits) + tag_len - 1,
        )
        tag = toeplitz_hash(self.reconciled_bits, seed, tag_len)
        self.verification_leaked_bits = tag_len
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.verify.tag",
            body={
                "hash_family": "toeplitz",
                "input_len": len(self.reconciled_bits),
                "tag_len": tag_len,
                "tag_seed": seed,
                "tag": tag,
            },
        )

    def _on_verify_result(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        verified = body["verified"]
        if not isinstance(verified, bool):
            raise TypeError("E91 verification result must be boolean")
        self.verification_complete = True
        self.verification_accepted = verified
        if not verified:
            self.aborted_reason = "E91 verification failed"
            return
        self._start_privacy_amplification(ctx)

    def _start_privacy_amplification(self, ctx: Any) -> None:
        if self._privacy_rng is None:
            raise RuntimeError("D E91 privacy RNG is not bound")
        if self.s_lower is None or self.estimated_qber is None:
            raise RuntimeError("E91 security parameters are incomplete")
        budget = e91_privacy_budget(
            reconciled_length=len(self.reconciled_bits),
            s_lower=self.s_lower,
            estimated_qber=self.estimated_qber,
            qber_safety_margin=self.postprocessing.qber_safety_margin,
            cascade_disclosures=self.cascade_leaked_bits,
            verification_tag_len=self.postprocessing.verification_tag_len,
            security_margin_bits=self.postprocessing.security_margin_bits,
        )
        self.privacy_budget = budget
        if budget.final_key_length < self.postprocessing.min_final_key_bits:
            self._abort(ctx, "E91 final key too short")
            return

        seed = random_bits(
            self._privacy_rng,
            len(self.reconciled_bits) + budget.final_key_length - 1,
        )
        self.final_key = toeplitz_hash(
            self.reconciled_bits,
            seed,
            budget.final_key_length,
        )
        self.final_key_length = len(self.final_key)
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.privacy.seed",
            body={
                "input_len": len(self.reconciled_bits),
                "final_key_len": budget.final_key_length,
                "seed": seed,
                "budget": budget.as_dict(),
            },
        )

    def _on_privacy_done(self, message: Any, ctx: Any) -> None:
        body = decode_body(message)
        if int(body["final_key_len"]) != self.final_key_length:
            self.aborted_reason = "E91 privacy-amplification length mismatch"
            return
        self.privacy_complete = True
        self.protocol_end_time = ctx.timeline.current_time

    def _abort(self, ctx: Any, reason: str) -> None:
        if self.aborted_reason is not None:
            return
        self.aborted_reason = reason
        self.protocol_end_time = ctx.timeline.current_time
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="e91.abort",
            body={"reason": reason},
        )


__all__ = [
    "E91BAgent",
    "E91DAgent",
    "E91DetectionRecord",
    "E91ReceiverBase",
    "E91SourceAgent",
]
