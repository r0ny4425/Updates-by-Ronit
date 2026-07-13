"""Physical Alice and Bob agents for the single-photon BB84 example."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
from simyuj.components import SinglePhotonSource, SourcePreparationReport
from simyuj.components.detectors import DetectionReport
from simyuj.control import AGENT_EVENT, NodeAgent
from simyuj.engine import Event

from .configs import (
    BB84CascadeConfig,
    BB84PrivacyAmplificationConfig,
    BB84QBERConfig,
    BB84VerificationConfig,
)
from .helpers import (
    assign_detection_slot_index,
    bb84_basis_bit,
    bb84_final_key_length,
    bob_report_basis_bit,
    report_detection_time,
    send_json_message,
)

qber_config = BB84QBERConfig()
cascade_config = BB84CascadeConfig()
verification_config = BB84VerificationConfig()
privacy_config = BB84PrivacyAmplificationConfig()


@dataclass(frozen=True)
class AlicePreparationRecord:
    """Alice's local record for one emitted source slot."""

    slot_index: int
    signal_id: str
    time: int
    emission_slot_tick: int
    emission_delay_ticks: int
    attempt_index: int
    emission_index: int
    basis: str
    bit: int
    sampler_label: str
    report_id: str


@dataclass(slots=True)
class BB84AliceAgent(NodeAgent):
    """Alice-side protocol state machine.

    Alice starts the source, records preparation reports, announces the end of
    the quantum frame, and answers Bob's public post-processing messages.
    """

    source: SinglePhotonSource
    quantum_done_delay_ticks: int = 0
    qber_settings: BB84QBERConfig = field(default_factory=BB84QBERConfig)
    privacy_settings: BB84PrivacyAmplificationConfig = field(
        default_factory=BB84PrivacyAmplificationConfig
    )

    preparations: list[AlicePreparationRecord] = field(default_factory=list)
    preparations_by_signal_id: dict[str, AlicePreparationRecord] = field(
        default_factory=dict
    )
    preparations_by_slot_index: dict[int, AlicePreparationRecord] = field(
        default_factory=dict
    )

    peer_id: str = "bob_agent"
    out_port: str = "to_bob"
    sifted_slot_indices: list[int] = field(default_factory=list)
    sifted_signal_ids: list[str] = field(default_factory=list)
    sifted_bits: list[int] = field(default_factory=list)
    no_emission_sift_rejections: int = 0
    sifting_complete: bool = False

    sample_positions: list[int] = field(default_factory=list)
    estimated_qber: float | None = None
    sample_errors: int = 0
    qber_accepted: bool | None = None
    reconciled_bits: list[int] = field(default_factory=list)
    qber_complete: bool = False
    aborted_reason: str | None = None

    cascade_parity_requests: int = 0
    cascade_leaked_bits: int = 0

    verification_complete: bool = False
    verification_accepted: bool | None = None
    verification_leaked_bits: int = 0

    privacy_complete: bool = False
    final_key: list[int] = field(default_factory=list)
    final_key_length: int = 0
    privacy_revealed_bits: int = 0

    message_counter: int = 0
    _sample_rng: object | None = field(default=None, repr=False)

    def bind(self, context) -> None:
        self._sample_rng = context.timeline.rng(
            self.agent_id,
            "bb84",
            "qber_sample",
        )

    def _schedule_local_action(
        self,
        ctx,
        action: str,
        *,
        delay_ticks: int = 0,
    ) -> None:
        ctx.timeline.schedule(
            Event(
                time=ctx.timeline.current_time + delay_ticks,
                target_ref=self,
                action=AGENT_EVENT,
                payload_ref={"bb84_action": action},
                source=self,
                subsystem_id="control",
                meta={
                    "session_id": ctx.session_id,
                    "agent_id": self.agent_id,
                    "bb84_action": action,
                },
            )
        )

    def on_start(self, start, ctx) -> None:
        del start
        self.source.schedule_start(ctx.timeline)
        self._schedule_local_action(
            ctx,
            "quantum.done",
            delay_ticks=self.quantum_done_delay_ticks,
        )

    def on_report(self, report: object, ctx) -> None:
        del ctx

        if not isinstance(report, SourcePreparationReport):
            raise TypeError(f"Alice received unsupported report: {type(report)!r}")

        if report.sampler_label is None:
            raise ValueError("source report must include a BB84 sampler label")
        if report.emission_index is None:
            raise ValueError("source report must include an emission_index")

        basis, bit = bb84_basis_bit(report.sampler_label)

        for signal_id in report.signal_ids:
            record = AlicePreparationRecord(
                slot_index=int(report.attempt_index),
                signal_id=signal_id,
                time=report.time,
                emission_slot_tick=report.emission_slot_tick,
                emission_delay_ticks=report.emission_delay_ticks,
                attempt_index=report.attempt_index,
                emission_index=report.emission_index,
                basis=basis,
                bit=bit,
                sampler_label=report.sampler_label,
                report_id=report.report_id,
            )
            self.preparations.append(record)
            self.preparations_by_signal_id[signal_id] = record
            self.preparations_by_slot_index[record.slot_index] = record

    def on_message(self, message, ctx) -> None:
        message_type = message.message.message_type

        if message_type == "sift.bob_bases":
            self._on_bob_bases(message, ctx)
            return

        if message_type == "estimate.result":
            self._on_estimate_result(message, ctx)
            return

        if message_type == "cascade.parity_request":
            self._on_cascade_parity_request(message, ctx)
            return

        if message_type == "verify.tag":
            self._on_verify_tag(message, ctx)
            return

        if message_type == "privacy.seed":
            self._on_privacy_seed(message, ctx)
            return

        raise ValueError(f"unsupported Alice message: {message_type!r}")

    def on_event(self, event: Event, ctx) -> None:
        payload = event.payload_ref
        if not isinstance(payload, dict):
            raise TypeError("Alice local event payload must be dict")

        action = payload.get("bb84_action")

        if action == "quantum.done":
            self._send_quantum_done(ctx)
            return

        raise ValueError(f"unsupported Alice local action: {payload!r}")

    def _send_quantum_done(self, ctx) -> None:
        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="quantum.done",
            body={
                "prepared_count": len(self.preparations),
                "done_time": ctx.timeline.current_time,
            },
        )

    def _on_bob_bases(self, message, ctx) -> None:
        body = decode_body(message)
        detections = body["detections"]

        accepted_slot_indices = []
        seen_slot_indices = set()
        self.no_emission_sift_rejections = 0

        for item in detections:
            slot_index = int(item["slot_index"])
            bob_basis = str(item["basis"])

            if bob_basis not in ("Z", "X"):
                raise ValueError(f"invalid Bob basis: {bob_basis!r}")

            if slot_index in seen_slot_indices:
                raise ValueError(f"duplicate Bob slot_index announcement: {slot_index}")

            seen_slot_indices.add(slot_index)

            alice_record = self.preparations_by_slot_index.get(slot_index)
            if alice_record is None:
                self.no_emission_sift_rejections += 1
                continue

            if alice_record.basis == bob_basis:
                accepted_slot_indices.append(slot_index)

        self.sifted_slot_indices = accepted_slot_indices
        self.sifted_signal_ids = [
            self.preparations_by_slot_index[slot_index].signal_id
            for slot_index in accepted_slot_indices
        ]
        self.sifted_bits = [
            self.preparations_by_slot_index[slot_index].bit
            for slot_index in accepted_slot_indices
        ]
        self.reconciled_bits = self.sifted_bits.copy()
        self.sifting_complete = True

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="sift.accepted",
            body={"slot_indices": accepted_slot_indices},
        )

        self._start_qber_estimation(ctx)

    def _start_qber_estimation(self, ctx) -> None:
        if not self.sifting_complete:
            raise RuntimeError("cannot estimate QBER before sifting completes")

        n = len(self.sifted_bits)
        needed = (
            self.qber_settings.min_sample_bits + self.qber_settings.min_remaining_bits
        )
        if n < needed:
            self.aborted_reason = (
                f"not enough sifted bits for QBER estimation: have {n}, need {needed}"
            )
            send_json_message(
                self,
                ctx,
                receiver_id=self.peer_id,
                out_port=self.out_port,
                message_type="abort",
                body={"reason": self.aborted_reason},
            )
            return

        if self._sample_rng is None:
            raise RuntimeError("Alice QBER sample RNG is not bound")

        sample_size = min(
            n - self.qber_settings.min_remaining_bits,
            max(
                self.qber_settings.min_sample_bits,
                int(self.qber_settings.sample_fraction * n),
            ),
        )

        self.sample_positions = choose_sample_positions(
            self._sample_rng,
            n,
            sample_size,
        )
        sample_bits = [self.sifted_bits[position] for position in self.sample_positions]

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="estimate.sample",
            body={
                "sample_positions": self.sample_positions,
                "sample_bits": sample_bits,
            },
        )

    def _on_estimate_result(self, message, ctx) -> None:
        del ctx

        body = decode_body(message)

        self.estimated_qber = float(body["qber"])
        self.qber_accepted = bool(body["accept"])
        self.sample_errors = int(body["sample_errors"])

        returned_positions = [int(pos) for pos in body["sample_positions"]]
        require_positions(
            "sample_positions",
            returned_positions,
            len(self.sifted_bits),
        )

        if returned_positions != self.sample_positions:
            raise ValueError("Bob returned different sample positions")

        self.reconciled_bits = remove_positions(
            self.sifted_bits,
            self.sample_positions,
        )
        self.qber_complete = True

        if not self.qber_accepted:
            self.aborted_reason = "estimated QBER exceeds abort threshold"

    def _on_cascade_parity_request(self, message, ctx) -> None:
        if not self.qber_complete:
            raise RuntimeError("cannot answer Cascade before QBER estimation completes")

        body = decode_body(message)
        request_id = str(body["request_id"])
        indices = [int(index) for index in body["indices"]]

        require_positions(
            "cascade request indices",
            indices,
            len(self.reconciled_bits),
        )

        self.cascade_parity_requests += 1
        self.cascade_leaked_bits += 1

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="cascade.parity_response",
            body={
                "request_id": request_id,
                "parity": parity(self.reconciled_bits, indices),
            },
        )

    def _on_verify_tag(self, message, ctx) -> None:
        if not self.qber_complete:
            raise RuntimeError("cannot verify before QBER estimation completes")

        body = decode_body(message)

        input_len = int(body["input_len"])
        tag_len = int(body["tag_len"])
        tag_seed = [int(bit) for bit in body["tag_seed"]]
        bob_tag = [int(bit) for bit in body["tag"]]

        require_binary_bits("verification tag_seed", tag_seed)
        require_binary_bits("verification bob_tag", bob_tag)

        verified = False

        if input_len == len(self.reconciled_bits):
            alice_tag = toeplitz_hash(
                self.reconciled_bits,
                tag_seed,
                tag_len,
            )
            verified = alice_tag == bob_tag

        self.verification_complete = True
        self.verification_accepted = verified
        self.verification_leaked_bits = tag_len

        if not verified:
            self.aborted_reason = "verification failed"

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="verify.result",
            body={
                "verified": verified,
                "tag_len": tag_len,
            },
        )

    def _on_privacy_seed(self, message, ctx) -> None:
        if not self.verification_accepted:
            raise RuntimeError(
                "cannot run privacy amplification before verification passes"
            )

        body = decode_body(message)

        input_len = int(body["input_len"])
        final_key_len = int(body["final_key_len"])
        seed = [int(bit) for bit in body["seed"]]

        require_binary_bits("privacy seed", seed)

        if input_len != len(self.reconciled_bits):
            raise ValueError("privacy amplification input length mismatch")

        if final_key_len < self.privacy_settings.min_final_key_bits:
            self.aborted_reason = "final key too short"
            return

        self.final_key = toeplitz_hash(
            self.reconciled_bits,
            seed,
            final_key_len,
        )
        self.final_key_length = len(self.final_key)
        self.privacy_revealed_bits = int(body["revealed_bits"])
        self.privacy_complete = True

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="privacy.done",
            body={
                "final_key_len": self.final_key_length,
            },
        )


@dataclass(frozen=True)
class BobDetectionRecord:
    """Bob's local record for one usable detector report."""

    slot_index: int
    signal_id: str
    time: int
    detection_time: int
    basis: str
    bit: int
    outcome: object
    report_id: str
    detector_id: str | None
    flags: tuple[str, ...]


@dataclass(slots=True)
class BB84BobAgent(NodeAgent):
    """Bob-side protocol state machine.

    Bob records detector reports, assigns detections to source slots by timing,
    starts sifting after Alice's quantum-done message, and drives the later
    post-processing stages.
    """

    sifting_guard_ticks: int = 0
    slot_period_ticks: int = 0
    slot_zero_arrival_tick: int = 0
    slot_assignment_window_ticks: int = 0
    num_slots: int = 0
    qber_settings: BB84QBERConfig = field(default_factory=BB84QBERConfig)
    cascade_settings: BB84CascadeConfig = field(default_factory=BB84CascadeConfig)
    verification_settings: BB84VerificationConfig = field(
        default_factory=BB84VerificationConfig
    )
    privacy_settings: BB84PrivacyAmplificationConfig = field(
        default_factory=BB84PrivacyAmplificationConfig
    )

    detector_reports: list[DetectionReport] = field(default_factory=list)
    detections: list[BobDetectionRecord] = field(default_factory=list)
    detections_by_signal_id: dict[str, BobDetectionRecord] = field(default_factory=dict)
    detections_by_slot_index: dict[int, BobDetectionRecord] = field(
        default_factory=dict
    )
    failed_reports: list[DetectionReport] = field(default_factory=list)
    unassigned_reports: list[DetectionReport] = field(default_factory=list)
    duplicate_slot_reports: list[DetectionReport] = field(default_factory=list)

    quantum_done_received: bool = False
    quantum_done_time: int | None = None
    sifting_started: bool = False

    peer_id: str = "alice_agent"
    out_port: str = "to_alice"
    sifted_slot_indices: list[int] = field(default_factory=list)
    sifted_signal_ids: list[str] = field(default_factory=list)
    sifted_bits: list[int] = field(default_factory=list)
    sifting_complete: bool = False

    sample_positions: list[int] = field(default_factory=list)
    sample_errors: int = 0
    estimated_qber: float | None = None
    qber_accepted: bool | None = None
    reconciled_bits: list[int] = field(default_factory=list)
    qber_complete: bool = False
    aborted_reason: str | None = None

    cascade_complete: bool = False
    cascade_first_block_size: int | None = None
    cascade_parity_requests: int = 0
    cascade_corrections: int = 0
    cascade_leaked_bits: int = 0
    _cascade_controller: CascadeController | None = field(default=None, repr=False)
    _cascade_rng: object | None = field(default=None, repr=False)

    verification_complete: bool = False
    verification_accepted: bool | None = None
    verification_tag_len: int = 0
    verification_leaked_bits: int = 0
    _verification_rng: object | None = field(default=None, repr=False)

    privacy_complete: bool = False
    final_key: list[int] = field(default_factory=list)
    final_key_length: int = 0
    privacy_revealed_bits: int = 0
    _privacy_rng: object | None = field(default=None, repr=False)

    message_counter: int = 0

    def bind(self, context) -> None:
        self._cascade_rng = context.timeline.rng(
            self.agent_id,
            "bb84",
            "cascade",
        )
        self._verification_rng = context.timeline.rng(
            self.agent_id,
            "bb84",
            "verification",
        )
        self._privacy_rng = context.timeline.rng(
            self.agent_id,
            "bb84",
            "privacy_amplification",
        )

    def on_start(self, start, ctx) -> None:
        del start, ctx

    def on_report(self, report: object, ctx) -> None:
        del ctx

        if not isinstance(report, DetectionReport):
            raise TypeError(f"Bob received unsupported report: {type(report)!r}")

        self.detector_reports.append(report)

        decoded = bob_report_basis_bit(report)
        if decoded is None:
            self.failed_reports.append(report)
            return

        if report.signal_id is None:
            raise ValueError("successful Bob detection report must include signal_id")

        detection_time = report_detection_time(report)
        slot_index = assign_detection_slot_index(
            detection_time,
            slot_period_ticks=self.slot_period_ticks,
            slot_zero_arrival_tick=self.slot_zero_arrival_tick,
            slot_assignment_window_ticks=self.slot_assignment_window_ticks,
            num_slots=self.num_slots,
        )
        if slot_index is None:
            self.unassigned_reports.append(report)
            return

        if slot_index in self.detections_by_slot_index:
            self.duplicate_slot_reports.append(report)
            return

        basis, bit = decoded
        detector_id = report.raw_clicks[0].detector_id if report.raw_clicks else None

        record = BobDetectionRecord(
            slot_index=slot_index,
            signal_id=str(report.signal_id),
            time=report.time,
            detection_time=detection_time,
            basis=basis,
            bit=bit,
            outcome=report.outcome,
            report_id=report.report_id,
            detector_id=detector_id,
            flags=report.flags,
        )

        self.detections.append(record)
        self.detections_by_signal_id[record.signal_id] = record
        self.detections_by_slot_index[record.slot_index] = record

    def on_event(self, event: Event, ctx) -> None:
        payload = event.payload_ref
        if not isinstance(payload, dict):
            raise TypeError("Bob local event payload must be dict")

        action = payload.get("bb84_action")

        if action == "sifting.start":
            self._start_sifting(ctx)
            return

        if action == "cascade.next_request":
            self._cascade_next_request(ctx)
            return

        raise ValueError(f"unsupported Bob local action: {payload!r}")

    def on_message(self, message, ctx) -> None:
        message_type = message.message.message_type

        if message_type == "quantum.done":
            self._on_quantum_done(message, ctx)
            return

        if message_type == "sift.accepted":
            self._on_sift_accepted(message, ctx)
            return

        if message_type == "estimate.sample":
            self._on_estimate_sample(message, ctx)
            return

        if message_type == "cascade.parity_response":
            self._on_cascade_parity_response(message, ctx)
            return

        if message_type == "abort":
            body = decode_body(message)
            self.aborted_reason = str(body.get("reason", "peer aborted"))
            return

        if message_type == "verify.result":
            self._on_verify_result(message, ctx)
            return

        if message_type == "privacy.done":
            self._on_privacy_done(message, ctx)
            return

        raise ValueError(f"unsupported Bob message: {message_type!r}")

    def _on_quantum_done(self, message, ctx) -> None:
        body = decode_body(message)
        self.quantum_done_received = True
        self.quantum_done_time = int(body["done_time"])
        self._schedule_local_action(
            ctx,
            "sifting.start",
            delay_ticks=self.sifting_guard_ticks,
        )

    def _start_sifting(self, ctx) -> None:
        if self.sifting_started or self.sifting_complete:
            return

        self.sifting_started = True

        detections = [
            {"slot_index": record.slot_index, "basis": record.basis}
            for record in self.detections
        ]

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="sift.bob_bases",
            body={"detections": detections},
        )

    def _on_sift_accepted(self, message, ctx) -> None:
        del ctx

        body = decode_body(message)
        accepted_slot_indices = [int(slot_index) for slot_index in body["slot_indices"]]

        missing = [
            slot_index
            for slot_index in accepted_slot_indices
            if slot_index not in self.detections_by_slot_index
        ]
        if missing:
            raise ValueError(f"Alice accepted unknown Bob slots: {missing[:5]}")

        self.sifted_slot_indices = accepted_slot_indices
        self.sifted_signal_ids = [
            self.detections_by_slot_index[slot_index].signal_id
            for slot_index in accepted_slot_indices
        ]
        self.sifted_bits = [
            self.detections_by_slot_index[slot_index].bit
            for slot_index in accepted_slot_indices
        ]
        self.reconciled_bits = self.sifted_bits.copy()
        self.sifting_complete = True

    def _on_estimate_sample(self, message, ctx) -> None:
        if not self.sifting_complete:
            raise RuntimeError("cannot estimate QBER before sifting completes")

        body = decode_body(message)
        sample_positions = [int(pos) for pos in body["sample_positions"]]
        sample_bits = [int(bit) for bit in body["sample_bits"]]

        require_positions(
            "sample_positions",
            sample_positions,
            len(self.sifted_bits),
        )
        require_binary_bits("sample_bits", sample_bits)

        if len(sample_positions) != len(sample_bits):
            raise ValueError("sample_positions and sample_bits length mismatch")

        if not sample_positions:
            raise ValueError("sample_positions must not be empty")

        sample_errors = sum(
            self.sifted_bits[position] != alice_bit
            for position, alice_bit in zip(sample_positions, sample_bits)
        )
        qber = sample_errors / len(sample_positions)
        accept = qber <= self.qber_settings.abort_threshold

        self.sample_positions = sample_positions
        self.sample_errors = sample_errors
        self.estimated_qber = qber
        self.qber_accepted = accept
        self.reconciled_bits = remove_positions(
            self.sifted_bits,
            sample_positions,
        )
        self.qber_complete = True

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="estimate.result",
            body={
                "qber": qber,
                "accept": accept,
                "sample_positions": sample_positions,
                "sample_size": len(sample_positions),
                "sample_errors": sample_errors,
            },
        )

        if not accept:
            self.aborted_reason = "estimated QBER exceeds abort threshold"
            return

        self._start_cascade(ctx)

    def _schedule_local_action(
        self,
        ctx,
        action: str,
        *,
        delay_ticks: int = 0,
    ) -> None:
        ctx.timeline.schedule(
            Event(
                time=ctx.timeline.current_time + delay_ticks,
                target_ref=self,
                action=AGENT_EVENT,
                payload_ref={"bb84_action": action},
                source=self,
                subsystem_id="control",
                meta={
                    "session_id": ctx.session_id,
                    "agent_id": self.agent_id,
                    "bb84_action": action,
                },
            )
        )

    def _schedule_cascade_next(self, ctx) -> None:
        self._schedule_local_action(ctx, "cascade.next_request")

    def _start_cascade(self, ctx) -> None:
        if not self.qber_complete:
            raise RuntimeError("cannot start Cascade before QBER estimation completes")

        if not self.qber_accepted:
            raise RuntimeError("cannot start Cascade after QBER abort")

        n = len(self.reconciled_bits)
        if n == 0:
            raise RuntimeError("cannot start Cascade with empty reconciled bits")

        if self._cascade_rng is None:
            raise RuntimeError("Bob Cascade RNG is not bound")

        block_qber = max(
            float(self.estimated_qber or 0.0),
            self.cascade_settings.block_qber_floor,
            1.0 / n,
        )

        first_block_size = min(
            n,
            self.cascade_settings.max_first_block_size,
            max(
                self.cascade_settings.min_block_size,
                math.ceil(0.73 / block_qber),
            ),
        )

        self.cascade_first_block_size = first_block_size
        self._cascade_controller = CascadeController(
            bits=self.reconciled_bits,
            rng=self._cascade_rng,
            passes=self.cascade_settings.passes,
            first_block_size=first_block_size,
        )
        self._schedule_cascade_next(ctx)

    def _cascade_next_request(self, ctx) -> None:
        cascade = self._require_cascade_controller()
        request = cascade.next_request()

        if request is None:
            if cascade.complete:
                self.cascade_complete = True
                self.cascade_parity_requests = cascade.parity_requests
                self.cascade_corrections = cascade.corrections
                self.cascade_leaked_bits = cascade.leaked_bits
                self._start_verification(ctx)
                return

            self._schedule_cascade_next(ctx)
            return

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="cascade.parity_request",
            body=request.as_body(),
        )

    def _on_cascade_parity_response(self, message, ctx) -> None:
        body = decode_body(message)
        cascade = self._require_cascade_controller()
        cascade.apply_parity_response(
            request_id=str(body["request_id"]),
            alice_parity=int(body["parity"]),
        )
        self._schedule_cascade_next(ctx)

    def _require_cascade_controller(self) -> CascadeController:
        if self._cascade_controller is None:
            raise RuntimeError("Cascade controller is not initialized")
        return self._cascade_controller

    def _start_verification(self, ctx) -> None:
        if not self.cascade_complete:
            raise RuntimeError("cannot verify before Cascade completes")

        if self._verification_rng is None:
            raise RuntimeError("Bob verification RNG is not bound")

        input_len = len(self.reconciled_bits)
        if input_len == 0:
            raise RuntimeError("cannot verify an empty key")

        tag_len = self.verification_settings.tag_len
        tag_seed = random_bits(self._verification_rng, input_len + tag_len - 1)
        tag = toeplitz_hash(self.reconciled_bits, tag_seed, tag_len)

        self.verification_tag_len = tag_len
        self.verification_leaked_bits = tag_len

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="verify.tag",
            body={
                "hash_family": "toeplitz",
                "input_len": input_len,
                "tag_len": tag_len,
                "tag_seed": tag_seed,
                "tag": tag,
            },
        )

    def _on_verify_result(self, message, ctx) -> None:
        body = decode_body(message)
        verified = body["verified"]

        if not isinstance(verified, bool):
            raise TypeError("verification result must be boolean")

        self.verification_complete = True
        self.verification_accepted = verified

        if not verified:
            self.aborted_reason = "verification failed"
            return

        self._start_privacy_amplification(ctx)

    def _start_privacy_amplification(self, ctx) -> None:
        if not self.verification_accepted:
            raise RuntimeError("cannot amplify privacy before verification passes")

        if self._privacy_rng is None:
            raise RuntimeError("Bob privacy amplification RNG is not bound")

        reconciled_len = len(self.reconciled_bits)

        revealed_bits = (
            len(self.sample_positions)
            + self.cascade_leaked_bits
            + self.verification_leaked_bits
        )

        final_key_len = bb84_final_key_length(
            reconciled_len=reconciled_len,
            estimated_qber=float(self.estimated_qber or 0.0),
            revealed_bits=revealed_bits,
            qber_safety_margin=self.privacy_settings.qber_safety_margin,
            security_margin_bits=self.privacy_settings.security_margin_bits,
        )

        if final_key_len < self.privacy_settings.min_final_key_bits:
            self.aborted_reason = "final key too short"
            return

        seed = random_bits(
            self._privacy_rng,
            reconciled_len + final_key_len - 1,
        )

        self.final_key = toeplitz_hash(
            self.reconciled_bits,
            seed,
            final_key_len,
        )
        self.final_key_length = len(self.final_key)
        self.privacy_revealed_bits = revealed_bits

        send_json_message(
            self,
            ctx,
            receiver_id=self.peer_id,
            out_port=self.out_port,
            message_type="privacy.seed",
            body={
                "input_len": reconciled_len,
                "final_key_len": final_key_len,
                "seed": seed,
                "revealed_bits": revealed_bits,
                "estimated_qber": self.estimated_qber,
                "qber_safety_margin": self.privacy_settings.qber_safety_margin,
                "security_margin_bits": self.privacy_settings.security_margin_bits,
            },
        )

    def _on_privacy_done(self, message, ctx) -> None:
        del ctx

        body = decode_body(message)
        alice_final_key_len = int(body["final_key_len"])

        if alice_final_key_len != self.final_key_length:
            self.aborted_reason = "privacy amplification length mismatch"
            return

        self.privacy_complete = True


__all__ = [
    "AlicePreparationRecord",
    "BB84AliceAgent",
    "BB84BobAgent",
    "BobDetectionRecord",
    "cascade_config",
    "privacy_config",
    "qber_config",
    "verification_config",
]
