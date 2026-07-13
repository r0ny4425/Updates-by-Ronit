"""Runnable teaching demo for event-based BB84 post-processing."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "examples.postprocessing.bb84_event"

from simyuj.components import ACTION_TRANSMIT_CLASSICAL, ClassicalChannel, connect_ports
from simyuj.control import AGENT_MESSAGE, SessionRuntime
from simyuj.engine import Timeline
from simyuj.network import Network, Node

from .agents import AlicePostProcessor, BobPostProcessor
from .helpers import measured_bit


@dataclass(slots=True)
class DemoResult:
    """Objects produced by one demo run."""

    alice: AlicePostProcessor
    bob: BobPostProcessor
    timeline: Timeline
    alice_to_bob: ClassicalChannel
    bob_to_alice: ClassicalChannel


def make_demo_inputs(
    raw_bits: int = 1024,
    *,
    error_sift_positions: tuple[int, ...] = (17,),
    missed_detection_count: int | None = None,
) -> tuple[list[str], list[int | None], list[str], list[int | None]]:
    """Build deterministic post-measurement data for the teaching run.

    Assumptions: roughly half the bases match, Bob has a small number of
    injected bit errors plus a few missed detections on matching positions, and
    no quantum transmission is simulated here. Missed detections are represented
    by ``None`` and are rejected during sifting before key bits are formed.
    """
    if raw_bits <= 0:
        raise ValueError("raw_bits must be positive")
    alice_bases = ["Z" if index % 2 == 0 else "X" for index in range(raw_bits)]
    bob_bases = [
        (
            alice_bases[index]
            if index % 4 in (0, 1)
            else ("X" if alice_bases[index] == "Z" else "Z")
        )
        for index in range(raw_bits)
    ]
    alice_bits: list[int | None] = [
        ((index * 7) + (index // 5)) & 1 for index in range(raw_bits)
    ]
    bob_bits: list[int | None] = alice_bits.copy()

    matching = [
        index
        for index, (alice_basis, bob_basis) in enumerate(zip(alice_bases, bob_bases))
        if alice_basis == bob_basis
    ]
    injected_positions = {
        sift_pos for sift_pos in error_sift_positions if 0 <= sift_pos < len(matching)
    }
    if not matching:
        return alice_bases, alice_bits, bob_bases, bob_bits

    # Default teaching error positions keep the run reproducible. User-provided
    # positions are sifted-key positions, not raw-bit positions.
    for sift_pos in sorted(injected_positions):
        raw_index = matching[sift_pos]
        bob_bits[raw_index] = 1 - measured_bit(bob_bits, raw_index)

    if missed_detection_count is None:
        desired_misses = min(5, len(matching) // 80)
    else:
        if missed_detection_count < 0:
            raise ValueError("missed_detection_count must be non-negative")
        desired_misses = min(missed_detection_count, len(matching))
    missed_positions = {
        max(0, len(matching) - 1 - miss_index * max(1, len(matching) // 20))
        for miss_index in range(desired_misses)
    }
    for sift_pos in sorted(missed_positions - injected_positions):
        bob_bits[matching[sift_pos]] = None

    return alice_bases, alice_bits, bob_bases, bob_bits


def build_demo(
    *,
    master_seed: int = 2025,
    raw_bits: int = 1024,
    distance_km: float = 20.0,
    error_sift_positions: tuple[int, ...] = (17,),
    missed_detection_count: int | None = None,
    qber_abort_threshold: float = 0.11,
    sample_fraction: float = 0.20,
    min_sample_bits: int = 8,
    min_remaining_bits: int = 64,
    cascade_passes: int = 4,
    verification_tag_len: int = 64,
    statistical_margin: float = 0.02,
    security_margin_bits: int = 32,
    max_final_key_len: int = 64,
    min_final_key_len: int = 16,
) -> DemoResult:
    """Build the two-agent public-channel network and run it to completion."""
    alice_bases, alice_bits, bob_bases, bob_bits = make_demo_inputs(
        raw_bits=raw_bits,
        error_sift_positions=error_sift_positions,
        missed_detection_count=missed_detection_count,
    )
    session_id = "bb84-postprocessing"
    channel_length_m = distance_km * 1000

    timeline = Timeline(master_seed=master_seed)
    alice = AlicePostProcessor(
        agent_id="alice",
        peer_id="bob",
        bases=alice_bases,
        bits=alice_bits,
        out_port="to_bob",
        qber_abort_threshold=qber_abort_threshold,
        sample_fraction=sample_fraction,
        min_sample_bits=min_sample_bits,
        min_remaining_bits=min_remaining_bits,
        cascade_passes=cascade_passes,
        verification_tag_len=verification_tag_len,
        statistical_margin=statistical_margin,
        security_margin_bits=security_margin_bits,
        max_final_key_len=max_final_key_len,
        min_final_key_len=min_final_key_len,
    )
    bob = BobPostProcessor(
        agent_id="bob",
        peer_id="alice",
        bases=bob_bases,
        bits=bob_bits,
        out_port="to_alice",
        qber_abort_threshold=qber_abort_threshold,
        sample_fraction=sample_fraction,
        min_sample_bits=min_sample_bits,
        min_remaining_bits=min_remaining_bits,
        cascade_passes=cascade_passes,
        verification_tag_len=verification_tag_len,
        statistical_margin=statistical_margin,
        security_margin_bits=security_margin_bits,
        max_final_key_len=max_final_key_len,
        min_final_key_len=min_final_key_len,
    )

    alice_to_bob = ClassicalChannel(
        channel_id="alice_to_bob",
        length_m=channel_length_m,
        loss_probability=0.0,
        session_id=session_id,
    )
    bob_to_alice = ClassicalChannel(
        channel_id="bob_to_alice",
        length_m=channel_length_m,
        loss_probability=0.0,
        session_id=session_id,
    )

    connect_ports(
        alice.classical.out_port("to_bob"),
        alice_to_bob.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
    )
    connect_ports(
        alice_to_bob.output_port,
        bob.classical.in_port("from_alice"),
        target_action=AGENT_MESSAGE,
    )
    connect_ports(
        bob.classical.out_port("to_alice"),
        bob_to_alice.input_port,
        target_action=ACTION_TRANSMIT_CLASSICAL,
    )
    connect_ports(
        bob_to_alice.output_port,
        alice.classical.in_port("from_bob"),
        target_action=AGENT_MESSAGE,
    )

    network = Network(topology_id="bb84-postprocessing-topology")
    alice_node = Node("alice-node")
    alice_node.add_agent(alice)
    bob_node = Node("bob-node")
    bob_node.add_agent(bob)
    link_node = Node("classical-links")
    link_node.add_device("alice_to_bob", alice_to_bob)
    link_node.add_device("bob_to_alice", bob_to_alice)
    network.add_node(alice_node)
    network.add_node(bob_node)
    network.add_node(link_node)

    runtime = SessionRuntime(
        timeline=timeline,
        network=network,
        session_id=session_id,
    )
    runtime.run()

    return DemoResult(
        alice=alice,
        bob=bob,
        timeline=timeline,
        alice_to_bob=alice_to_bob,
        bob_to_alice=bob_to_alice,
    )


def format_summary(result: DemoResult) -> str:
    """Return a compact human-readable report for the command-line demo."""
    alice = result.alice
    bob = result.bob
    if (
        alice.aborted_reason
        or bob.aborted_reason
        or not (alice.complete and bob.complete)
    ):
        return "\n".join(
            [
                "BB84 event-based post-processing demo",
                "------------------------------------",
                "aborted: true",
                f"alice reason: {alice.aborted_reason}",
                f"bob reason: {bob.aborted_reason}",
                f"events executed: {result.timeline.events_executed}",
            ]
        )

    key_match = alice.final_key == bob.final_key and alice.final_key is not None
    final_key_len = 0 if bob.final_key is None else len(bob.final_key)
    privacy_seed_bits = bob.privacy_seed_bits
    classical_distance_km = result.alice_to_bob.length_m / 1000
    return "\n".join(
        [
            "BB84 event-based post-processing demo",
            "------------------------------------",
            f"raw bits:                       {len(alice.bits):4d}",
            f"sifted bits:                    {len(alice.sifted_bits):4d}",
            f"matched detections rejected:    {len(bob.missed_detection_indices):4d}",
            f"parameter sample bits:          {alice.sample_size:4d}",
            f"estimated qber:               {bob.estimated_qber:0.4f}",
            f"qber threshold:               {bob.qber_abort_threshold:0.4f}",
            f"remaining bits after sample:    {len(bob.reconciled_bits):4d}",
            "",
            f"cascade passes:                 {bob.cascade_passes:4d}",
            "cascade pass indexing:       0-based",
            f"cascade first block size:       {bob.cascade_first_block_size:4d}",
            f"cascade parity requests:        {bob.cascade_parity_requests:4d}",
            f"cascade corrections:            {bob.cascade_corrections:4d}",
            f"cascade leaked bits:            {bob.cascade_leaked_bits:4d}",
            "",
            "verification hash:          Toeplitz",
            f"verification tag bits:          {bob.verification_tag_len:4d}",
            f"verification seed bits:         "
            f"{len(bob.reconciled_bits) + bob.verification_tag_len - 1:4d}",
            "",
            f"demo phase-error bound:       {bob.demo_phase_error_bound:0.4f}",
            f"demo Eve-info bound bits:       {bob.demo_eve_info_bound:4d}",
            f"privacy security margin bits:   {bob.security_margin_bits:4d}",
            f"demo entropy budget bits:       {bob.demo_entropy_budget_bits:4d}",
            "budget model:       demo entropy budget, not finite-key proof",
            "",
            "privacy hash:              Toeplitz",
            f"toeplitz privacy seed bits:     {privacy_seed_bits:4d}",
            f"final key bits:                 {final_key_len:4d}",
            "",
            f"alice final key == bob final key: {str(key_match).lower()}",
            f"timeline events scheduled:      {result.timeline.events_scheduled:4d}",
            f"timeline events executed:       {result.timeline.events_executed:4d}",
            f"final simulation time ticks:    {result.timeline.current_time}",
            (
                "classical channel length:      "
                f"{classical_distance_km:6.1f} km each direction"
            ),
        ]
    )


def parse_sift_positions(text: str) -> tuple[int, ...]:
    """Parse comma-separated sifted-key error positions for the CLI."""
    stripped = text.strip()
    if stripped.lower() in {"", "none", "off"}:
        return ()
    positions: list[int] = []
    for part in stripped.split(","):
        value = part.strip()
        if not value:
            continue
        positions.append(int(value))
    return tuple(positions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the event-based BB84 post-processing demo."
    )
    parser.add_argument("--raw-bits", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--distance-km",
        type=float,
        default=20.0,
        help="classical channel length in each direction",
    )
    parser.add_argument(
        "--error-sift-positions",
        type=parse_sift_positions,
        default=(17,),
        help=(
            "comma-separated sifted-key positions where Bob has an injected "
            "error; use 'none' for no injected errors"
        ),
    )
    parser.add_argument(
        "--missed-detection-count",
        type=int,
        default=None,
        help=(
            "number of matching-basis detections to mark missing; default "
            "scales with raw bits"
        ),
    )
    parser.add_argument("--qber-threshold", type=float, default=0.11)
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--min-sample-bits", type=int, default=8)
    parser.add_argument("--min-remaining-bits", type=int, default=64)
    parser.add_argument("--cascade-passes", type=int, default=4)
    parser.add_argument("--verification-tag-len", type=int, default=64)
    parser.add_argument("--statistical-margin", type=float, default=0.02)
    parser.add_argument("--security-margin-bits", type=int, default=32)
    parser.add_argument("--max-final-key-len", type=int, default=64)
    parser.add_argument("--min-final-key-len", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_demo(
        master_seed=args.seed,
        raw_bits=args.raw_bits,
        distance_km=args.distance_km,
        error_sift_positions=args.error_sift_positions,
        missed_detection_count=args.missed_detection_count,
        qber_abort_threshold=args.qber_threshold,
        sample_fraction=args.sample_fraction,
        min_sample_bits=args.min_sample_bits,
        min_remaining_bits=args.min_remaining_bits,
        cascade_passes=args.cascade_passes,
        verification_tag_len=args.verification_tag_len,
        statistical_margin=args.statistical_margin,
        security_margin_bits=args.security_margin_bits,
        max_final_key_len=args.max_final_key_len,
        min_final_key_len=args.min_final_key_len,
    )
    print(format_summary(result))


if __name__ == "__main__":
    main()
