from __future__ import annotations

import random

import pytest

from examples.postprocessing.bb84_event.cascade import CascadeController
from examples.postprocessing.bb84_event.helpers import parity


class SeededRng:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)


def run_cascade(
    alice_bits: list[int],
    bob_bits: list[int],
    *,
    seed: int = 17,
    passes: int = 4,
    first_block_size: int = 4,
) -> CascadeController:
    controller = CascadeController(
        bits=bob_bits,
        rng=SeededRng(seed),
        passes=passes,
        first_block_size=first_block_size,
    )

    for _ in range(10_000):
        if controller.complete:
            return controller
        request = controller.next_request()
        if request is None:
            continue
        controller.apply_parity_response(
            request_id=request.request_id,
            alice_parity=parity(alice_bits, request.indices),
        )

    raise AssertionError("Cascade did not complete")


def test_cascade_corrects_single_bit_error():
    alice_bits = [index & 1 for index in range(32)]
    bob_bits = alice_bits.copy()
    bob_bits[13] ^= 1

    controller = run_cascade(alice_bits, bob_bits)

    assert controller.complete
    assert bob_bits == alice_bits
    assert controller.corrections == 1
    assert controller.leaked_bits == controller.parity_requests


def test_cascade_no_error_does_not_flip_bits():
    alice_bits = [((index * 3) + 1) & 1 for index in range(32)]
    bob_bits = alice_bits.copy()

    controller = run_cascade(alice_bits, bob_bits)

    assert controller.complete
    assert bob_bits == alice_bits
    assert controller.corrections == 0
    assert controller.leaked_bits == controller.parity_requests


def test_cascade_continues_binary_search_before_other_pending_blocks():
    alice_bits = [0] * 24
    bob_bits = alice_bits.copy()
    controller = CascadeController(
        bits=bob_bits,
        rng=SeededRng(3),
        passes=1,
        first_block_size=8,
    )

    first_check = controller.next_request()
    assert first_check is not None
    bob_bits[first_check.indices[0]] ^= 1
    controller.apply_parity_response(
        request_id=first_check.request_id,
        alice_parity=parity(alice_bits, first_check.indices),
    )

    first_search = controller.next_request()
    assert first_search is not None
    assert first_search.phase == "binary_search"
    assert first_search.block_id == first_check.block_id
    controller.apply_parity_response(
        request_id=first_search.request_id,
        alice_parity=parity(alice_bits, first_search.indices),
    )

    continued_search = controller.next_request()
    assert continued_search is not None
    assert continued_search.phase == "binary_search"
    assert continued_search.block_id == first_check.block_id
    assert continued_search.depth == first_search.depth + 1


def test_cascade_response_requires_outstanding_request():
    controller = CascadeController(
        bits=[0, 1, 0, 1],
        rng=SeededRng(1),
        passes=1,
        first_block_size=2,
    )

    with pytest.raises(RuntimeError, match="without request"):
        controller.apply_parity_response(
            request_id="cascade-000000",
            alice_parity=0,
        )


def test_cascade_rejects_mismatched_request_id():
    controller = CascadeController(
        bits=[0, 1, 0, 1],
        rng=SeededRng(1),
        passes=1,
        first_block_size=2,
    )
    request = controller.next_request()
    assert request is not None

    with pytest.raises(ValueError, match="request_id mismatch"):
        controller.apply_parity_response(
            request_id="wrong-request-id",
            alice_parity=0,
        )

    controller.apply_parity_response(
        request_id=request.request_id,
        alice_parity=parity([0, 1, 0, 1], request.indices),
    )


def test_cascade_rejects_empty_bits():
    with pytest.raises(ValueError, match="must not be empty"):
        CascadeController(
            bits=[],
            rng=SeededRng(1),
            passes=1,
            first_block_size=2,
        )
