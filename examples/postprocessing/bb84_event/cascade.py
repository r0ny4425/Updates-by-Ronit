"""Cascade reconciliation state for the BB84 post-processing example."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .helpers import choose_permutation, parity


@dataclass(slots=True)
class CascadeBlock:
    """One block or search region waiting for a parity query."""

    pass_index: int
    block_id: int
    indices: tuple[int, ...]
    phase: str
    search_indices: tuple[int, ...] | None = None
    depth: int = 0


@dataclass(slots=True)
class CascadeOutstanding:
    """Parity request that has been sent but not answered yet."""

    request_id: str
    block: CascadeBlock
    queried_indices: tuple[int, ...]
    search_indices: tuple[int, ...]


@dataclass(slots=True)
class CascadeParityRequest:
    """Serializable parity question sent from Bob to Alice."""

    request_id: str
    pass_index: int
    block_id: int
    phase: str
    indices: tuple[int, ...]
    search_indices: tuple[int, ...]
    depth: int

    def as_body(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "pass_index": self.pass_index,
            "block_id": self.block_id,
            "phase": self.phase,
            "indices": list(self.indices),
            "search_indices": list(self.search_indices),
            "depth": self.depth,
        }


@dataclass(slots=True)
class CascadeState:
    """Mutable reconciliation state owned by ``CascadeController``."""

    passes: int
    current_pass: int
    first_block_size: int
    pending: list[CascadeBlock]
    previous_pass_blocks: list[list[tuple[int, ...]]]
    outstanding: CascadeOutstanding | None = None
    request_counter: int = 0
    corrections: int = 0
    parity_requests: int = 0
    leaked_bits: int = 0
    queued_block_keys: set[tuple[int, tuple[int, ...], str, tuple[int, ...] | None]] = (
        field(default_factory=set)
    )


def cascade_block_key(
    block: CascadeBlock,
) -> tuple[int, tuple[int, ...], str, tuple[int, ...] | None]:
    """Return a de-duplication key for queued Cascade work."""
    search_indices = (
        tuple(block.search_indices)
        if block.phase == "binary_search" and block.search_indices is not None
        else None
    )
    return (block.pass_index, block.indices, block.phase, search_indices)


_cascade_block_key = cascade_block_key


class CascadeController:
    """State machine for interactive Cascade reconciliation.

    The controller owns only reconciliation mechanics. Callers remain
    responsible for transporting parity requests/responses over whatever event
    or message system they use.

    This implementation is intentionally compact for teaching. It demonstrates
    block checks, binary search, backtracking, and leakage counting. It is not
    a tuned reconciliation library for every block schedule and error pattern.
    """

    def __init__(
        self,
        *,
        bits: list[int],
        rng: Any,
        passes: int,
        first_block_size: int,
    ) -> None:
        if passes <= 0:
            raise ValueError("Cascade passes must be positive")
        if first_block_size <= 0:
            raise ValueError("Cascade first_block_size must be positive")
        if not bits:
            raise ValueError("Cascade bits must not be empty")

        self.bits = bits
        self.rng = rng
        self.state = CascadeState(
            passes=passes,
            current_pass=0,
            first_block_size=min(len(bits), first_block_size),
            pending=[],
            previous_pass_blocks=[],
        )
        self.complete = False
        self._start_pass(pass_index=0)

    @property
    def parity_requests(self) -> int:
        return self.state.parity_requests

    @property
    def corrections(self) -> int:
        return self.state.corrections

    @property
    def leaked_bits(self) -> int:
        return self.state.leaked_bits

    def next_request(self) -> CascadeParityRequest | None:
        """Return the next parity request, or ``None`` while no request is ready."""
        state = self.state
        if self.complete or state.outstanding is not None:
            return None

        if not state.pending:
            if state.current_pass + 1 < state.passes:
                self._start_pass(pass_index=state.current_pass + 1)
                return None

            self.complete = True
            return None

        block = state.pending.pop(0)
        state.queued_block_keys.discard(cascade_block_key(block))

        query_indices = self._query_indices(block)
        if not query_indices:
            return None

        request_id = f"cascade-{state.request_counter:06d}"
        state.request_counter += 1
        search_indices = block.search_indices or block.indices
        state.outstanding = CascadeOutstanding(
            request_id=request_id,
            block=block,
            queried_indices=query_indices,
            search_indices=search_indices,
        )

        return CascadeParityRequest(
            request_id=request_id,
            pass_index=block.pass_index,
            block_id=block.block_id,
            phase=block.phase,
            indices=query_indices,
            search_indices=search_indices,
            depth=block.depth,
        )

    def apply_parity_response(self, *, request_id: str, alice_parity: int) -> None:
        """Apply Alice's parity answer and update Bob's local bit estimate."""
        state = self.state
        outstanding = state.outstanding

        if outstanding is None:
            raise RuntimeError("received Cascade parity response without request")
        if request_id != outstanding.request_id:
            raise ValueError("Cascade response request_id mismatch")

        state.outstanding = None
        state.leaked_bits += 1
        state.parity_requests += 1

        bob_parity = parity(self.bits, outstanding.queried_indices)
        mismatched = bob_parity != int(alice_parity)
        block = outstanding.block

        if block.phase in ("block_check", "backtrack_check"):
            if mismatched:
                self._continue_search_after_mismatch(block, block.indices)
        elif block.phase == "binary_search":
            self._resolve_binary_step(
                block,
                outstanding.search_indices,
                left_mismatched=mismatched,
            )
        else:
            raise ValueError(f"unsupported Cascade block phase: {block.phase!r}")

    def _start_pass(self, *, pass_index: int) -> None:
        """Shuffle key indices, split them into blocks, and queue checks."""
        state = self.state
        state.current_pass = pass_index
        blocks = self._make_pass_blocks(pass_index=pass_index)
        state.previous_pass_blocks.append(blocks)

        for block_id, indices in enumerate(blocks):
            self._enqueue_block(
                CascadeBlock(
                    pass_index=pass_index,
                    block_id=block_id,
                    indices=indices,
                    phase="block_check",
                    search_indices=indices,
                )
            )

    def _make_pass_blocks(self, *, pass_index: int) -> list[tuple[int, ...]]:
        n = len(self.bits)
        block_size = min(n, self.state.first_block_size * (2**pass_index))
        permutation = choose_permutation(self.rng, n)
        return [
            tuple(permutation[start : start + block_size])
            for start in range(0, n, block_size)
        ]

    def _enqueue_block(self, block: CascadeBlock, *, front: bool = False) -> None:
        """Queue work once; duplicate backtracking checks are ignored."""
        state = self.state
        key = cascade_block_key(block)
        if key in state.queued_block_keys:
            return
        state.queued_block_keys.add(key)
        if front:
            state.pending.insert(0, block)
        else:
            state.pending.append(block)

    def _query_indices(self, block: CascadeBlock) -> tuple[int, ...]:
        """Choose the exact indices for the next public parity question."""
        if block.phase in ("block_check", "backtrack_check"):
            return block.indices
        if block.phase == "binary_search":
            search = block.search_indices or block.indices
            return search[: len(search) // 2]
        raise ValueError(f"unsupported Cascade block phase: {block.phase!r}")

    def _continue_search_after_mismatch(
        self,
        block: CascadeBlock,
        search_indices: tuple[int, ...],
    ) -> None:
        if len(search_indices) == 1:
            self._flip_and_backtrack(search_indices[0])
            return

        self._enqueue_block(
            CascadeBlock(
                pass_index=block.pass_index,
                block_id=block.block_id,
                indices=block.indices,
                phase="binary_search",
                search_indices=search_indices,
                depth=block.depth + 1,
            ),
            front=True,
        )

    def _resolve_binary_step(
        self,
        block: CascadeBlock,
        search_indices: tuple[int, ...],
        *,
        left_mismatched: bool,
    ) -> None:
        mid = len(search_indices) // 2
        left = search_indices[:mid]
        right = search_indices[mid:]
        next_search = left if left_mismatched else right

        if len(next_search) == 1:
            self._flip_and_backtrack(next_search[0])
            return

        self._enqueue_block(
            CascadeBlock(
                pass_index=block.pass_index,
                block_id=block.block_id,
                indices=block.indices,
                phase="binary_search",
                search_indices=next_search,
                depth=block.depth + 1,
            ),
            front=True,
        )

    def _flip_and_backtrack(self, corrected_index: int) -> None:
        """Flip one corrected bit and queue older blocks that used that bit."""
        self.bits[corrected_index] ^= 1
        self.state.corrections += 1
        self._enqueue_backtracking_blocks(corrected_index)

    def _enqueue_backtracking_blocks(self, corrected_index: int) -> None:
        state = self.state
        for pass_index, blocks in enumerate(state.previous_pass_blocks):
            if pass_index >= state.current_pass:
                continue

            for block_id, indices in enumerate(blocks):
                if corrected_index not in indices:
                    continue

                self._enqueue_block(
                    CascadeBlock(
                        pass_index=pass_index,
                        block_id=block_id,
                        indices=indices,
                        phase="backtrack_check",
                        search_indices=indices,
                    )
                )
