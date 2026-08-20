"""Small pure functions for the DPS-QKD example."""

from __future__ import annotations

from collections.abc import Sequence

from simyuj.primitives.units import seconds_to_ticks

from .configs import DPS_ENCODING_PHASES


def dps_slot_period_ticks(clock_hz: float) -> int:
    """Return the pulse slot period in simulation ticks."""
    return int(seconds_to_ticks(1.0 / float(clock_hz)))


def dps_source_duration_s(*, clock_hz: float, num_slots: int) -> float:
    """Return the source-active duration that emits exactly ``num_slots`` pulses.

    Notes
    -----
    The source's stop tick is **exclusive** and both the nominal slot and the
    delayed emission must fall before it, so a duration of exactly
    ``num_slots / clock_hz`` yields ``num_slots`` pulses under a zero-delay
    timing profile.
    """
    if num_slots < 1:
        raise ValueError("num_slots must be at least 1")
    return float(num_slots) / float(clock_hz)


def dps_differential_bit(index_prev: int, index_curr: int) -> int:
    """Decode one DPS bit from two adjacent encoding-phase alphabet indices.

    Parameters
    ----------
    index_prev, index_curr : int
        ``encoding_phase_index`` values from two adjacent
        ``CoherentPulsePreparationReport`` records.

    Returns
    -------
    int
        ``0`` when the two pulses share a phase, ``1`` when they differ by
        ``pi``.

    Notes
    -----
    **Indices, not radians.** A floating-point phase must never enter a bit
    decision: the amplitude carries the wrapped *sum* of the carrier and
    encoding phases, and after channel phase noise it diverges further. The
    alphabet position is the only thing a message conveyed.

    This is defined against ``DPS_ENCODING_PHASES``, where index ``0`` is phase
    ``0`` and index ``1`` is phase ``pi``. Reordering that constant silently
    inverts every bit this function returns; see the comment where it is
    defined.
    """
    if index_prev not in (0, 1) or index_curr not in (0, 1):
        raise ValueError(
            "DPS encoding phase indices must be 0 or 1 for a two-phase alphabet"
        )
    return index_prev ^ index_curr


def dps_differential_bits(indices: Sequence[int]) -> tuple[int, ...]:
    """Decode the differential bit sequence for a whole pulse train.

    Notes
    -----
    ``n`` pulses give ``n - 1`` differential bits: the first pulse has no
    predecessor to pair with. A receiver's interferometer produces ``n + 1``
    output slots, of which the first and last pair a real pulse with vacuum and
    carry no bit; dropping those is the receiving agent's job at step 6, not
    this function's.
    """
    return tuple(
        dps_differential_bit(previous, current)
        for previous, current in zip(indices, indices[1:])
    )


def dps_phase_histogram(indices: Sequence[int]) -> tuple[int, ...]:
    """Return the count of each alphabet index in ``indices``."""
    counts = [0] * len(DPS_ENCODING_PHASES)
    for index in indices:
        counts[index] += 1
    return tuple(counts)


__all__ = [
    "dps_differential_bit",
    "dps_differential_bits",
    "dps_phase_histogram",
    "dps_slot_period_ticks",
    "dps_source_duration_s",
]
