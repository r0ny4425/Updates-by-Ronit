"""Return records for single-batch timeline execution."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExecutionSummary:
    """Immutable summary returned by ``Timeline.run_one_step``.

    A summary describes one scheduler batch: all non-cancelled events that
    shared the earliest executable timestamp when the step began.

    Attributes
    ----------
    batch_time : int
        Simulation time, in engine ticks, for the executed batch.
    num_executed : int
        Number of events dispatched in the batch.
    event_ids : Tuple[int, ...]
        Event identifiers for dispatched events, in execution order.

    Notes
    -----
    The record is frozen but does not validate types or invariants at
    construction time.
    """

    batch_time: int
    num_executed: int  # number of executed events in the batch
    event_ids: Tuple[int, ...]  # IDs of executed events
