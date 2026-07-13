"""Deterministic event timeline for simulation execution.

``Timeline`` owns event identifiers, current simulation time, batch execution,
lazy cancellation behavior, deterministic RNG streams, qstate access, and
observational logging. Components participate by receiving events and scheduling
follow-up events through the timeline.
"""

from __future__ import annotations

from typing import List, Optional

from simyuj.qstate.manager import QuantumStateManager
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import NullLogger, SimulationLogger
from simyuj.tracing.records import MetaInput

from .event import Event
from .event_ordering import event_ordering_key
from .event_queue import EventQueue
from .execution_summary import ExecutionSummary
from .rng_manager import DeterministicRNG, RNGManager
from .timeline_statistics import TimelineStatistics


class Timeline:
    """Single authority for deterministic event execution.

    The timeline assigns event ids, advances integer simulation time, extracts
    same-time batches, dispatches events to targets, freezes RNG stream creation
    when execution begins, and records execution statistics. It intentionally
    keeps protocol and component semantics outside the engine core.

    Parameters
    ----------
    master_seed : int, default=42
        Root seed for deterministic named RNG streams.
    logger : SimulationLogger, optional
        Logger used for observational records. If omitted, a ``NullLogger`` is
        used.
    qstate_noise_mode : str, default="density"
        Manager-level policy for qstate noise models. ``"density"`` keeps exact
        mixed-state evolution. ``"sampled_ket"`` samples Kraus branches for ket
        records using the timeline-owned ``qstate/noise`` RNG stream.

    Notes
    -----
    Event execution is single-threaded. Components must not mutate
    ``current_time`` or directly invoke other components' handlers; they should
    schedule new events instead.

    Examples
    --------
    >>> from simyuj.engine import Event, Timeline
    >>>
    >>> class Recorder:
    ...     def __init__(self):
    ...         self.seen = []
    ...
    ...     def handle_event(self, event, timeline):
    ...         self.seen.append((timeline.current_time, event.action))
    ...         if event.action == "ping":
    ...             _ = timeline.schedule(
    ...                 Event(
    ...                     time=timeline.current_time + 1,
    ...                     target_ref=self,
    ...                     action="pong",
    ...                     payload_ref=None,
    ...                 )
    ...             )
    >>>
    >>> target = Recorder()
    >>> timeline = Timeline()
    >>> _ = timeline.schedule(
    ...     Event(time=5, target_ref=target, action="ping", payload_ref=None)
    ... )
    >>> summaries = timeline.run_until_empty()
    >>> target.seen
    [(5, 'ping'), (6, 'pong')]
    >>> [summary.batch_time for summary in summaries]
    [5, 6]
    """

    def __init__(
        self,
        *,
        master_seed: int = 42,
        logger: SimulationLogger | None = None,
        qstate_noise_mode: str = "density",
    ) -> None:
        self._queue = EventQueue()
        self._next_event_id: int = 0
        self._current_time: int = 0  # time is represented via integer ticks
        self._executing: bool = False
        # Generate unique owner ID for this timeline instance
        self._timeline_id = f"timeline_{id(self)}"
        self._logger: SimulationLogger = logger if logger is not None else NullLogger()
        # Initialize RNG manager with both master_seed and owner_id
        self._rng_frozen: bool = False
        self._rng_manager = RNGManager(
            master_seed=master_seed, owner_id=self._timeline_id
        )
        qstate_noise_rng = None
        if isinstance(qstate_noise_mode, str):
            if qstate_noise_mode.strip().lower() == "sampled_ket":
                qstate_noise_rng = self.rng("qstate", "noise")
        self._qstate = QuantumStateManager(
            noise_mode=qstate_noise_mode,
            noise_rng=qstate_noise_rng,
        )
        # ─────────────Statistics tracking───────────────────────
        self._stats_total_events_scheduled: int = 0
        self._stats_total_events_executed: int = 0
        self._stats_max_queue_size: int = 0

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────
    def _allocate_event_id(self) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    def _update_max_queue_size(self) -> None:
        """Update the maximum raw queue size statistic.

        Notes
        -----
        This is observational bookkeeping. It does not affect event ordering or
        execution.
        """
        size = len(self._queue)
        if size > self._stats_max_queue_size:
            self._stats_max_queue_size = size

    def _has_active_events(self) -> bool:
        """
        Return whether the queue has any executable events.

        EventQueue.peek() performs lazy cancellation cleanup before reporting
        exhaustion, so this reflects active events rather than raw heap length.
        """
        try:
            self._queue.peek()
        except IndexError:
            return False
        return True

    @staticmethod
    def _ref_name(ref: object | None) -> str | None:
        """Return a stable class-name label for log context."""
        if ref is None:
            return None
        return type(ref).__name__

    def log(
        self,
        level: LogLevel,
        category: str,
        message: str,
        *,
        event: Event | None = None,
        sim_time: int | None = None,
        event_id: int | None = None,
        action: str | None = None,
        target_name: str | None = None,
        source_name: str | None = None,
        node_id: str | None = None,
        link_id: str | None = None,
        meta: MetaInput = None,
    ) -> None:
        """Emit an observational log record through the timeline logger.

        Parameters
        ----------
        level : LogLevel
            Minimum log level required for emission.
        category : str
            Structured category label.
        message : str
            Human-readable log message.
        event : Event, optional
            Event used to fill missing ``event_id``, ``action``,
            ``target_name``, and ``source_name`` fields.
        sim_time : int, optional
            Simulation time for the record. Defaults to ``current_time``.
        event_id, action, target_name, source_name : optional
            Explicit event context values. Values supplied here override fields
            inferred from ``event``.
        node_id, link_id : str, optional
            Optional topology context.
        meta : MetaInput, optional
            Structured metadata passed to the logger.

        Notes
        -----
        Logging is observational. If the configured logger is disabled for
        ``level``, this method returns without constructing a record.
        """
        if not self._logger.is_enabled(level):
            return

        if event is not None:
            if event_id is None:
                event_id = event.event_id
            if action is None:
                action = event.action
            if target_name is None:
                target_name = self._ref_name(event.target_ref)
            if source_name is None:
                source_name = self._ref_name(event.source)

        if sim_time is None:
            sim_time = self.current_time

        self._logger.log(
            level=level,
            category=category,
            message=message,
            sim_time=sim_time,
            event_id=event_id,
            action=action,
            target_name=target_name,
            source_name=source_name,
            session_id=self._timeline_id,
            node_id=node_id,
            link_id=link_id,
            meta=meta,
        )

    def schedule(self, event: Event) -> Event:
        """Assign an id to an event and add it to the queue.

        Parameters
        ----------
        event : Event
            Unscheduled event with ``event_id=None``.

        Returns
        -------
        Event
            The same event instance, with a timeline-assigned ``event_id``.

        Raises
        ------
        ValueError
            If the event already has an id or is scheduled before
            ``current_time``.

        Notes
        -----
        Scheduling updates the total scheduled-event count and may update the
        maximum raw queue size. Events scheduled at ``current_time`` are allowed.
        """
        if event.event_id is not None:
            raise ValueError("Cannot schedule Event with pre-assigned event_id")
        if self.current_time > event.time:
            raise ValueError("Cannot schedule events in the past")

        event_id = self._allocate_event_id()
        event._set_event_id(event_id)

        self._queue.push(event)

        # Update statistics after the event has entered the queue.
        self._stats_total_events_scheduled += 1
        self._update_max_queue_size()

        if self._logger.is_enabled(LogLevel.TRACE):
            self.log(
                LogLevel.TRACE,
                "engine.timeline.schedule",
                "event scheduled",
                event=event,
                meta={"scheduled_time": event.time, "priority": event.priority},
            )

        return event

    def cancel(self, event: Event) -> None:
        """Mark an event so the queue will skip it later.

        Parameters
        ----------
        event : Event
            Event to cancel.

        Notes
        -----
        Cancellation is lazy. The event can remain in the queue until
        ``EventQueue.peek`` or ``EventQueue.pop`` reaches it. Cancelling does
        not decrement the scheduled-event statistic.
        """
        event._mark_cancelled()
        if self._logger.is_enabled(LogLevel.TRACE):
            self.log(
                LogLevel.TRACE,
                "engine.timeline.cancel",
                "event cancelled",
                event=event,
            )

    def reschedule(
        self,
        event: Event,
        *,
        new_time: Optional[int] = None,
        new_priority: Optional[int] = None,
    ) -> Event:
        """Cancel an event and schedule a replacement.

        Parameters
        ----------
        event : Event
            Event whose target, action, payload, source, subsystem, and metadata
            should be preserved.
        new_time : int, optional
            Replacement timestamp. Defaults to ``event.time``.
        new_priority : int, optional
            Replacement priority. Defaults to ``event.priority``.

        Returns
        -------
        Event
            Newly scheduled event with a fresh ``event_id``.

        Raises
        ------
        ValueError
            If the replacement event is scheduled in the past.

        Notes
        -----
        The original event remains cancelled. The replacement reuses the same
        ``meta`` object rather than copying it.
        """
        self.cancel(event)

        return self.schedule(
            Event(
                time=new_time if new_time is not None else event.time,
                priority=new_priority if new_priority is not None else event.priority,
                event_id=None,
                target_ref=event.target_ref,
                action=event.action,
                payload_ref=event.payload_ref,
                source=event.source,
                subsystem_id=event.subsystem_id,
                meta=event.meta,
            )
        )

    def pop_batch(self) -> List[Event]:
        """Remove and return the next same-time active event batch.

        Returns
        -------
        List[Event]
            Non-cancelled events sharing the earliest executable timestamp,
            ordered by ``event_ordering_key``.

        Raises
        ------
        IndexError
            If the queue is empty or contains only cancelled events.
        RuntimeError
            If the next batch would execute before ``current_time`` or the
            underlying queue returns events out of order.

        Notes
        -----
        This method extracts a batch but does not advance ``current_time`` and
        does not call event handlers. Events scheduled later at the same time
        become part of a subsequent batch.
        """
        batch: List[Event] = []

        try:
            batch_time = self._queue.peek().time
        except IndexError as exc:
            raise IndexError("pop_batch called on empty EventQueue") from exc

        if batch_time < self.current_time:
            raise RuntimeError(
                f"Timeline invariant violated: batch_time={batch_time} "
                f"< current_time={self.current_time}"
            )

        previous_key: tuple[int, int, int] | None = None
        while True:
            try:
                event = self._queue.peek()
            except IndexError:
                break

            if event.time != batch_time:
                break

            event = self._queue.pop()  # pop already skips cancelled events
            current_key = event_ordering_key(event)
            if previous_key is not None and current_key < previous_key:
                raise RuntimeError(
                    "EventQueue ordering violates Timeline execution semantics"
                )
            previous_key = current_key
            batch.append(event)

        return batch

    def run_one_step(self) -> ExecutionSummary:
        """Execute one batch and return its execution summary.

        Returns
        -------
        ExecutionSummary
            Batch time, executed event count, and executed event ids in
            dispatch order.

        Raises
        ------
        RuntimeError
            If the timeline is already executing, the raw queue is empty, an
            event would execute in the past, a handler mutates
            ``current_time``, or an executed event is missing an id.
        TypeError
            If an event target does not provide ``handle_event``.
        IndexError
            If the queue contains no active events after lazy cancellation.

        Notes
        -----
        The first execution step freezes RNG stream creation. Handler
        exceptions propagate unchanged. The timeline clears its re-entrancy flag
        in a ``finally`` block even when execution fails.

        Execution is not transactional. Once a batch has been popped from the
        queue, events in that batch are not automatically restored if a handler
        raises partway through dispatch.
        """

        event_ids = []
        if self._executing:
            raise RuntimeError("Timeline is already executing")

        if len(self._queue) == 0:
            raise RuntimeError("Cannot execute: Timeline event queue is empty")

        if not self._rng_frozen:
            self._rng_manager.freeze()
            self._rng_frozen = True

        self._executing = True

        try:
            batch = self.pop_batch()

            batch_time = batch[0].time
            if self.current_time > batch_time:
                raise RuntimeError("Cannot execute events in the past.")
            self._current_time = batch_time

            if self._logger.is_enabled(LogLevel.TRACE):
                self.log(
                    LogLevel.TRACE,
                    "engine.timeline.batch_start",
                    "batch started",
                    sim_time=batch_time,
                    meta={"batch_size": len(batch)},
                )

            for event in batch:
                if event.cancelled:
                    continue

                handler = event.target_ref
                if not hasattr(handler, "handle_event"):
                    raise TypeError(
                        f"Event target {handler!r} does not implement handle_event()"
                    )

                prev_time = self._current_time

                handler.handle_event(event, self)

                # Handlers may schedule future events, but time advancement is
                # owned by the timeline.
                if self._current_time != prev_time:
                    raise RuntimeError("Event handler mutated Timeline.current_time")

                if event.event_id is None:
                    raise RuntimeError("Executed event missing event_id")

                event_ids.append(event.event_id)

                self._stats_total_events_executed += 1

                if self._logger.is_enabled(LogLevel.TRACE):
                    self.log(
                        LogLevel.TRACE,
                        "engine.timeline.event_execution",
                        "event executed",
                        event=event,
                        sim_time=batch_time,
                    )

            if self._logger.is_enabled(LogLevel.TRACE):
                self.log(
                    LogLevel.TRACE,
                    "engine.timeline.batch_complete",
                    "batch completed",
                    sim_time=batch_time,
                    meta={"batch_size": len(batch), "num_executed": len(event_ids)},
                )

        finally:
            self._executing = False

        return ExecutionSummary(
            batch_time=batch_time,
            num_executed=len(event_ids),
            event_ids=tuple(event_ids),
        )

    def run_until_empty(self) -> tuple[ExecutionSummary, ...]:
        """Execute batches until the timeline has no active events.

        Returns
        -------
        tuple[ExecutionSummary, ...]
            Summary for each executed batch, in execution order.

        Notes
        -----
        Cancelled events are discarded lazily through ``EventQueue.peek``.
        Handler exceptions and ``run_one_step`` errors propagate unchanged.
        """
        summaries: list[ExecutionSummary] = []
        while self._has_active_events():
            summaries.append(self.run_one_step())
        return tuple(summaries)

    def run_until(self, t_end: int) -> None:
        """Execute whole batches up to a time boundary.

        Parameters
        ----------
        t_end : int
            Inclusive simulation-time boundary. A batch whose timestamp equals
            ``t_end`` executes; the first batch after ``t_end`` remains queued.

        Notes
        -----
        ``run_until(a); run_until(b)`` is equivalent to ``run_until(b)`` for
        ``a <= b``. Calling with ``t_end`` before ``current_time`` is a no-op.
        The method does not type-check ``t_end``.
        """

        if self._logger.is_enabled(LogLevel.INFO):
            self.log(
                LogLevel.INFO,
                "engine.timeline.run_until_start",
                "run_until started",
                meta={"t_end": t_end},
            )

        # The loop performs no state updates after run_one_step().
        # This ensures that if run_one_step() raises, no post-execution
        # control or bookkeeping state is left inconsistent.
        while True:
            if t_end < self.current_time:
                if self._logger.is_enabled(LogLevel.INFO):
                    self.log(
                        LogLevel.INFO,
                        "engine.timeline.run_until_finish",
                        "run_until finished",
                        meta={
                            "t_end": t_end,
                            "reason": "t_end_before_current_time",
                        },
                    )
                return

            try:
                next_batch_time = self._queue.peek().time
            except IndexError:
                if self._logger.is_enabled(LogLevel.INFO):
                    self.log(
                        LogLevel.INFO,
                        "engine.timeline.queue_exhausted",
                        "event queue exhausted",
                        meta={"t_end": t_end},
                    )
                    self.log(
                        LogLevel.INFO,
                        "engine.timeline.run_until_finish",
                        "run_until finished",
                        meta={"t_end": t_end, "reason": "queue_exhausted"},
                    )
                return

            if next_batch_time > t_end:
                if self._logger.is_enabled(LogLevel.INFO):
                    self.log(
                        LogLevel.INFO,
                        "engine.timeline.run_until_finish",
                        "run_until finished",
                        meta={
                            "t_end": t_end,
                            "reason": "t_end_reached",
                            "next_batch_time": next_batch_time,
                        },
                    )
                return

            self.run_one_step()

    def rng(self, *path: str) -> DeterministicRNG:
        """Return a deterministic RNG stream for a named path.

        Parameters
        ----------
        *path : str
            Hierarchical stream path passed to ``RNGManager``.

        Returns
        -------
        DeterministicRNG
            Restricted draw-only RNG wrapper for the path.

        Raises
        ------
        RuntimeError
            If a new stream is requested after execution has begun.
        ValueError
            If the path is invalid.

        Notes
        -----
        Stochastic entities should declare streams before execution begins,
        ideally through the runtime binding lifecycle:

        .. code-block:: python

            bind_many([...], timeline)
            # entity.bind(...) calls timeline.rng(...)

        Stream names are case-sensitive.
        """
        return self._rng_manager.rng(
            *path,
            requester_id=self._timeline_id,  # Use stored timeline ID)
        )

    @property
    def stats(self) -> TimelineStatistics:
        """Return an immutable snapshot of current execution statistics.

        Returns
        -------
        TimelineStatistics
            Read-only statistics record. Reading it has no execution side
            effects.
        """
        return TimelineStatistics(
            total_events_scheduled=self._stats_total_events_scheduled,
            total_events_executed=self._stats_total_events_executed,
            max_queue_size=self._stats_max_queue_size,
            current_time=self.current_time,
        )

    @property
    def current_time(self) -> int:
        """Current simulation time in integer ticks."""
        return self._current_time

    @property
    def qstate(self) -> QuantumStateManager:
        """Timeline-owned qstate manager for new state workflows."""
        return self._qstate

    @property
    def logger(self) -> SimulationLogger:
        """Timeline-owned tracing logger."""
        return self._logger

    @property
    def events_scheduled(self) -> int:
        """Total number of events successfully scheduled."""
        return self._stats_total_events_scheduled

    @property
    def events_executed(self) -> int:
        """Total number of events dispatched to targets."""
        return self._stats_total_events_executed

    @property
    def max_queue_size(self) -> int:
        """Maximum raw event queue size observed after scheduling."""
        return self._stats_max_queue_size
