"""Timer scheduling service for generic control-plane agents.

Timers are ordinary timeline events targeted back to the owning agent with an
``AGENT_TIMER`` action and a ``TimerFired`` payload. The service does not keep a
separate timer registry; cancellation uses the returned ``Event``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import Meta, validate_meta
from simyuj.primitives.validation import validate_non_negative_int

from .actions import AGENT_TIMER
from .payloads import TimerFired

if TYPE_CHECKING:
    from .agent import Agent


class TimerService:
    """Schedule timer events directly to one owning agent.

    Parameters
    ----------
    owner_agent : Agent
        Agent component that will receive timer callbacks.
    timeline : Timeline
        Timeline used to schedule timer events.
    session_id : str
        Non-empty runtime session identifier stored in payload and event
        metadata.
    priority : int, default=0
        Event priority used for all timers scheduled by this service.
    """

    def __init__(
        self,
        *,
        owner_agent: Agent,
        timeline: Timeline,
        session_id: str,
        priority: int = 0,
    ) -> None:
        if not isinstance(owner_agent, Component):
            raise TypeError("owner_agent must be Component")
        agent_id = getattr(owner_agent, "agent_id", None)
        if not isinstance(agent_id, str) or agent_id == "":
            raise ValueError("owner_agent must expose non-empty agent_id")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        if type(priority) is not int:
            raise TypeError("priority must be int")

        self._owner_agent = owner_agent
        self._timeline = timeline
        self._session_id = ensure_nonempty_id(session_id, field_name="session_id")
        self._priority = priority
        self._active: dict[str, Event] = {}

    def _get_active(self, timer_id: str) -> Event | None:
        event = self._active.get(timer_id)
        if event is None:
            return None

        if event.cancelled or event.time <= self._timeline.current_time:
            del self._active[timer_id]
            return None

        return event

    def set(
        self,
        timer_id: str,
        delay: int,
        *,
        replace: bool = False,
        set_once: bool = False,
        correlation_id: str | int | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a timer ``delay`` ticks from the current timeline time.

        Parameters
        ----------
        timer_id : str
            Non-empty local timer name.
        delay : int
            Non-negative number of ticks after ``timeline.current_time``.
        replace : bool, default=False
            If True and the timer is already active, cancel the existing timer
            before scheduling the new one.
        set_once : bool, default=False
            If True and the timer is already active, return the existing timer
            without scheduling a new one.
        correlation_id : str or int or None, default=None
            Optional value copied into the ``TimerFired`` payload.
        meta : tuple[tuple[str, object], ...], default=()
            Hashable metadata copied into the ``TimerFired`` payload.

        Returns
        -------
        Event
            Scheduled ``AGENT_TIMER`` event.

        Notes
        -----
        ``delay=0`` schedules a same-tick timeline event; it is not an
        immediate callback.
        """
        validate_non_negative_int(delay, field_name="delay")
        return self.at(
            timer_id,
            self._timeline.current_time + delay,
            replace=replace,
            set_once=set_once,
            correlation_id=correlation_id,
            meta=meta,
        )

    def at(
        self,
        timer_id: str,
        time: int,
        *,
        replace: bool = False,
        set_once: bool = False,
        correlation_id: str | int | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a timer at an absolute timeline tick.

        Parameters
        ----------
        timer_id : str
            Non-empty local timer name.
        time : int
            Absolute timeline tick when the timer should fire. It must be
            greater than or equal to ``timeline.current_time``.
        replace : bool, default=False
            If True and the timer is already active, cancel the existing timer
            before scheduling the new one.
        set_once : bool, default=False
            If True and the timer is already active, return the existing timer
            without scheduling a new one.
        correlation_id : str or int or None, default=None
            Optional value copied into the ``TimerFired`` payload.
        meta : tuple[tuple[str, object], ...], default=()
            Hashable metadata copied into the ``TimerFired`` payload.

        Returns
        -------
        Event
            Scheduled ``AGENT_TIMER`` event targeted to ``owner_agent``.

        Notes
        -----
        ``scheduled_at`` is captured from ``timeline.current_time`` when this
        method is called. The method schedules through ``Timeline.schedule`` and
        does not call the agent hook directly. Past times are rejected by the
        ``TimerFired`` payload invariant that ``fires_at >= scheduled_at``.
        """
        if replace and set_once:
            raise ValueError("cannot specify both replace=True and set_once=True")

        ensure_nonempty_id(timer_id, field_name="timer_id")
        validate_non_negative_int(time, field_name="time")
        validate_meta(meta)

        active = self._get_active(timer_id)
        if active is not None:
            if set_once:
                return active
            if replace:
                self._timeline.cancel(active)

        event = self._timeline.schedule(
            Event(
                time=time,
                priority=self._priority,
                target_ref=self._owner_agent,
                action=AGENT_TIMER,
                payload_ref=TimerFired(
                    timer_id=timer_id,
                    owner_agent_id=self._owner_agent.agent_id,
                    scheduled_at=self._timeline.current_time,
                    fires_at=time,
                    correlation_id=correlation_id,
                    meta=meta,
                ),
                source=self._owner_agent,
                subsystem_id="control",
                meta={
                    "session_id": self._session_id,
                    "agent_id": self._owner_agent.agent_id,
                    "timer_id": timer_id,
                },
            )
        )
        self._active[timer_id] = event
        return event

    def cancel(self, timer: str | Event) -> None:
        """Cancel a previously scheduled timer event through the timeline.

        Parameters
        ----------
        timer : str or Event
            The logical timer id string to cancel, or the specific ``Event``
            record returned by a previous schedule call.
        """
        if isinstance(timer, str):
            active = self._get_active(timer)
            if active is not None:
                self._timeline.cancel(active)
                del self._active[timer]
        elif isinstance(timer, Event):
            self._timeline.cancel(timer)
        else:
            raise TypeError("timer must be str or Event")


__all__ = ["TimerService"]
