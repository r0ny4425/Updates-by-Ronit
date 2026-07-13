"""Abstract event-target contract for the simulation engine.

The engine dispatches events to objects that expose ``handle_event``. This
module provides the public base class used by simulator components, while the
timeline remains responsible for time advancement, ordering, and scheduling.
"""

from abc import ABC, abstractmethod


class Component(ABC):
    """Base class for objects that receive scheduled engine events.

    A component is an event target. ``Timeline`` dispatches an ``Event`` by
    calling ``handle_event(event, timeline)`` on the target object, and the
    component may schedule future events through the supplied timeline.

    Notes
    -----
    The interface is intentionally small so the engine stays domain-agnostic.
    Components should treat events as read-only causal records, leave time and
    queue ownership to ``Timeline``, and communicate with other components by
    scheduling events rather than by calling their handlers directly.
    """

    @abstractmethod
    def handle_event(self, event, timeline) -> None:
        """Handle one event dispatched by a timeline.

        Parameters
        ----------
        event : Event
            Scheduled event being delivered. Implementations should not mutate
            the event.
        timeline : Timeline
            Timeline that is executing the event. Components use it to schedule
            follow-up events, request deterministic RNG streams, or emit
            simulation logs.

        Notes
        -----
        Implementations are expected to be deterministic under the timeline's
        event ordering and RNG-stream rules.
        """
        pass
