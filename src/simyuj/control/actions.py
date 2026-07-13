"""Shared event action names for the generic control plane.

The constants in this module are string actions carried by
``simyuj.engine.Event`` records. They are intentionally small and
protocol-neutral so agents, endpoints, services, and tests can agree on the
same scheduled event vocabulary.
"""

AGENT_START = "agent_start"
AGENT_TIMER = "agent_timer"
AGENT_MESSAGE = "agent_message"
AGENT_REPORT = "agent_report"
AGENT_EVENT = "agent_event"

__all__ = [
    "AGENT_EVENT",
    "AGENT_MESSAGE",
    "AGENT_REPORT",
    "AGENT_START",
    "AGENT_TIMER",
]
