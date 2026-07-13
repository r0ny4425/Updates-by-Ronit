"""Generic control-plane primitives for simulator agents.

The package exports the small public surface needed to build event-driven
control agents: action names, ``Agent``/``NodeAgent`` base classes,
``AgentContext``, and ``SessionRuntime``. Lower-level endpoint and service
modules remain importable from their concrete submodules when needed.
"""

from __future__ import annotations

from .actions import AGENT_EVENT, AGENT_MESSAGE, AGENT_REPORT, AGENT_START, AGENT_TIMER
from .agent import Agent, NodeAgent
from .context import AgentContext
from .runtime import SessionRuntime

__all__ = [
    "AGENT_EVENT",
    "AGENT_MESSAGE",
    "AGENT_REPORT",
    "AGENT_START",
    "AGENT_TIMER",
    "Agent",
    "AgentContext",
    "NodeAgent",
    "SessionRuntime",
]
