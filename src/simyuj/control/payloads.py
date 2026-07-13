"""Validated payload records for generic agent control events.

The records here are immutable event payloads used by ``Agent`` and
``SessionRuntime``. They validate public boundary fields at construction time
and leave payload-specific objects, such as report bodies, owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.messages.transport import ClassicalMessage
from simyuj.primitives.meta import Meta, validate_meta
from simyuj.primitives.validation import validate_non_negative_int

ControlMeta = Meta


def _validate_correlation_id(value: object, *, field_name: str) -> None:
    if value is not None and not isinstance(value, (str, int)):
        raise TypeError(f"{field_name} must be str, int, or None")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentStart:
    """Payload scheduled when a session runtime starts an agent.

    Parameters
    ----------
    agent_id : str
        Non-empty identifier of the agent receiving the start event.
    session_id : str
        Non-empty identifier for the runtime session.
    node_id : str or None, default=None
        Node identifier where the agent was registered, when available.
    meta : tuple[tuple[str, object], ...], default=()
        Hashable metadata attached by the caller.

    Notes
    -----
    Instances are frozen and slot-backed, so they can be stored as immutable
    event payload records. The constructor validates identifiers and metadata
    shape but does not check that ``agent_id`` or ``node_id`` exists in a
    runtime; ``SessionRuntime`` owns that cross-object validation.
    """

    agent_id: str
    session_id: str
    node_id: str | None = None
    meta: ControlMeta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.agent_id, field_name="agent_id")
        ensure_nonempty_id(self.session_id, field_name="session_id")
        if self.node_id is not None:
            ensure_nonempty_id(self.node_id, field_name="node_id")
        validate_meta(self.meta)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentMessage:
    """Explicit wrapper for delivered classical messages.

    Parameters
    ----------
    message : ClassicalMessage
        The raw transport message.
    receive_time : int
        Timeline tick at which the message arrived.
    source_port_name : str or None, default=None
        Name of the source port, if delivered via connection.
    target_port_name : str or None, default=None
        Name of the target port, if delivered via connection.
    connection_id : str or None, default=None
        Identifier of the connection that scheduled delivery.
    meta : tuple[tuple[str, object], ...], default=()
        Hashable metadata attached to the wrapper.
    """

    message: ClassicalMessage
    receive_time: int
    source_port_name: str | None = None
    target_port_name: str | None = None
    connection_id: str | None = None
    meta: ControlMeta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.message, ClassicalMessage):
            raise TypeError("message must be ClassicalMessage")
        validate_non_negative_int(self.receive_time, field_name="receive_time")
        if self.source_port_name is not None:
            ensure_nonempty_id(self.source_port_name, field_name="source_port_name")
        if self.target_port_name is not None:
            ensure_nonempty_id(self.target_port_name, field_name="target_port_name")
        if self.connection_id is not None:
            ensure_nonempty_id(self.connection_id, field_name="connection_id")
        validate_meta(self.meta)


@dataclass(frozen=True, slots=True, kw_only=True)
class TimerFired:
    """Payload delivered to an agent when one of its timers fires.

    Parameters
    ----------
    timer_id : str
        Non-empty local timer name chosen by the owning agent.
    owner_agent_id : str
        Non-empty identifier of the agent that scheduled the timer.
    scheduled_at : int
        Timeline tick at which the timer request was made. Must be
        non-negative.
    fires_at : int
        Timeline tick at which the timer event is scheduled. Must be greater
        than or equal to ``scheduled_at``.
    correlation_id : str or int or None, default=None
        Optional caller-supplied value for matching timer callbacks to a
        protocol round or request.
    meta : tuple[tuple[str, object], ...], default=()
        Hashable metadata attached to the timer request.

    Notes
    -----
    The payload stores ticks only; cancellation is represented by the scheduled
    ``Event`` object, not by this record.
    """

    timer_id: str
    owner_agent_id: str
    scheduled_at: int
    fires_at: int
    correlation_id: str | int | None = None
    meta: ControlMeta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.timer_id, field_name="timer_id")
        ensure_nonempty_id(self.owner_agent_id, field_name="owner_agent_id")
        validate_non_negative_int(self.scheduled_at, field_name="scheduled_at")
        validate_non_negative_int(self.fires_at, field_name="fires_at")
        if self.fires_at < self.scheduled_at:
            raise ValueError("fires_at must be >= scheduled_at")
        _validate_correlation_id(self.correlation_id, field_name="correlation_id")
        validate_meta(self.meta)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentReport:
    """Explicit wrapper for direct-scheduled agent report events.

    Parameters
    ----------
    report : object
        Report object delivered to ``Agent.on_report`` after endpoint
        extraction. The object is stored by reference.
    source_id : str or None, default=None
        Optional non-empty source identifier, such as a reporting component or
        detector label.
    correlation_id : str or int or None, default=None
        Optional caller-supplied value for matching reports to a request.
    meta : tuple[tuple[str, object], ...], default=()
        Hashable metadata attached to the report wrapper.

    Notes
    -----
    ``AgentReport`` is useful when scheduling an ``AGENT_REPORT`` event without
    a component ``PortDelivery``. ``AgentReportEndpoint.extract`` unwraps the
    record and passes only ``report`` to the agent hook.
    """

    report: object
    source_id: str | None = None
    correlation_id: str | int | None = None
    meta: ControlMeta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.source_id is not None:
            ensure_nonempty_id(self.source_id, field_name="source_id")
        _validate_correlation_id(self.correlation_id, field_name="correlation_id")
        validate_meta(self.meta)


__all__ = [
    "AgentMessage",
    "AgentReport",
    "AgentStart",
    "ControlMeta",
    "TimerFired",
]
