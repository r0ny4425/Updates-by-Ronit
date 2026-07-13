"""Structured records emitted by the tracing logger.

This module provides the immutable record type and metadata normalization used
by tracing loggers and sinks. It is public-facing because users may inspect
records from :class:`simyuj.tracing.sinks.MemorySink` or build records for a
custom sink.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .levels import LogLevel

MetaItems: TypeAlias = tuple[tuple[str, Any], ...]
"""Immutable outer representation for structured log metadata."""

MetaInput: TypeAlias = Mapping[str, Any] | Iterable[tuple[str, Any]] | None
"""Accepted metadata input for logger calls and timeline log helpers."""


def freeze_meta(meta: MetaInput) -> MetaItems:
    """
    Convert metadata into an immutable tuple of key/value pairs.

    Parameters
    ----------
    meta : MetaInput
        ``None``, a mapping, or an iterable of ``(key, value)`` tuples. Keys
        must be strings. Values are accepted as supplied and are not copied.

    Returns
    -------
    MetaItems
        Tuple of ``(key, value)`` pairs in the input iteration order.

    Raises
    ------
    TypeError
        If an iterable entry is not a two-item tuple, or if any key is not a
        string.

    Notes
    -----
    Only the outer container is frozen. Mutable values inside metadata remain
    owned by the caller, so callers should avoid mutating those values after
    emission when reproducible trace inspection matters.
    """
    if meta is None:
        return ()

    if isinstance(meta, Mapping):
        items = tuple(meta.items())
    else:
        items = tuple(meta)

    frozen_items: list[tuple[str, Any]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("meta entries must be (key, value) pairs")
        key, value = item
        if not isinstance(key, str):
            raise TypeError("meta keys must be strings")
        frozen_items.append((key, value))

    return tuple(frozen_items)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationLogRecord:
    """
    Immutable, structured simulation log record.

    Parameters
    ----------
    sequence : int
        Non-negative sequence number assigned by the logger.
    level : LogLevel
        Verbosity level for the record.
    category : str
        Structured category label such as ``"engine.timeline.schedule"``.
    message : str
        Human-readable message.
    sim_time : int, optional
        Integer simulation time tick associated with the record.
    event_id : int, optional
        Timeline-assigned event identifier associated with the record.
    action : str, optional
        Event or component action label.
    target_name, source_name : str, optional
        Human-readable endpoint labels used for trace context.
    session_id : str, optional
        Simulation/session label supplied by the logger or caller.
    node_id, link_id : str, optional
        Optional network topology context.
    meta : MetaItems, optional
        Immutable outer tuple of structured metadata items.

    Notes
    -----
    The record is a frozen dataclass with slot-backed attributes. Validation
    checks the public field types listed above, but it does not require
    ``sim_time`` or ``event_id`` to be non-negative and does not recursively
    validate or freeze metadata values.
    """

    sequence: int
    level: LogLevel
    category: str
    message: str

    sim_time: int | None = None
    event_id: int | None = None
    action: str | None = None
    target_name: str | None = None
    source_name: str | None = None

    session_id: str | None = None
    node_id: str | None = None
    link_id: str | None = None

    meta: MetaItems = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if type(self.sequence) is not int:
            raise TypeError("sequence must be int")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")

        if not isinstance(self.level, LogLevel):
            raise TypeError("level must be a LogLevel")

        if not isinstance(self.category, str):
            raise TypeError("category must be str")
        if not isinstance(self.message, str):
            raise TypeError("message must be str")

        if self.sim_time is not None and type(self.sim_time) is not int:
            raise TypeError("sim_time must be int or None")
        if self.event_id is not None and type(self.event_id) is not int:
            raise TypeError("event_id must be int or None")

        for field_name in (
            "action",
            "target_name",
            "source_name",
            "session_id",
            "node_id",
            "link_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{field_name} must be str or None")

        if not isinstance(self.meta, tuple):
            raise TypeError("meta must be tuple")
        for item in self.meta:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("meta entries must be (key, value) pairs")
            if not isinstance(item[0], str):
                raise TypeError("meta keys must be strings")
