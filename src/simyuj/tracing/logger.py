"""Structured simulation logger for tracing records.

The logger is the coordination point between event-producing code and tracing
sinks. It applies level filtering, assigns sequence numbers, freezes metadata,
and forwards immutable records to the configured sinks.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .levels import LogLevel
from .records import MetaInput, SimulationLogRecord, freeze_meta
from .sinks import LogSink, NullSink


class SimulationLogger:
    """
    Structured logger for simulation tracing.

    Parameters
    ----------
    level : LogLevel, default=LogLevel.INFO
        Maximum verbosity to emit. ``OFF`` disables emission.
    sinks : Iterable[LogSink], optional
        Output sinks. When omitted, a ``NullSink`` is installed.
    session_id : str, optional
        Default session id copied into records whose log call does not supply
        one explicitly.

    Attributes
    ----------
    level : LogLevel
        Current level filter. Callers may update it between log calls.
    session_id : str or None
        Default session context for emitted records.

    Notes
    -----
    Sequence numbers are assigned only to records that pass level filtering.
    Calling :meth:`emit` with a pre-built record does not assign a sequence
    number or apply level filtering.

    Sink exceptions are not swallowed. If a configured sink raises while
    receiving a record, the logging call raises the same exception.

    Examples
    --------
    >>> from simyuj.tracing.levels import LogLevel
    >>> from simyuj.tracing.sinks import MemorySink
    >>> sink = MemorySink()
    >>> logger = SimulationLogger(level=LogLevel.INFO, sinks=[sink])
    >>> record = logger.log(level=LogLevel.INFO, category="demo", message="hello")
    >>> record.sequence
    0
    >>> hidden = logger.log(level=LogLevel.DEBUG, category="demo", message="hidden")
    >>> hidden
    >>> [record.message for record in sink.records]
    ['hello']
    """

    def __init__(
        self,
        *,
        level: LogLevel = LogLevel.INFO,
        sinks: Iterable[LogSink] | None = None,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(level, LogLevel):
            raise TypeError("level must be a LogLevel")

        sink_tuple = tuple(sinks) if sinks is not None else (NullSink(),)
        for sink in sink_tuple:
            if not hasattr(sink, "emit"):
                raise TypeError("all sinks must implement emit(record)")

        self.level = level
        self.session_id = session_id
        self._sequence = 0
        self._sinks = sink_tuple

    @property
    def sequence(self) -> int:
        """
        Next sequence number to be assigned.

        Returns
        -------
        int
            Sequence value that will be used for the next emitted record.
        """
        return self._sequence

    def is_enabled(self, level: LogLevel) -> bool:
        """
        Return True if logs at `level` should be emitted.

        Parameters
        ----------
        level : LogLevel
            Candidate level for a record.

        Returns
        -------
        bool
            ``True`` when the logger is not ``OFF`` and ``level`` is no more
            verbose than the configured logger level.

        Raises
        ------
        TypeError
            If ``level`` is not a ``LogLevel``.
        """
        if not isinstance(level, LogLevel):
            raise TypeError("level must be a LogLevel")
        if self.level == LogLevel.OFF:
            return False
        return level.value <= self.level.value

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Emit a pre-built record to all sinks.

        Parameters
        ----------
        record : SimulationLogRecord
            Record to forward.

        Notes
        -----
        This method does not apply level filtering and does not update
        :attr:`sequence`; it is for callers that already own the record.
        """
        for sink in self._sinks:
            sink.emit(record)

    def log(
        self,
        *,
        level: LogLevel,
        category: str,
        message: str,
        sim_time: int | None = None,
        event_id: int | None = None,
        action: str | None = None,
        target_name: str | None = None,
        source_name: str | None = None,
        session_id: str | None = None,
        node_id: str | None = None,
        link_id: str | None = None,
        meta: MetaInput = None,
    ) -> SimulationLogRecord | None:
        """
        Create and emit a structured record if level-filtered in.

        Parameters
        ----------
        level : LogLevel
            Level of this record.
        category : str
            Structured category label.
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
            Per-record session id. Defaults to the logger's ``session_id``.
        node_id, link_id : str, optional
            Optional network topology context.
        meta : MetaInput, optional
            Structured metadata accepted by
            :func:`simyuj.tracing.records.freeze_meta`.

        Returns
        -------
        SimulationLogRecord or None
            The emitted record, or ``None`` when ``level`` is filtered out.

        Raises
        ------
        TypeError
            If ``level`` is not a ``LogLevel`` or record field validation fails.
        ValueError
            If record field validation rejects a value.

        Notes
        -----
        Disabled records are filtered before metadata is frozen or a
        ``SimulationLogRecord`` is constructed.
        """
        if not self.is_enabled(level):
            return None

        record = SimulationLogRecord(
            sequence=self._sequence,
            level=level,
            category=category,
            message=message,
            sim_time=sim_time,
            event_id=event_id,
            action=action,
            target_name=target_name,
            source_name=source_name,
            session_id=session_id if session_id is not None else self.session_id,
            node_id=node_id,
            link_id=link_id,
            meta=freeze_meta(meta),
        )
        self._sequence += 1
        self.emit(record)
        return record

    def error(self, category: str, message: str, **kwargs: Any) -> None:
        """
        Log an ``ERROR`` record using the context fields accepted by :meth:`log`.
        """
        self.log(level=LogLevel.ERROR, category=category, message=message, **kwargs)

    def warning(self, category: str, message: str, **kwargs: Any) -> None:
        """
        Log a ``WARNING`` record using the context fields accepted by :meth:`log`.
        """
        self.log(level=LogLevel.WARNING, category=category, message=message, **kwargs)

    def info(self, category: str, message: str, **kwargs: Any) -> None:
        """
        Log an ``INFO`` record using the context fields accepted by :meth:`log`.
        """
        self.log(level=LogLevel.INFO, category=category, message=message, **kwargs)

    def debug(self, category: str, message: str, **kwargs: Any) -> None:
        """
        Log a ``DEBUG`` record using the context fields accepted by :meth:`log`.
        """
        self.log(level=LogLevel.DEBUG, category=category, message=message, **kwargs)

    def trace(self, category: str, message: str, **kwargs: Any) -> None:
        """
        Log a ``TRACE`` record using the context fields accepted by :meth:`log`.
        """
        self.log(level=LogLevel.TRACE, category=category, message=message, **kwargs)


class NullLogger(SimulationLogger):
    """
    Explicit no-op logger.

    ``NullLogger`` configures the base logger at ``LogLevel.OFF`` with no sinks.
    It is used when a timeline or caller wants the logging surface available
    without producing records.
    """

    def __init__(self) -> None:
        super().__init__(level=LogLevel.OFF, sinks=())
