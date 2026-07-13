"""Output targets for structured tracing records.

Sinks receive :class:`simyuj.tracing.records.SimulationLogRecord` instances
from :class:`simyuj.tracing.logger.SimulationLogger`. They are observational
outputs only; emitting to a sink must not affect event ordering or timeline
execution semantics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Protocol, TextIO

from .records import SimulationLogRecord


class LogSink(Protocol):
    """
    Protocol implemented by tracing output targets.

    Custom sinks can implement this protocol to write records to streams,
    files, databases, test buffers, or other observability systems.
    """

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Emit a log record to an output target.

        Parameters
        ----------
        record : SimulationLogRecord
            Structured record to consume.
        """


class NullSink:
    """
    Sink that discards every record.

    ``NullSink`` is used by default when a logger is constructed without
    explicit sinks.
    """

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Discard ``record``.

        Parameters
        ----------
        record : SimulationLogRecord
            Record supplied by the logger.
        """
        del record


class MemorySink:
    """
    Sink that stores records in a list.

    The sink keeps the original record objects in insertion order. It is useful
    for tests and for small interactive runs where callers want to inspect the
    trace after execution.

    Attributes
    ----------
    records : list[SimulationLogRecord]
        Records emitted so far.
    """

    def __init__(self) -> None:
        self.records: list[SimulationLogRecord] = []

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Append ``record`` to :attr:`records`.

        Parameters
        ----------
        record : SimulationLogRecord
            Record to retain.
        """
        self.records.append(record)

    def clear(self) -> None:
        """
        Remove all stored records.
        """
        self.records.clear()


def _format_ps_time(tick: int) -> str:
    """
    Format integer picosecond ticks for text logs.

    The caller keeps the exact tick value in the rendered line, so this helper
    can choose a compact display unit without losing precision.
    """
    units = (
        ("s", 1_000_000_000_000),
        ("ms", 1_000_000_000),
        ("us", 1_000_000),
        ("ns", 1_000),
    )
    abs_tick = abs(tick)
    for unit, scale in units:
        if abs_tick >= scale:
            return f"{tick / scale:.3f} {unit}"
    return f"{tick} ps"


class TextSink:
    """
    Human-readable sink that writes one formatted line per record.

    Parameters
    ----------
    stream : TextIO, optional
        Output stream. Defaults to ``sys.stdout``.
    include_meta : bool, default=True
        Include metadata at the end of the line when present.
    auto_flush : bool, default=False
        Flush the stream after each record.

    Notes
    -----
    ``sim_time`` is formatted as picosecond ticks with a scaled display unit
    and the raw integer tick preserved beside it.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        include_meta: bool = True,
        auto_flush: bool = False,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._include_meta = include_meta
        self._auto_flush = auto_flush

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Write ``record`` as a single text line.

        Parameters
        ----------
        record : SimulationLogRecord
            Record to format and write.
        """
        line = self._format_record(record)
        self._stream.write(line)
        self._stream.write("\n")
        if self._auto_flush:
            self._stream.flush()

    def _format_record(self, record: SimulationLogRecord) -> str:
        """Return the human-readable single-line representation."""
        parts = []
        if record.sim_time is not None:
            parts.append(
                f"[t={_format_ps_time(record.sim_time)} | tick={record.sim_time}]"
            )
        parts.extend(
            [
                f"[{record.sequence:06d}]",
                f"[{record.level.name}]",
                f"[{record.category}]",
            ]
        )

        if record.event_id is not None:
            parts.append(f"event_id={record.event_id}")
        if record.action is not None:
            parts.append(f"action={record.action}")
        if record.target_name is not None:
            parts.append(f"target={record.target_name}")
        if record.source_name is not None:
            parts.append(f"source={record.source_name}")
        if record.session_id is not None:
            parts.append(f"session={record.session_id}")
        if record.node_id is not None:
            parts.append(f"node={record.node_id}")
        if record.link_id is not None:
            parts.append(f"link={record.link_id}")

        parts.append(record.message)

        if self._include_meta and record.meta:
            meta_text = ", ".join(f"{key}={value!r}" for key, value in record.meta)
            parts.append(f"meta={{{meta_text}}}")

        return " ".join(parts)


def _to_jsonable(value: object) -> object:
    """
    Return a JSON-compatible representation of ``value``.

    Basic JSON scalar values pass through unchanged. Tuples and lists are
    converted recursively to lists, dictionaries are converted recursively with
    string keys, and all other values fall back to ``repr(value)``.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return repr(value)


class JsonlSink:
    """
    Structured sink that writes one JSON object per line.

    Parameters
    ----------
    path : str or Path
        Destination JSONL file.
    session_id : str, optional
        Session id to write for every record. When omitted, each record's own
        ``session_id`` is used.
    auto_flush : bool, default=False
        Flush the file after every emitted record.
    create_parents : bool, default=True
        Create missing parent directories before opening the file.
    append : bool, default=False
        Append to an existing file instead of truncating it.

    Raises
    ------
    ValueError
        If ``session_id`` is neither ``None`` nor a non-empty string.

    Notes
    -----
    Records are serialized with sorted JSON keys and ASCII escaping. Metadata is
    written as a list of ``[key, value]`` pairs so duplicate metadata keys and
    input order can be preserved. Values that are not directly JSON-serializable
    are converted by the module's JSON conversion helper.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        session_id: str | None = None,
        auto_flush: bool = False,
        create_parents: bool = True,
        append: bool = False,
    ) -> None:
        resolved_path = Path(path)
        if create_parents:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved_path
        mode = "a" if append else "w"
        self._stream = resolved_path.open(mode, encoding="utf-8")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise ValueError("session_id must be non-empty str or None")
        self._session_id = session_id
        self._auto_flush = auto_flush

    @property
    def path(self) -> Path:
        """
        Destination path opened by the sink.

        Returns
        -------
        Path
            ``Path`` object used by the constructor.
        """
        return self._path

    def emit(self, record: SimulationLogRecord) -> None:
        """
        Serialize ``record`` as one JSON line.

        Parameters
        ----------
        record : SimulationLogRecord
            Record to serialize.
        """
        resolved_session_id = (
            self._session_id if self._session_id is not None else record.session_id
        )
        payload = {
            "action": record.action,
            "category": record.category,
            "event_id": record.event_id,
            "level": record.level.name,
            "link_id": record.link_id,
            "message": record.message,
            "meta": [[key, _to_jsonable(value)] for key, value in record.meta],
            "node_id": record.node_id,
            "sequence": record.sequence,
            "session_id": resolved_session_id,
            "sim_time": record.sim_time,
            "source_name": record.source_name,
            "target_name": record.target_name,
        }
        self._stream.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        )
        self._stream.write("\n")
        if self._auto_flush:
            self._stream.flush()

    def flush(self) -> None:
        """
        Flush pending bytes to the underlying file object.
        """
        self._stream.flush()

    def close(self) -> None:
        """
        Close the underlying file object.
        """
        self._stream.close()

    def __enter__(self) -> "JsonlSink":
        """
        Return this sink for use as a context manager.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """
        Close the sink when leaving a context manager block.
        """
        del exc_type, exc, tb
        self.close()
