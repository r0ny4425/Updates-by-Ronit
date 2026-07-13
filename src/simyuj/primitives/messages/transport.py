"""
Transport payload records for classical and quantum planes.

The records in this module are frozen, slot-backed dataclasses intended to
cross event and component boundaries without mutation. They validate public
constructor inputs but do not schedule events, copy signal state, or perform
channel behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simyuj.signal import Signal

from ..meta import Meta
from ..meta import validate_meta as _validate_meta
from ..validation import validate_bool as _validate_bool
from ..validation import validate_non_empty_str as _validate_non_empty_str
from ..validation import validate_non_negative_int as _validate_non_negative_int
from ..validation import (
    validate_optional_non_empty_str as _validate_optional_non_empty_str,
)
from ..validation import validate_optional_probability as _validate_optional_probability


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassicalMessage:
    """Immutable classical-plane transport message.

    Parameters
    ----------
    sender_id : str
        Non-empty identifier for the sender.
    receiver_id : str
        Non-empty identifier for the intended receiver.
    body : str or bytes
        Message payload carried by the classical plane.
    sent_time : int, default=0
        Non-negative simulation tick at which the message was produced.
    session_id : str or None, default=None
        Optional non-empty session identifier.
    message_id : str, int, or None, default=None
        Optional message identifier.
    message_type : str, default="generic"
        Non-empty message type label.
    correlation_id : str, int, or None, default=None
        Optional identifier used to correlate related messages.
    round_id : str, int, or None, default=None
        Optional protocol or session round identifier.
    meta : Meta, optional
        Hashable metadata tuple attached to the message.

    Raises
    ------
    TypeError
        If IDs, body, optional identifiers, time, or metadata have unsupported
        types.
    ValueError
        If required string fields are empty or `sent_time` is negative.

    Notes
    -----
    The record is immutable and hashable when all field values are hashable.
    It does not serialize or route itself; components and control-plane code
    interpret the body and schedule delivery events.

    Examples
    --------
    >>> message = ClassicalMessage(
    ...     sender_id="alice",
    ...     receiver_id="bob",
    ...     body="basis=Z",
    ...     sent_time=10,
    ...     message_type="basis.announce",
    ... )
    >>> message.message_type
    'basis.announce'
    """

    sender_id: str
    receiver_id: str
    body: str | bytes
    sent_time: int = 0
    session_id: str | None = None
    message_id: str | int | None = None
    message_type: str = "generic"
    correlation_id: str | int | None = None
    round_id: str | int | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.sender_id, field_name="sender_id")
        _validate_non_empty_str(self.receiver_id, field_name="receiver_id")
        _validate_non_negative_int(self.sent_time, field_name="sent_time")
        _validate_optional_non_empty_str(self.session_id, field_name="session_id")
        _validate_non_empty_str(self.message_type, field_name="message_type")
        if self.message_id is not None and not isinstance(self.message_id, (str, int)):
            raise TypeError("message_id must be str, int, or None")
        if self.correlation_id is not None and not isinstance(
            self.correlation_id, (str, int)
        ):
            raise TypeError("correlation_id must be str, int, or None")
        if self.round_id is not None and not isinstance(self.round_id, (str, int)):
            raise TypeError("round_id must be str, int, or None")

        if not isinstance(self.body, (str, bytes)):
            raise TypeError("body must be str or bytes")

        _validate_meta(self.meta)


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantumTransitPayload:
    """Immutable quantum-plane payload for channel forwarding.

    Parameters
    ----------
    sender_id : str
        Non-empty identifier for the sending component.
    receiver_id : str
        Non-empty identifier for the receiving component.
    signal : Signal
        Quantum signal object being forwarded. The payload keeps the original
        object reference.
    launched_time : int
        Non-negative simulation tick when the signal entered transport.
    session_id : str or None, default=None
        Optional non-empty session identifier.
    channel_id : str or None, default=None
        Optional non-empty channel identifier.
    meta : Meta, optional
        Hashable metadata tuple describing transport context.
    timing_meta : Meta, optional
        Hashable metadata tuple for per-hop timing values.

    Raises
    ------
    TypeError
        If IDs, `signal`, timing fields, or metadata have unsupported types.
    ValueError
        If required string fields are empty or `launched_time` is negative.

    Notes
    -----
    ``launched_time`` records when the payload entered transport. Per-hop
    receiver timing belongs in ``timing_meta``. The class does not copy or
    mutate the signal; qstate ownership is handled outside this record.
    """

    sender_id: str
    receiver_id: str
    signal: Signal
    launched_time: int
    session_id: str | None = None
    channel_id: str | None = None
    meta: Meta = field(default_factory=tuple)
    timing_meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.sender_id, field_name="sender_id")
        _validate_non_empty_str(self.receiver_id, field_name="receiver_id")
        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be Signal")
        _validate_non_negative_int(self.launched_time, field_name="launched_time")
        _validate_optional_non_empty_str(self.session_id, field_name="session_id")
        _validate_optional_non_empty_str(self.channel_id, field_name="channel_id")
        _validate_meta(self.meta)
        _validate_meta(self.timing_meta)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeliveryReport:
    """Immutable report for transport delivery or loss.

    Parameters
    ----------
    channel_id : str
        Non-empty identifier for the reporting channel or sink.
    report_time : int
        Non-negative simulation tick when the report was emitted.
    delivered : bool
        Whether the payload reached its intended receiver.
    session_id : str or None, default=None
        Optional non-empty session identifier.
    payload_id : str, int, or None, default=None
        Optional identifier for the payload being reported.
    loss_reason : str or None, default=None
        Optional non-empty loss reason when delivery failed.
    loss_fraction : float or None, default=None
        Optional loss fraction in ``[0.0, 1.0]``.
    meta : Meta, optional
        Hashable metadata tuple attached to the report.

    Raises
    ------
    TypeError
        If IDs, `delivered`, optional payload fields, loss fraction, time, or
        metadata have unsupported types.
    ValueError
        If required string fields are empty, time is negative,
        or `loss_fraction` is outside ``[0.0, 1.0]``.

    Notes
    -----
    A successful delivery may still carry metadata. A failed delivery may use
    ``loss_reason`` and ``loss_fraction`` to describe why or how much payload
    was lost, but the class does not enforce a relationship among those fields.
    """

    channel_id: str
    report_time: int
    delivered: bool
    session_id: str | None = None
    payload_id: str | int | None = None
    loss_reason: str | None = None
    loss_fraction: float | None = None
    meta: Meta = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.channel_id, field_name="channel_id")
        _validate_non_negative_int(self.report_time, field_name="report_time")
        _validate_bool(self.delivered, field_name="delivered")
        _validate_optional_non_empty_str(self.session_id, field_name="session_id")
        if self.payload_id is not None and not isinstance(self.payload_id, (str, int)):
            raise TypeError("payload_id must be str, int, or None")
        _validate_optional_non_empty_str(self.loss_reason, field_name="loss_reason")
        _validate_optional_probability(self.loss_fraction, field_name="loss_fraction")
        _validate_meta(self.meta)


__all__ = [
    "ClassicalMessage",
    "DeliveryReport",
    "QuantumTransitPayload",
]
