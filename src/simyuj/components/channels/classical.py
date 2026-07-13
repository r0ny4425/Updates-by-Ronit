"""Classical-message transport component.

``ClassicalChannel`` receives port-delivered ``ClassicalMessage`` payloads,
applies deterministic delay and optional Bernoulli loss, then schedules the
surviving message through its classical output port. It does not inspect or
mutate message contents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.messages.transport import ClassicalMessage
from simyuj.primitives.validation import (
    require_non_negative_real,
    require_positive_real,
)
from simyuj.runtime.binding import BindingContext
from simyuj.tracing.levels import LogLevel

from ..connections import PortDelivery, require_connection
from ..ports import Port, PortKind
from ._common import (
    _create_channel_ports,
    _non_negative_int,
    _probability,
    _resolve_delay_ticks,
)

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


ACTION_TRANSMIT_CLASSICAL = "transmit_classical"
ACTION_RECEIVE_CLASSICAL = "receive_classical"
DEFAULT_FIBER_LIGHT_SPEED_M_PER_S = 2.0e8


@dataclass(slots=True)
class ClassicalChannel(Component):
    """Forward classical-plane messages through the component port graph.

    ``ClassicalChannel`` accepts port-routed ``ClassicalMessage`` deliveries on
    its classical input port and schedules a delivery event from its output
    port after the configured propagation delay. The channel is the event
    target; ports are structural endpoints and do not handle events.

    Parameters
    ----------
    channel_id : str
        Non-empty channel identifier used for port ownership, logs, and RNG
        stream names.
    length_m : float, default=0.0
        Fiber length in meters used to derive delay when ``delay_ticks`` is
        ``None``.
    delay_ticks : int, optional
        Explicit non-negative propagation delay in simulation ticks. When set,
        it overrides the length-derived delay.
    fiber_speed_m_per_s : float, default=DEFAULT_FIBER_LIGHT_SPEED_M_PER_S
        Positive propagation speed used for length-derived delay.
    loss_probability : float, default=0.0
        Bernoulli drop probability for each message.
    session_id : str, optional
        Optional session identifier copied into downstream event metadata.
    delivery_priority : int, default=0
        Priority used for the scheduled downstream event.

    Attributes
    ----------
    input_port : Port
        Classical input port named ``"in"``.
    output_port : Port
        Classical output port named ``"out"``.

    Notes
    -----
    ``ACTION_TRANSMIT_CLASSICAL`` is the only action handled by this component.
    Its ``payload_ref`` must be a ``PortDelivery`` whose payload is a
    ``ClassicalMessage`` and whose target port is this channel's input port.

    Surviving messages are forwarded through the output port connection using
    the downstream connection's target action, commonly
    ``ACTION_RECEIVE_CLASSICAL``. The original message object is passed through;
    the channel does not copy or mutate the message body or metadata.

    Dropped messages return before the output connection is checked. A
    disconnected output port therefore fails only when a message survives loss
    and needs downstream delivery.

    Propagation is represented by the scheduled time of the downstream event.
    The channel does not schedule an internal propagation event.

    ``delivered_count`` means "scheduled for downstream delivery". It does not
    mean the downstream component has already handled the message.

    ``bind()`` declares the timeline-owned RNG stream
    ``(channel_id, "classical_channel", "loss")`` before execution. A zero
    loss probability avoids consuming that stream during transmission.

    This is a compact transport model. It represents delay and independent
    message loss only; it does not model bandwidth, congestion, framing,
    serialization, or classical error correction.
    """

    channel_id: str

    length_m: float = 0.0
    delay_ticks: int | None = None
    fiber_speed_m_per_s: float = DEFAULT_FIBER_LIGHT_SPEED_M_PER_S
    loss_probability: float = 0.0
    session_id: str | None = None
    delivery_priority: int = 0

    input_port: Port = field(init=False)
    output_port: Port = field(init=False)

    _received_count: int = field(init=False, default=0)
    _delivered_count: int = field(init=False, default=0)
    _dropped_count: int = field(init=False, default=0)
    _resolved_delay_ticks: int = field(init=False)

    _bound_timeline_id: int | None = field(init=False, default=None)
    _loss_rng: DeterministicRNG | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.channel_id, field_name="channel_id")

        length_m = require_non_negative_real(
            self.length_m,
            field_name="length_m",
            type_name="numeric",
        )
        fiber_speed_m_per_s = require_positive_real(
            self.fiber_speed_m_per_s,
            field_name="fiber_speed_m_per_s",
            type_name="numeric",
        )

        if self.delay_ticks is not None:
            _non_negative_int(
                "delay_ticks",
                self.delay_ticks,
                type_label="int or None",
            )

        self._resolved_delay_ticks = _resolve_delay_ticks(
            self.delay_ticks,
            length_m=length_m,
            speed_m_per_s=fiber_speed_m_per_s,
        )

        _probability("loss_probability", self.loss_probability)

        if self.session_id is not None:
            ensure_nonempty_id(self.session_id, field_name="session_id")

        if type(self.delivery_priority) is not int:
            raise TypeError("delivery_priority must be int")

        self.input_port, self.output_port = _create_channel_ports(
            owner=self,
            owner_id=self.channel_id,
            port_kind=PortKind.CLASSICAL,
        )

    @property
    def received_count(self) -> int:
        """Number of transmission events accepted by the channel."""
        return self._received_count

    @property
    def delivered_count(self) -> int:
        """Number of accepted messages scheduled for downstream delivery."""
        return self._delivered_count

    @property
    def dropped_count(self) -> int:
        """Number of accepted messages dropped by the loss model."""
        return self._dropped_count

    @property
    def resolved_delay_ticks(self) -> int:
        """Resolved non-negative propagation delay in simulation ticks."""
        return self._resolved_delay_ticks

    def bind(self, context: BindingContext) -> None:
        """Bind this channel to a timeline and declare deterministic streams.

        Parameters
        ----------
        context : BindingContext
            Runtime binding context that supplies the timeline.

        Raises
        ------
        TypeError
            If ``context`` is not a ``BindingContext``.
        RuntimeError
            If the channel is already bound to a different timeline.

        Notes
        -----
        Binding is idempotent for the same timeline and must happen before the
        first transmission event executes. The loss RNG stream is declared here
        so fixed seeds and fixed configuration replay the same drop decisions.
        """
        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)
        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "classical channel is already bound to another timeline"
                )
            return

        self._loss_rng = timeline.rng(
            self.channel_id,
            "classical_channel",
            "loss",
        )
        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.channels.classical.ready",
            "classical channel ready",
            meta={
                "channel_id": self.channel_id,
                "delay_ticks": self._resolved_delay_ticks,
                "loss_probability": float(self.loss_probability),
                "delivery_priority": self.delivery_priority,
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """Handle a port-routed classical transmission event.

        Parameters
        ----------
        event : Event
            Timeline event with action ``ACTION_TRANSMIT_CLASSICAL`` and
            ``PortDelivery`` payload.
        timeline : Timeline
            Timeline currently executing the event.

        Raises
        ------
        RuntimeError
            If the channel has not been bound before execution.
        TypeError
            If the payload wrapper or wrapped message has the wrong type.
        ValueError
            If the action is unsupported or the delivery targets another port.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("classical channel must be bound before execution")

        if event.action == ACTION_TRANSMIT_CLASSICAL:
            if not isinstance(event.payload_ref, PortDelivery):
                raise TypeError(
                    "ACTION_TRANSMIT_CLASSICAL payload_ref must be PortDelivery"
                )

            delivery = event.payload_ref

            if delivery.target_port is not self.input_port:
                raise ValueError("classical channel received delivery on unknown port")

            if not isinstance(delivery.payload, ClassicalMessage):
                raise TypeError("delivery payload must be ClassicalMessage")

            self._transmit_now(
                timeline,
                message=delivery.payload,
                event_id=event.event_id,
                action=event.action,
            )
            return

        raise ValueError(
            f"unsupported event action for classical channel: {event.action!r}"
        )

    def _transmit_now(
        self,
        timeline: Timeline,
        *,
        message: ClassicalMessage,
        event_id: int | None,
        action: str,
    ) -> None:
        """Apply loss and schedule downstream delivery for one message."""
        self._received_count += 1

        loss_probability = float(self.loss_probability)
        if loss_probability > 0.0:
            assert self._loss_rng is not None
            if self._loss_rng.random() < loss_probability:
                self._dropped_count += 1
                timeline.log(
                    LogLevel.DEBUG,
                    "components.channels.classical.message_dropped",
                    "classical message dropped",
                    event_id=event_id,
                    action=action,
                    meta={
                        "channel_id": self.channel_id,
                        "message_id": message.message_id,
                        "message_type": message.message_type,
                        "received_index": self._received_count,
                        "loss_probability": loss_probability,
                    },
                )
                return

        self._delivered_count += 1

        arrival_time = timeline.current_time + self._resolved_delay_ticks
        output_connection = require_connection(self.output_port)

        timeline.log(
            LogLevel.DEBUG,
            "components.channels.classical.message_forwarded",
            "classical message forwarded",
            event_id=event_id,
            action=action,
            meta={
                "channel_id": self.channel_id,
                "message_id": message.message_id,
                "message_type": message.message_type,
                "received_index": self._received_count,
                "connection_id": output_connection.connection_id,
                "delay_ticks": self._resolved_delay_ticks,
                "arrival_time": arrival_time,
            },
        )

        event_meta = {
            "channel_id": self.channel_id,
            "message_id": message.message_id,
            "message_type": message.message_type,
        }
        if self.session_id is not None:
            event_meta["session_id"] = self.session_id

        output_connection.transmit(
            message,
            timeline,
            time=arrival_time,
            priority=self.delivery_priority,
            source=self,
            subsystem_id="components",
            meta=event_meta,
        )


__all__ = [
    "ACTION_RECEIVE_CLASSICAL",
    "ACTION_TRANSMIT_CLASSICAL",
    "ClassicalChannel",
    "DEFAULT_FIBER_LIGHT_SPEED_M_PER_S",
]
