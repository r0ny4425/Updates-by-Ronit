"""Port-based terminating sink for component transport tests.

The three older stubs in this package -- ``EmitterStub``, ``ChannelStub``, and
``DetectorStub`` -- predate the port layer. They take a bare
``output_target: Component``, build ``Event(target_ref=...)`` by hand, and read
``event.payload_ref`` as the payload itself. They remain correct for the
``Timeline`` batching and dispatch tests they were written for, and should keep
being used there.

They cannot terminate a ``PortConnection``, for two reasons that are not about
their action strings (``PortConnection.target_action`` is arbitrary):

- they own no ``Port``, and a connection needs a ``target_port`` whose ``owner``
  is the component;
- ``PortConnection.transmit()`` wraps the payload in a ``PortDelivery``, which
  they would forward or inspect as though it were the payload.

``SignalSink`` is the port-based counterpart: one quantum ingress port, no
outputs, and it records what arrived. Use one instance per port -- a component
with two quantum outputs is terminated by two sinks with distinct
``device_id``s, since each owns its own ``"in"`` port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simyuj.components.connections import PortDelivery
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.signal import Signal

ACTION_RECEIVE_SIGNAL = "receive_signal"
"""Default ``target_action`` to wire into a :class:`SignalSink`."""


@dataclass(slots=True)
class SignalSink(Component):
    """Terminating component that records signals delivered to its input port.

    Parameters
    ----------
    device_id : str, default="sink"
        Identifier used for port ownership and in assertion messages.
    action : str, default=ACTION_RECEIVE_SIGNAL
        Event action this sink accepts. Any other action raises.

    Attributes
    ----------
    input_port : Port
        Quantum ingress port named ``"in"``.
    received : list[tuple[int, Signal]]
        ``(arrival_tick, signal)`` for every accepted delivery, in order.
    deliveries : list[PortDelivery]
        The raw delivery wrappers, for tests that need the target port.
    event_meta : list[dict[str, Any]]
        Event metadata for each accepted delivery.

    Notes
    -----
    The sink validates the delivery rather than trusting it: wrong action,
    wrong payload wrapper, wrong port, or a non-``Signal`` payload each raise.
    A silent sink would turn a wiring mistake into an empty ``received`` list,
    which is the failure this class exists to prevent.
    """

    device_id: str = "sink"
    action: str = ACTION_RECEIVE_SIGNAL

    input_port: Port = field(init=False)
    received: list[tuple[int, Signal]] = field(default_factory=list)
    deliveries: list[PortDelivery] = field(default_factory=list)
    event_meta: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    @property
    def signals(self) -> list[Signal]:
        """Just the received signals, without arrival ticks."""
        return [signal for _, signal in self.received]

    def handle_event(self, event, timeline) -> None:
        """Record one port-routed signal delivery."""
        if event.action != self.action:
            raise ValueError(
                f"{self.device_id} received unsupported action: {event.action!r}"
            )

        delivery = event.payload_ref
        if not isinstance(delivery, PortDelivery):
            raise TypeError(f"{self.device_id} payload_ref must be PortDelivery")

        if delivery.target_port is not self.input_port:
            raise ValueError(f"{self.device_id} delivery arrived on unknown port")

        if not isinstance(delivery.payload, Signal):
            raise TypeError(f"{self.device_id} delivery payload must be Signal")

        self.received.append((timeline.current_time, delivery.payload))
        self.deliveries.append(delivery)
        self.event_meta.append(dict(event.meta))


__all__ = ["ACTION_RECEIVE_SIGNAL", "SignalSink"]
