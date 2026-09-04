"""``SignalSink`` terminates a real ``PortConnection``.

The three older stubs in ``tests/support/mock_components/`` cannot: they own no
``Port`` and read ``event.payload_ref`` as the payload rather than unwrapping the
``PortDelivery``. This exercises the sink against an actual connection and
timeline so the scaffolding is known-good before a component depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simyuj.components.connections import connect_ports
from simyuj.components.ports import Port, PortDirection, PortKind
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.signal import EncodingScheme, Signal, SignalKind
from tests.support.mock_components import ACTION_RECEIVE_SIGNAL, SignalSink


@dataclass(slots=True)
class _Emitter(Component):
    """Minimal port-owning sender; transmits on demand."""

    device_id: str = "alice"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:  # pragma: no cover
        raise AssertionError("emitter should not receive events")


def _pulse(index: int) -> Signal:
    return Signal(
        id=f"alice:pulse:{index}",
        signal_kind=SignalKind.PULSE,
        encoding_scheme=EncodingScheme.PHASE,
        emission_time=0,
        origin="alice",
        coherent_state=CoherentState.from_mean_photon_number(0.2),
        temporal_mode_sigma_s=1e-11,
    )


def _wired() -> tuple[_Emitter, SignalSink]:
    emitter = _Emitter()
    sink = SignalSink(device_id="bob")
    connect_ports(
        emitter.output_port,
        sink.input_port,
        target_action=ACTION_RECEIVE_SIGNAL,
        connection_id="alice.out->bob.in",
    )
    return emitter, sink


def test_records_arrival_tick_and_signal() -> None:
    timeline = Timeline(master_seed=1)
    emitter, sink = _wired()
    connection = emitter.output_port.connection
    assert connection is not None

    signal = _pulse(1)
    connection.transmit(
        signal, timeline, time=5, source=emitter, subsystem_id="components"
    )
    timeline.run_until_empty()

    assert sink.received == [(5, signal)]
    assert sink.signals == [signal]


def test_carries_the_optical_fields_through_delivery() -> None:
    timeline = Timeline(master_seed=1)
    emitter, sink = _wired()
    connection = emitter.output_port.connection
    assert connection is not None

    connection.transmit(
        _pulse(1), timeline, time=0, source=emitter, subsystem_id="components"
    )
    timeline.run_until_empty()

    delivered = sink.signals[0]
    assert delivered.coherent_state is not None
    assert delivered.coherent_state.mean_photon_number == pytest.approx(0.2)
    assert delivered.temporal_mode_sigma_s == 1e-11


def test_preserves_delivery_order() -> None:
    timeline = Timeline(master_seed=1)
    emitter, sink = _wired()
    connection = emitter.output_port.connection
    assert connection is not None

    for index, tick in enumerate((3, 1, 2), start=1):
        connection.transmit(
            _pulse(index),
            timeline,
            time=tick,
            source=emitter,
            subsystem_id="components",
        )
    timeline.run_until_empty()

    assert [tick for tick, _ in sink.received] == [1, 2, 3]


def test_rejects_an_unexpected_action() -> None:
    timeline = Timeline(master_seed=1)
    emitter, sink = _wired()
    connection = emitter.output_port.connection
    assert connection is not None

    connection.transmit(
        _pulse(1),
        timeline,
        time=0,
        source=emitter,
        subsystem_id="components",
        action="not_the_wired_action",
    )
    with pytest.raises(ValueError, match="unsupported action"):
        timeline.run_until_empty()


def test_rejects_a_non_signal_payload() -> None:
    timeline = Timeline(master_seed=1)
    emitter, sink = _wired()
    connection = emitter.output_port.connection
    assert connection is not None

    connection.transmit(
        "not a signal", timeline, time=0, source=emitter, subsystem_id="components"
    )
    with pytest.raises(TypeError, match="payload must be Signal"):
        timeline.run_until_empty()
