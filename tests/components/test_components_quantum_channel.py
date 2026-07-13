from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pytest

from simyuj.components import (
    ACTION_TRANSMIT_QUANTUM,
    Port,
    PortDelivery,
    PortDirection,
    PortKind,
    QuantumChannel,
    connect_ports,
)
from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.engine.component import Component
from simyuj.engine.timeline import Timeline
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.primitives.units import ticks_to_seconds
from simyuj.qstate import DensityState, StateNotFoundError, SubsystemId
from simyuj.qstate.noise import dephasing, depolarizing
from simyuj.signal import EncodingScheme, Signal, SignalKind
from simyuj.tracing.levels import LogLevel
from simyuj.tracing.logger import SimulationLogger
from simyuj.tracing.sinks import MemorySink
from tests.support.binding import binding_context

ATOL = 1e-12


class _ChannelTrace(NamedTuple):
    delivered_ids: tuple[str, ...]
    delivered_times: tuple[int, ...]
    lost_ids: tuple[str, ...]
    live_target_ids: tuple[str, ...]
    received_count: int
    delivered_count: int
    lost_count: int
    events_scheduled: int
    events_executed: int


@dataclass(slots=True)
class QuantumSender(Component):
    device_id: str = "sender"
    output_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.EGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        raise AssertionError("sender should not receive events")


@dataclass(slots=True)
class QuantumSink(Component):
    device_id: str = "sink"
    received: list[tuple[int, PortDelivery]] = field(default_factory=list)
    event_meta: list[dict[str, Any]] = field(default_factory=list)
    input_port: Port = field(init=False)

    def __post_init__(self) -> None:
        self.input_port = Port(
            name="in",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )

    def handle_event(self, event, timeline) -> None:
        if event.action != "receive_signal":
            raise ValueError(event.action)

        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("payload must be PortDelivery")

        if event.payload_ref.target_port is not self.input_port:
            raise ValueError("delivery arrived on unknown port")

        if not isinstance(event.payload_ref.payload, Signal):
            raise TypeError("payload must be Signal")

        self.received.append((timeline.current_time, event.payload_ref))
        self.event_meta.append(dict(event.meta))


class CapturingDurationNoise:
    name = "capturing_duration"
    arity: int = 1

    def __init__(self) -> None:
        self.durations: list[float] = []

    def resolve(self, *, duration_s: float):
        self.durations.append(duration_s)
        return depolarizing(0.0)


def _signal(
    timeline: Timeline,
    *,
    state: str = "|0>",
    rep: str = "ket",
    signal_id: str = "s1",
) -> Signal:
    subsystem = SubsystemId(f"{signal_id}:q0")

    state_ref = timeline.qstate.prepare(
        state,
        rep=rep,
        subsystems=(subsystem,),
    )

    return Signal(
        id=signal_id,
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=timeline.current_time,
        origin="alice",
        wavelength_nm=1550.0,
        state_ref=state_ref,
        state_targets=(
            SubsystemHandle(
                label=str(subsystem),
                kind="qubit",
                index=0,
                metadata=(("qstate_subsystem", str(subsystem)),),
            ),
        ),
    )


def _wired_quantum_chain(**channel_kwargs):
    sender = QuantumSender()
    channel = QuantumChannel(channel_id="qch", **channel_kwargs)
    sink = QuantumSink()

    input_connection = connect_ports(
        sender.output_port,
        channel.input_port,
        target_action=ACTION_TRANSMIT_QUANTUM,
        connection_id="sender.out->qch.in",
    )

    output_connection = connect_ports(
        channel.output_port,
        sink.input_port,
        target_action="receive_signal",
        connection_id="qch.out->sink.in",
    )

    return sender, channel, sink, input_connection, output_connection


def _received_signal(sink: QuantumSink) -> Signal:
    assert len(sink.received) == 1
    payload = sink.received[0][1].payload
    assert isinstance(payload, Signal)
    return payload


def _density_payload(timeline: Timeline, subsystem: SubsystemId) -> DensityState:
    state_ref = timeline.qstate.state_of(subsystem)
    payload = timeline.qstate.get(state_ref)
    assert isinstance(payload, DensityState)
    return payload


def _stochastic_channel_trace(seed: int) -> _ChannelTrace:
    timeline = Timeline(master_seed=seed)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=10,
        fixed_insertion_loss_db=3.0,
        timing_jitter_stddev_ticks=2.0,
    )
    signals = tuple(_signal(timeline, signal_id=f"s{index}") for index in range(8))

    assert 0.0 < channel.survival_probability < 1.0

    channel.bind(binding_context(timeline))
    for index, signal in enumerate(signals):
        input_connection.transmit(signal, timeline, time=index, source=sender)

    timeline.run_until(30)

    delivered_ids = tuple(
        str(cast_signal.id)
        for _, delivery in sink.received
        for cast_signal in (delivery.payload,)
        if isinstance(cast_signal, Signal)
    )
    delivered_times = tuple(time for time, _ in sink.received)
    lost_ids = tuple(
        str(signal.id) for signal in signals if str(signal.id) not in set(delivered_ids)
    )
    live_target_ids = []
    for signal in signals:
        target = qstate_targets_from_signal(signal)[0]
        try:
            timeline.qstate.state_of(target)
        except StateNotFoundError:
            continue
        live_target_ids.append(str(target))

    return _ChannelTrace(
        delivered_ids=delivered_ids,
        delivered_times=delivered_times,
        lost_ids=lost_ids,
        live_target_ids=tuple(live_target_ids),
        received_count=channel.received_count,
        delivered_count=channel.delivered_count,
        lost_count=channel.lost_count,
        events_scheduled=timeline.events_scheduled,
        events_executed=timeline.events_executed,
    )


def test_quantum_channel_ports_are_structural_quantum_ports() -> None:
    channel = QuantumChannel(channel_id="qch")

    assert channel.input_port.owner is channel
    assert channel.input_port.port_kind is PortKind.QUANTUM
    assert channel.input_port.direction is PortDirection.INGRESS
    assert channel.output_port.owner is channel
    assert channel.output_port.port_kind is PortKind.QUANTUM
    assert channel.output_port.direction is PortDirection.EGRESS


def test_quantum_channel_bind_lifecycle() -> None:
    timeline = Timeline(master_seed=1)
    other_timeline = Timeline(master_seed=2)
    channel = QuantumChannel(channel_id="qch")

    channel.bind(binding_context(timeline))
    channel.bind(binding_context(timeline))

    with pytest.raises(RuntimeError, match="already bound"):
        channel.bind(binding_context(other_timeline))


def test_fixed_delay_delivers_downstream() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(delay_ticks=10)
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)

    timeline.run_until(9)
    assert sink.received == []
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.lost_count == 0

    timeline.run_until(10)

    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    assert time == 10
    assert delivery.source_port is channel.output_port
    assert delivery.target_port is sink.input_port


def test_surviving_photon_creates_only_downstream_delivery_event() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(delay_ticks=5)
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    assert timeline.events_scheduled == 1

    timeline.run_until(0)

    assert sink.received == []
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert timeline.events_scheduled == 2
    assert timeline.events_executed == 1

    timeline.run_until(5)

    assert len(sink.received) == 1
    assert sink.received[0][0] == 5
    assert timeline.events_executed == 2


def test_length_based_delay_uses_propagation_speed() -> None:
    _, channel, _, _, _ = _wired_quantum_chain(
        length_m=200.0,
        propagation_speed_m_per_s=2.0e8,
        delay_ticks=None,
    )

    assert channel.resolved_delay_ticks == 1_000_000


def test_delay_ticks_overrides_length_based_delay() -> None:
    _, channel, _, _, _ = _wired_quantum_chain(
        length_m=200.0,
        delay_ticks=7,
    )

    assert channel.resolved_delay_ticks == 7


def test_zero_loss_delivers_signal() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=0,
        attenuation_db_per_km=0.0,
        fixed_insertion_loss_db=0.0,
    )
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    assert channel.received_count == 1
    assert channel.delivered_count == 1
    assert channel.lost_count == 0


def test_forced_loss_drops_signal_and_discards_qstate() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=0,
        fixed_insertion_loss_db=1e9,
    )
    signal = _signal(timeline)
    subsystem = qstate_targets_from_signal(signal)[0]

    assert timeline.qstate.state_of(subsystem) == signal.state_ref

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert sink.received == []
    assert channel.lost_count == 1
    assert channel.delivered_count == 0
    with pytest.raises(StateNotFoundError):
        timeline.qstate.state_of(subsystem)


def test_intermediate_quantum_channel_loss_and_jitter_replay_from_seed() -> None:
    first = _stochastic_channel_trace(seed=2718)
    second = _stochastic_channel_trace(seed=2718)
    other_seed = _stochastic_channel_trace(seed=3141)

    assert first == second
    assert 0 < first.delivered_count < first.received_count
    assert first.lost_count == first.received_count - first.delivered_count
    assert len(first.delivered_times) == first.delivered_count
    assert other_seed.received_count == first.received_count


def test_quantum_channel_logs_ready_and_forwarded_signal() -> None:
    log_sink = MemorySink()
    timeline = Timeline(
        master_seed=1,
        logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]),
    )
    sender, channel, _, input_connection, output_connection = _wired_quantum_chain(
        delay_ticks=5,
        timing_jitter_stddev_ticks=0.0,
    )
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(5)

    ready = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.quantum.ready"
    )
    forwarded = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.quantum.signal_forwarded"
    )

    assert ready.level is LogLevel.INFO
    assert dict(ready.meta) == {
        "channel_id": channel.channel_id,
        "delay_ticks": 5,
        "survival_probability": 1.0,
        "timing_jitter_stddev_ticks": 0.0,
        "delivery_priority": 0,
    }

    assert forwarded.level is LogLevel.DEBUG
    assert forwarded.action == ACTION_TRANSMIT_QUANTUM
    assert dict(forwarded.meta) == {
        "channel_id": channel.channel_id,
        "signal_id": signal.id,
        "signal_kind": signal.signal_kind.value,
        "received_index": 1,
        "connection_id": output_connection.connection_id,
        "delay_ticks": 5,
        "jitter_ticks": 0,
        "duration_ticks": 5,
        "arrival_time": 5,
    }


def test_quantum_channel_logs_lost_signal_at_debug() -> None:
    log_sink = MemorySink()
    timeline = Timeline(
        master_seed=1,
        logger=SimulationLogger(level=LogLevel.DEBUG, sinks=[log_sink]),
    )
    sender, channel, _, input_connection, _ = _wired_quantum_chain(
        delay_ticks=5,
        fixed_insertion_loss_db=1e9,
    )
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(5)

    lost = next(
        record
        for record in log_sink.records
        if record.category == "components.channels.quantum.signal_lost"
    )

    assert lost.level is LogLevel.DEBUG
    assert lost.action == ACTION_TRANSMIT_QUANTUM
    assert dict(lost.meta) == {
        "channel_id": channel.channel_id,
        "signal_id": signal.id,
        "signal_kind": signal.signal_kind.value,
        "received_index": 1,
        "survival_probability": channel.survival_probability,
    }


def test_quantum_channel_concrete_dephasing_converts_to_density() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=0,
        noise_models=[dephasing(1.0)],
    )
    signal = _signal(timeline, state="|+>")
    target = qstate_targets_from_signal(signal)[0]

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    record = timeline.qstate.record(timeline.qstate.state_of(target))
    assert record.rep == "density"


def test_quantum_channel_concrete_depolarizing_converts_to_density() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=0,
        noise_models=[depolarizing(1.0)],
    )
    signal = _signal(timeline, state="|0>")
    target = qstate_targets_from_signal(signal)[0]

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    record = timeline.qstate.record(timeline.qstate.state_of(target))
    assert record.rep == "density"
    assert _density_payload(timeline, target).rho == pytest.approx(
        np.eye(2, dtype=np.complex128) / 2.0,
        abs=ATOL,
    )


def test_no_state_noise_leaves_ket_as_ket() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(delay_ticks=0)
    signal = _signal(timeline, state="|0>")
    target = qstate_targets_from_signal(signal)[0]

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    assert len(sink.received) == 1
    record = timeline.qstate.record(timeline.qstate.state_of(target))
    assert record.rep == "ket"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_m": -1.0},
        {"propagation_speed_m_per_s": 0.0},
        {"propagation_speed_m_per_s": -1.0},
        {"attenuation_db_per_km": -0.1},
        {"fixed_insertion_loss_db": -0.1},
        {"timing_jitter_stddev_ticks": -0.1},
        {"delay_ticks": -1},
    ],
)
def test_invalid_quantum_channel_values_rejected(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {"channel_id": "qch"}
    base.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        QuantumChannel(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length_m": True},
        {"propagation_speed_m_per_s": True},
        {"attenuation_db_per_km": True},
        {"fixed_insertion_loss_db": True},
        {"timing_jitter_stddev_ticks": True},
        {"delay_ticks": True},
    ],
)
def test_boolean_quantum_channel_numeric_fields_rejected(
    kwargs: dict[str, Any],
) -> None:
    base: dict[str, Any] = {"channel_id": "qch"}
    base.update(kwargs)

    with pytest.raises(TypeError):
        QuantumChannel(**base)


def test_quantum_channel_rejects_non_sequence_noise_models() -> None:
    with pytest.raises(TypeError, match="noise_models must be a sequence"):
        QuantumChannel(
            channel_id="qch",
            noise_models=depolarizing(0.01),
        )


def test_channel_metadata_is_appended_to_received_signal() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=10,
        length_m=200.0,
        session_id="session-1",
    )
    signal = _signal(timeline)

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(10)

    received = _received_signal(sink)
    meta = dict(received.meta)
    timing = dict(received.timing_meta)
    event_meta = sink.event_meta[0]

    assert meta == {
        "quantum_channel_id": "qch",
        "survival_probability": 1.0,
    }
    assert timing["channel_delay_ticks"] == 10
    assert timing["channel_timing_jitter_ticks"] == 0
    assert timing["channel_duration_ticks"] == 10
    assert timing["channel_duration_s"] == ticks_to_seconds(10)
    assert timing["channel_arrival_time"] == 10

    assert event_meta["channel_id"] == "qch"
    assert event_meta["signal_id"] == signal.id
    assert event_meta["session_id"] == "session-1"
    for key in (
        "input_port",
        "output_port",
        "source_event_id",
        "source_connection_id",
        "delay_ticks",
        "timing_jitter_ticks",
    ):
        assert key not in event_meta


def test_survival_probability_is_reported_in_metadata() -> None:
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=0,
        length_m=1000.0,
        attenuation_db_per_km=3.0,
        fixed_insertion_loss_db=7.0,
    )
    signal = _signal(timeline)
    expected_survival_probability = 10.0 ** (-10.0 / 10.0)

    assert channel.survival_probability == pytest.approx(
        expected_survival_probability,
        abs=ATOL,
    )

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(0)

    received = _received_signal(sink)
    meta = dict(received.meta)
    assert meta["survival_probability"] == pytest.approx(
        channel.survival_probability,
        abs=ATOL,
    )
    for key in (
        "channel_length_m",
        "attenuation_db_per_km",
        "fixed_insertion_loss_db",
        "dephasing_probability",
        "depolarization_probability",
    ):
        assert key not in meta


def test_quantum_channel_time_dependent_noise_uses_delay_plus_jitter(
    monkeypatch,
) -> None:
    model = CapturingDurationNoise()

    monkeypatch.setattr(
        QuantumChannel,
        "_sample_timing_jitter_ticks",
        lambda self: 3,
    )
    timeline = Timeline(master_seed=1)
    sender, channel, sink, input_connection, _ = _wired_quantum_chain(
        delay_ticks=10,
        noise_models=[model],
    )
    signal = _signal(timeline, state="|0>")
    target = qstate_targets_from_signal(signal)[0]

    channel.bind(binding_context(timeline))
    input_connection.transmit(signal, timeline, time=0, source=sender)
    timeline.run_until(13)

    assert len(sink.received) == 1
    record = timeline.qstate.record(timeline.qstate.state_of(target))
    assert record.rep == "density"
    assert model.durations == [ticks_to_seconds(13)]
    timing = dict(_received_signal(sink).timing_meta)
    assert timing["channel_delay_ticks"] == 10
    assert timing["channel_timing_jitter_ticks"] == 3
    assert timing["channel_duration_ticks"] == 13
    assert timing["channel_duration_s"] == ticks_to_seconds(13)
    assert timing["channel_arrival_time"] == 13
