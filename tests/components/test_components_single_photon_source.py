from __future__ import annotations

from dataclasses import dataclass, field
from math import expm1
from typing import Any, NamedTuple, cast

import numpy as np
import pytest

from simyuj.components import (
    DeltaTiming,
    ExGaussianTiming,
    GaussianTiming,
    Port,
    PortDelivery,
    PortDirection,
    PortKind,
    SinglePhotonSource,
    SourcePreparationReport,
    connect_ports,
)
from simyuj.components.sources._common import (
    validate_sampled_emission_delay_ticks,
    validate_timing_profile,
    validate_timing_profile_for_chained_scheduler,
)
from simyuj.control import AGENT_REPORT, Agent, AgentContext
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.rng_manager import DeterministicRNG
from simyuj.engine.timeline import Timeline
from simyuj.network import Network
from simyuj.network.routing import RoutePlanner
from simyuj.network.topology import NetworkTopology
from simyuj.primitives.units import ticks_to_seconds
from simyuj.qstate import DensityState, StateSampler, SubsystemId
from simyuj.qstate.noise import (
    DepolarizingNoise,
    NoiseChannel,
    TimeDependentNoiseModel,
    dephasing,
    depolarizing,
)
from simyuj.signal import EncodingScheme, Signal, SignalKind

ATOL = 1e-12
PSD_ATOL = 1e-10


class _EmissionTrace(NamedTuple):
    received_times: tuple[int, ...]
    signal_ids: tuple[str, ...]
    report_ids: tuple[str, ...]
    attempt_indices: tuple[int, ...]
    report_signal_ids: tuple[tuple[str, ...], ...]
    qstate_size: int
    events_scheduled: int
    events_executed: int


@dataclass(slots=True)
class Sink(Component):
    device_id: str = "sink"
    received: list[tuple[int, PortDelivery]] = field(default_factory=list)
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
        self.received.append((timeline.current_time, event.payload_ref))


@dataclass(slots=True)
class ReportAgent(Agent):
    reports_seen: list[object] = field(default_factory=list)

    def on_report(self, report: object, ctx: AgentContext) -> None:
        del ctx
        self.reports_seen.append(report)


def _connect_source_to_sink(source: SinglePhotonSource, sink: Sink):
    return connect_ports(
        source.output_port,
        sink.input_port,
        target_action="receive_signal",
    )


def _attach_context(agent: Agent, timeline: Timeline) -> Network:
    network = Network()
    topology = NetworkTopology(network)
    planner = RoutePlanner(topology)

    def provider(event: Event, current_timeline: Timeline) -> AgentContext:
        return AgentContext(
            agent_id=agent.agent_id,
            node_id=None,
            session_id="session-1",
            timeline=current_timeline,
            event=event,
            network=network,
            topology=topology,
            route_planner=planner,
        )

    agent.attach_context_provider(provider)
    return network


def _emitted_density(timeline: Timeline, signal: Signal) -> np.ndarray:
    assert signal.state_ref is not None
    record = timeline.qstate.record(signal.state_ref)
    assert record.rep == "density"
    assert isinstance(record.payload, DensityState)
    return record.payload.rho


def _assert_valid_density(rho: np.ndarray) -> None:
    assert rho == pytest.approx(rho.conj().T, abs=ATOL)
    assert np.trace(rho) == pytest.approx(1.0, abs=ATOL)
    assert np.min(np.linalg.eigvalsh(rho)) >= -PSD_ATOL


@dataclass(frozen=True, slots=True)
class _FixedNormalRNG:
    value: float

    def normal(self, *, loc: float, scale: float) -> float:
        return self.value


@dataclass(slots=True)
class _SequenceNormalRNG:
    values: tuple[float, ...]
    index: int = 0

    def normal(self, *, loc: float, scale: float) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


@dataclass(frozen=True, slots=True)
class _FixedExGaussianRNG:
    normal_value: float
    exponential_value: float

    def normal(self, *, loc: float, scale: float) -> float:
        return self.normal_value

    def exponential(self, scale: float) -> float:
        return self.exponential_value


@dataclass(slots=True)
class _SequenceExGaussianRNG:
    normal_values: tuple[float, ...]
    exponential_values: tuple[float, ...]
    index: int = 0

    def normal(self, *, loc: float, scale: float) -> float:
        return self.normal_values[self.index]

    def exponential(self, scale: float) -> float:
        value = self.exponential_values[self.index]
        self.index += 1
        return value


@dataclass(slots=True)
class RecordingNoiseModel:
    durations: list[float] = field(default_factory=list)
    name: str = "recording_noise"
    arity: int = 1

    def resolve(self, *, duration_s: float) -> NoiseChannel:
        self.durations.append(duration_s)
        return depolarizing(0.0)


class _NoTimingSampler:
    pass


@dataclass(frozen=True, slots=True)
class _NonCallableTimingSampler:
    sample_emission_delay_ticks: int = 1


def _source_emission_trace(seed: int) -> _EmissionTrace:
    timeline = Timeline(master_seed=seed)
    sink = Sink()
    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=0.45,
        duration_s=12e-12,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(20)

    signal_ids = tuple(
        str(signal.id)
        for _, delivery in sink.received
        for signal in (delivery.payload,)
        if isinstance(signal, Signal)
    )
    return _EmissionTrace(
        received_times=tuple(time for time, _ in sink.received),
        signal_ids=signal_ids,
        report_ids=tuple(report.report_id for report in source.reports),
        attempt_indices=tuple(report.attempt_index for report in source.reports),
        report_signal_ids=tuple(report.signal_ids for report in source.reports),
        qstate_size=timeline.qstate.size(),
        events_scheduled=timeline.events_scheduled,
        events_executed=timeline.events_executed,
    )


def test_source_emits_photon_signal_with_default_sampler() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,  # 1 ps period
        emission_probability=1.0,
        duration_s=1e-12,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert len(sink.received) == 1
    time, delivery = sink.received[0]
    signal = delivery.payload
    assert time == 0
    assert delivery.source_port is source.output_port
    assert delivery.target_port is sink.input_port
    assert isinstance(signal, Signal)
    assert signal.signal_kind is SignalKind.PHOTON
    assert signal.encoding_scheme is EncodingScheme.POLARIZATION
    assert signal.state_ref is not None


def test_source_requires_connected_output_port_when_it_emits() -> None:
    timeline = Timeline(master_seed=1)

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
    )

    source.schedule_start(timeline)

    with pytest.raises(RuntimeError, match="not connected"):
        timeline.run_until(0)

    assert source._attempt_count == 1
    assert source._emitted_count == 0


def test_emission_probability_zero_creates_no_signal() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=0.0,
        duration_s=3e-12,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(3)

    assert sink.received == []


def test_intermediate_source_emission_probability_replays_from_seed() -> None:
    first = _source_emission_trace(seed=123)
    second = _source_emission_trace(seed=123)
    other_seed = _source_emission_trace(seed=456)

    assert first == second
    assert 0 < len(first.received_times) < 12
    assert first.signal_ids == tuple(ids[0] for ids in first.report_signal_ids)
    assert len(first.report_ids) == len(first.received_times)
    assert 0 <= len(other_seed.received_times) <= 12


def test_duration_stops_future_emissions() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=3e-12,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(10)

    assert [time for time, _ in sink.received] == [0, 1, 2]
    assert [
        cast(Signal, delivery.payload).state_targets[0].index
        for _, delivery in sink.received
    ] == [0, 0, 0]


def test_frequency_hz_converts_to_picosecond_period() -> None:
    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e9,  # 1 ns = 1000 ps
    )

    assert source.emission_period_ticks == 1000


def test_sampler_metadata_is_attached_to_signal() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    sampler = StateSampler(
        states=("|0>",),
        probabilities=(1.0,),
        rep="ket",
        labels=("H",),
    )

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
        sampler=sampler,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(0)

    signal = sink.received[0][1].payload
    assert isinstance(signal, Signal)
    assert ("sampler_label", "H") in signal.meta
    assert ("sampler_index", 0) in signal.meta


def test_source_stores_preparation_report_with_sampler_choice() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    sampler = StateSampler(
        states=("|0>",),
        probabilities=(1.0,),
        rep="ket",
        labels=("H",),
    )
    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
        sampler=sampler,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert len(source.reports) == 1
    report = source.reports[0]
    assert isinstance(report, SourcePreparationReport)
    assert report.report_id == "alice_source:prep:1"
    assert report.device_id == "alice_source"
    assert report.time == 0
    assert report.attempt_index == 1
    assert report.emission_index == 1
    assert report.signal_ids == ("alice_source:photon:1",)
    assert report.sampler_index == 0
    assert report.sampler_label == "H"
    assert report.state_targets == (SubsystemId("alice_source:photon:1"),)
    assert report.emission_slot_tick == 0
    assert report.emission_delay_ticks == 0


def test_skipped_emission_does_not_create_preparation_report() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=0.0,
        duration_s=1e-12,
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert source.reports == []


def test_source_preparation_report_can_arrive_at_agent_report_port() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()
    agent = ReportAgent(agent_id="alice_agent")
    network = _attach_context(agent, timeline)

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
    )
    _connect_source_to_sink(source, sink)
    network.wire_ports(
        "alice_source_reports",
        source.report_port,
        agent.reports.port("source"),
        target_action=AGENT_REPORT,
    )

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert agent.reports_seen == [source.reports[0]]
    assert isinstance(agent.reports_seen[0], SourcePreparationReport)


def test_noise_models_apply_nonzero_dephasing_to_emitted_state() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()
    sampler = StateSampler(
        states=("|+>",),
        probabilities=(1.0,),
        rep="ket",
        labels=("plus",),
    )

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=1e12,
        emission_probability=1.0,
        duration_s=1e-12,
        sampler=sampler,
        noise_models=(dephasing(1.0),),
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(0)

    assert len(sink.received) == 1
    signal = sink.received[0][1].payload
    assert isinstance(signal, Signal)
    rho = _emitted_density(timeline, signal)
    expected_minus = np.array(
        [[0.5, -0.5], [-0.5, 0.5]],
        dtype=np.complex128,
    )

    assert rho == pytest.approx(expected_minus, abs=ATOL)
    _assert_valid_density(rho)


def test_time_dependent_source_noise_uses_emission_delay_probability() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()
    rate_hz = 1.0e12
    delay_ticks = 2

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=2e11,
        emission_probability=1.0,
        duration_s=6e-12,
        timing_profile=DeltaTiming(emission_delay_ticks=delay_ticks),
        noise_models=(
            cast(TimeDependentNoiseModel, DepolarizingNoise(rate_hz=rate_hz)),
        ),
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(delay_ticks)

    assert len(sink.received) == 1
    signal = sink.received[0][1].payload
    assert isinstance(signal, Signal)
    rho = _emitted_density(timeline, signal)
    p = -expm1(-rate_hz * ticks_to_seconds(delay_ticks))
    expected = np.array(
        [[1.0 - p / 2.0, 0.0], [0.0, p / 2.0]],
        dtype=np.complex128,
    )

    assert rho == pytest.approx(expected, abs=ATOL)
    _assert_valid_density(rho)


def test_source_rejects_non_sequence_noise_models() -> None:
    with pytest.raises(TypeError):
        SinglePhotonSource(
            device_id="alice_source",
            frequency_hz=1e12,
            noise_models=depolarizing(0.01),
        )


def test_noise_models_receive_emission_delay_duration() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()
    noise_model = RecordingNoiseModel()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=2e11,
        emission_probability=1.0,
        duration_s=6e-12,
        timing_profile=DeltaTiming(emission_delay_ticks=2),
        noise_models=(noise_model,),
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(2)

    assert len(sink.received) == 1
    signal = sink.received[0][1].payload
    assert isinstance(signal, Signal)
    assert signal.state_ref is not None
    assert timeline.qstate.record(signal.state_ref).rep == "density"
    assert noise_model.durations == [ticks_to_seconds(2)]


def test_delta_timing_zero_emits_on_periodic_clock() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=5e11,  # 2 ps period
        emission_probability=1.0,
        start_time_s=3e-12,
        duration_s=7e-12,
        timing_profile=DeltaTiming(emission_delay_ticks=0),
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(10)

    assert [time for time, _ in sink.received] == [3, 5, 7, 9]


def test_delta_timing_fixed_offset_preserves_nominal_clock() -> None:
    timeline = Timeline(master_seed=1)
    sink = Sink()

    source = SinglePhotonSource(
        device_id="alice_source",
        frequency_hz=2e11,
        emission_probability=1.0,
        duration_s=13e-12,
        timing_profile=DeltaTiming(emission_delay_ticks=2),
    )
    _connect_source_to_sink(source, sink)

    source.schedule_start(timeline)
    timeline.run_until(12)

    assert [time for time, _ in sink.received] == [2, 7, 12]


@pytest.mark.parametrize(
    ("draw", "expected"),
    [
        (0.49, 0),
        (0.5, 1),
        (2.5, 3),
    ],
)
def test_gaussian_timing_uses_half_up_tick_quantization(
    draw,
    expected,
) -> None:
    timing = GaussianTiming(max_emission_delay_ticks=expected)

    assert (
        timing.sample_emission_delay_ticks(
            cast(DeterministicRNG, _FixedNormalRNG(draw))
        )
        == expected
    )


def test_exgaussian_timing_uses_half_up_tick_quantization() -> None:
    timing = ExGaussianTiming()

    assert (
        timing.sample_emission_delay_ticks(
            cast(
                DeterministicRNG,
                _FixedExGaussianRNG(normal_value=1.0, exponential_value=1.5),
            )
        )
        == 3
    )


def test_validate_timing_profile_requires_callable_sampler() -> None:
    with pytest.raises(TypeError, match="sample_emission_delay_ticks"):
        validate_timing_profile(_NoTimingSampler())

    with pytest.raises(TypeError, match="sample_emission_delay_ticks"):
        validate_timing_profile(_NonCallableTimingSampler())


def test_validate_timing_profile_for_chained_scheduler_rejects_unbounded_delay() -> (
    None
):
    with pytest.raises(ValueError, match="unbounded emission delay"):
        validate_timing_profile_for_chained_scheduler(
            timing_profile=GaussianTiming(emission_delay_stddev_ticks=1.0),
            emission_period_tick_count=10,
        )


def test_validate_sampled_emission_delay_ticks_enforces_period_bound() -> None:
    assert validate_sampled_emission_delay_ticks(3, emission_period_tick_count=10) == 3

    with pytest.raises(ValueError, match="greater than or equal"):
        validate_sampled_emission_delay_ticks(10, emission_period_tick_count=10)


def test_gaussian_timing_resamples_until_it_fits_max_delay() -> None:
    timing = GaussianTiming(
        emission_delay_stddev_ticks=1.0,
        max_emission_delay_ticks=3,
        max_resample_attempts=2,
    )

    assert (
        timing.sample_emission_delay_ticks(
            cast(DeterministicRNG, _SequenceNormalRNG(values=(3.6, 2.6)))
        )
        == 3
    )


def test_exgaussian_timing_resamples_until_it_fits_max_delay() -> None:
    timing = ExGaussianTiming(
        gaussian_emission_delay_stddev_ticks=1.0,
        max_emission_delay_ticks=3,
        max_resample_attempts=2,
    )

    assert (
        timing.sample_emission_delay_ticks(
            cast(
                DeterministicRNG,
                _SequenceExGaussianRNG(
                    normal_values=(2.0, 1.0),
                    exponential_values=(2.0, 1.0),
                ),
            )
        )
        == 2
    )


def test_gaussian_timing_raises_when_resampling_cannot_meet_max_delay() -> None:
    timing = GaussianTiming(
        emission_delay_stddev_ticks=1.0,
        max_emission_delay_ticks=1,
        max_resample_attempts=2,
    )

    with pytest.raises(RuntimeError, match="failed to sample Gaussian emission delay"):
        timing.sample_emission_delay_ticks(
            cast(DeterministicRNG, _SequenceNormalRNG(values=(2.6, 2.6)))
        )


def test_fixed_gaussian_timing_infers_max_delay() -> None:
    timing = GaussianTiming(mean_emission_delay_ticks=2.5)

    assert timing.max_emission_delay_ticks == 3


@pytest.mark.parametrize("frequency_hz", [0.0, -1.0, float("inf")])
def test_invalid_frequency_rejected(frequency_hz) -> None:
    with pytest.raises(ValueError):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=frequency_hz,
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("inf")])
def test_invalid_emission_probability_rejected(probability) -> None:
    with pytest.raises(ValueError):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=1e12,
            emission_probability=probability,
        )


def test_source_rejects_timing_delay_reaching_period() -> None:
    with pytest.raises(ValueError, match="strictly smaller"):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=5e11,
            timing_profile=DeltaTiming(emission_delay_ticks=2),
        )


def test_source_rejects_unbounded_gaussian_timing() -> None:
    with pytest.raises(ValueError, match="finite max_emission_delay_ticks"):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=1e12,
            timing_profile=GaussianTiming(emission_delay_stddev_ticks=1.0),
        )


def test_source_rejects_unbounded_exgaussian_timing() -> None:
    with pytest.raises(ValueError, match="finite max_emission_delay_ticks"):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=1e12,
            timing_profile=ExGaussianTiming(
                gaussian_emission_delay_stddev_ticks=1.0,
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frequency_hz": True},
        {"emission_probability": True},
        {"wavelength_nm": True},
        {"start_time_s": True},
        {"duration_s": True},
    ],
)
def test_boolean_source_numeric_fields_rejected(kwargs) -> None:
    source_kwargs: dict[str, Any] = {
        "device_id": "source",
        "frequency_hz": 1e12,
    }
    source_kwargs.update(kwargs)

    with pytest.raises(TypeError):
        SinglePhotonSource(**source_kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_emission_delay_ticks": True},
        {"emission_delay_stddev_ticks": True},
        {"max_emission_delay_ticks": True},
        {"max_resample_attempts": True},
    ],
)
def test_boolean_gaussian_timing_fields_rejected(kwargs) -> None:
    with pytest.raises(TypeError):
        GaussianTiming(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_mean_emission_delay_ticks": True},
        {"gaussian_emission_delay_stddev_ticks": True},
        {"exponential_emission_delay_scale_ticks": True},
        {"max_emission_delay_ticks": True},
        {"max_resample_attempts": True},
    ],
)
def test_boolean_exgaussian_timing_fields_rejected(kwargs) -> None:
    with pytest.raises(TypeError):
        ExGaussianTiming(**kwargs)


def test_multi_qubit_sampler_rejected() -> None:
    sampler = StateSampler(
        states=([1.0, 0.0, 0.0, 0.0],),
        probabilities=(1.0,),
        rep="ket",
    )

    with pytest.raises(ValueError, match="one-qubit sampler"):
        SinglePhotonSource(
            device_id="source",
            frequency_hz=1e12,
            sampler=sampler,
        )
