"""Bell-state analyzer component and BSM support records.

This module receives qstate-backed quantum signals on left/right ingress ports,
buffers unmatched arrivals, performs Bell measurements on matched targets, and
stores or emits ``DetectionReport`` objects. It also contains the optional
linear-optical click-pattern layer used to turn BSM decisions into detector
channel exposures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Protocol, cast

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import validate_meta
from simyuj.primitives.validation import validate_bool
from simyuj.qstate import SubsystemId
from simyuj.runtime.binding import BindingContext
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel

from ..connections import PortConnection, PortDelivery
from ..ports import Port, PortDirection, PortKind
from ..quantum_targets import qstate_targets_from_signal
from .primitives.actions import ACTION_COINCIDENCE_TIMEOUT, ACTION_RUN_BELL_ANALYSIS
from .primitives.click import ClickPattern, resolve_click_pattern
from .primitives.gate import AlwaysOpenGate, GateModel
from .primitives.measurement import (
    Measure,
    MeasurementCall,
    MeasurementContext,
    execute_measurement_call,
)
from .primitives.readout import DetectorExposure
from .primitives.reports import (
    FLAG_DOUBLE_CLICK,
    FLAG_NO_CLICK,
    FLAG_NO_OUTCOME,
    FLAG_TIMEOUT,
    DetectionReport,
    RawClick,
)
from .primitives.result_labels import result_label
from .primitives.rng import DetectorRNGStreams
from .primitives.window import (
    bind_detector_rngs,
    evaluate_detector_windows,
    normalize_detectors,
    validate_gate_model,
)
from .single_photon import SinglePhotonDetector

_LEFT = "left"
_RIGHT = "right"
_VALID_SIDES = frozenset((_LEFT, _RIGHT))

_BELL_LABELS = frozenset(("phi+", "phi-", "psi+", "psi-"))
_LINEAR_OPTICAL_LABELS = frozenset(("psi+", "psi-"))

BSMKind = Literal["ideal", "linear_optical"]


class _RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class BSMModel:
    """
    Classical Bell-state measurement decision model.

    Parameters
    ----------
    kind : {"ideal", "linear_optical"}, default="ideal"
        Bell-state measurement model. ``"ideal"`` can report all Bell labels;
        ``"linear_optical"`` can report only ``"psi+"`` and ``"psi-"``.
    heralding_efficiency : float, default=1.0
        Probability that a detectable Bell label survives the heralding model.

    Notes
    -----
    ``heralding_efficiency`` is conditional on the sampled Bell label being
    detectable by ``kind``. For ``linear_optical`` with uniformly distributed
    Bell labels, the total idealized success probability is therefore
    ``0.5 * heralding_efficiency`` before detector-channel losses and
    click-pattern resolution. In ``linear_optical`` mode, ``"phi+"`` and
    ``"phi-"`` are logical no-outcome labels even though qstate Bell
    measurement itself may have succeeded.
    """

    kind: BSMKind = "ideal"
    heralding_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"ideal", "linear_optical"}:
            raise ValueError("kind must be 'ideal' or 'linear_optical'")

        if type(self.heralding_efficiency) is bool or not isinstance(
            self.heralding_efficiency,
            (int, float),
        ):
            raise TypeError("heralding_efficiency must be numeric")

        efficiency = float(self.heralding_efficiency)

        if not isfinite(efficiency):
            raise ValueError("heralding_efficiency must be finite")
        if efficiency < 0.0 or efficiency > 1.0:
            raise ValueError("heralding_efficiency must be in [0, 1]")

        object.__setattr__(self, "heralding_efficiency", efficiency)

    def is_detectable(self, label: object | None) -> bool:
        if self.kind == "ideal":
            return label in _BELL_LABELS

        if self.kind == "linear_optical":
            return label in _LINEAR_OPTICAL_LABELS

        raise ValueError(f"unsupported BSM model kind: {self.kind!r}")


def bsm_model_from_spec(value: object) -> BSMModel:
    """
    Convert a public Bell-state measurement model spec to ``BSMModel``.

    Parameters
    ----------
    value : object
        Existing ``BSMModel`` or a model-kind string.
    """

    if isinstance(value, BSMModel):
        return value

    if isinstance(value, str):
        return BSMModel(kind=cast(BSMKind, value))

    raise TypeError("bsm_model must be BSMModel or str")


@dataclass(frozen=True, slots=True)
class BSMDecision:
    """
    Immutable classical decision made after qstate Bell measurement.

    Parameters
    ----------
    true_label : object or None
        Bell label returned by the qstate measurement result.
    detectable : bool
        Whether ``BSMModel.kind`` can in principle report ``true_label``.
    survived_efficiency : bool
        Whether the heralding-efficiency Bernoulli decision succeeded.
    success : bool
        Whether the model reports a logical Bell outcome before any physical
        detector click-pattern checks.
    reported_label : object or None
        Label reported by the decision model.
    failure_reason : str or None, default=None
        Compact failure reason for report metadata.
    flags : tuple[str, ...], default=()
        Flags copied to label-only Bell reports.

    Notes
    -----
    This decision is made before the optional physical click-pattern layer. In
    ``linear_optical`` mode, ``success=True`` can still produce a failed final
    report if detector clicks are absent or do not resolve to a configured
    pattern.
    """

    true_label: object | None
    detectable: bool
    survived_efficiency: bool
    success: bool
    reported_label: object | None
    failure_reason: str | None = None
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_bool(self.detectable, field_name="detectable")
        validate_bool(self.survived_efficiency, field_name="survived_efficiency")
        validate_bool(self.success, field_name="success")
        if self.failure_reason is not None and not isinstance(
            self.failure_reason,
            str,
        ):
            raise TypeError("failure_reason must be str or None")
        if not isinstance(self.flags, tuple):
            raise TypeError("flags must be tuple")
        for flag in self.flags:
            if not isinstance(flag, str):
                raise TypeError("flags must contain str")


@dataclass(frozen=True, slots=True)
class BufferedBellInput:
    """
    One Bell-analyzer input stored while waiting for its partner.

    Parameters
    ----------
    buffer_id : str
        Analyzer-local unique buffer identifier.
    side : {"left", "right"}
        Input side that received the signal.
    signal : Signal
        Quantum signal delivered to the analyzer.
    targets : tuple[SubsystemId, ...]
        Exactly one qstate target resolved from the signal.
    arrival_time : int
        Signal arrival tick.
    pair_key : object or None
        Optional metadata value used to pair left/right inputs.
    source_port_name : str
        Upstream egress port name from the ``PortDelivery``.
    connection_id : str
        Runtime connection identifier from the ``PortDelivery``.
    event_id : int or None, default=None
        Timeline event id associated with the arrival event.

    Notes
    -----
    Buffered inputs preserve qstate target ownership until a match is analyzed
    or the coincidence timeout fires. The analyzer optionally discards matched
    or unmatched targets after producing the corresponding report. Exactly one
    qstate target per side is required because this component performs one
    two-target Bell measurement, not multi-mode or multi-photon analysis.
    """

    buffer_id: str
    side: str
    signal: Signal
    targets: tuple[SubsystemId, ...]
    arrival_time: int
    pair_key: object | None
    source_port_name: str
    connection_id: str
    event_id: int | None = None

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.buffer_id, field_name="buffer_id")

        if self.side not in _VALID_SIDES:
            raise ValueError("side must be 'left' or 'right'")

        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be Signal")

        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be tuple")

        if len(self.targets) != 1:
            raise ValueError(
                "BellStateAnalyzer requires exactly one qstate target per input"
            )

        for target in self.targets:
            if not isinstance(target, SubsystemId):
                raise TypeError("targets must contain SubsystemId")

        if type(self.arrival_time) is not int:
            raise TypeError("arrival_time must be int")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")

        ensure_nonempty_id(self.source_port_name, field_name="source_port_name")
        ensure_nonempty_id(self.connection_id, field_name="connection_id")

        if self.event_id is not None and type(self.event_id) is not int:
            raise TypeError("event_id must be int or None")


@dataclass(frozen=True, slots=True)
class CoincidenceTimeout:
    """
    Internal timeout payload for an unmatched Bell input.

    Parameters
    ----------
    buffer_id : str
        Buffered input to expire.
    side : {"left", "right"}
        Buffer side containing the input.
    """

    buffer_id: str
    side: str

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.buffer_id, field_name="buffer_id")
        if self.side not in _VALID_SIDES:
            raise ValueError("side must be 'left' or 'right'")


@dataclass(slots=True)
class BellStateAnalyzer(Component):
    """
    Event-driven Bell-state analyzer with optional physical click model.

    Parameters
    ----------
    device_id : str
        Non-empty component identifier.
    bsm_model : object, optional
        Bell-state decision model spec converted by ``bsm_model_from_spec``.
    detectors : Sequence[SinglePhotonDetector], default=()
        Detector channels used by the ``linear_optical`` physical click model.
    click_patterns : Sequence[ClickPattern], default=()
        Detector-pair patterns mapping physical clicks to ``"psi+"`` or
        ``"psi-"`` outcomes for ``linear_optical`` mode.
    measurement : object, optional
        Bell measurement spec. It must resolve to a collapsing Bell
        ``MeasurementCall``.
    pairing_key : str or None, default="bsa_pair_id"
        Metadata key used to pair left/right signals. ``None`` disables keyed
        pairing.
    allow_fifo_fallback : bool, default=True
        Whether unkeyed inputs can match the oldest opposite-side input inside
        the coincidence window.
    coincidence_window_ticks : int, default=0
        Maximum allowed absolute arrival-time delta for a pair.
    consume_matched_inputs : bool, default=True
        Whether matched qstate targets are discarded after analysis.
    consume_unmatched_inputs : bool, default=True
        Whether timed-out qstate targets are discarded.
    gate_model : GateModel, optional
        Gate schedule used for physical detector windows in linear-optical
        mode.
    detection_window_ticks : int, default=1
        Positive physical detector window length in simulation ticks.
    detector_meta : tuple[tuple[str, object], ...], default=()
        Extra metadata added to physical click records.
    output_latency_ticks : int, default=0
        Additional report-emission delay after analysis, timeout, or latest raw
        click time.
    output_priority : int, default=0
        Priority used for scheduled output-report events.
    timeout_priority : int, default=10000
        Priority used for internally scheduled coincidence-timeout events.

    Attributes
    ----------
    left_input_port, right_input_port : Port
        Quantum ingress ports named ``"left"`` and ``"right"``.
    output_port : Port
        Classical egress port named ``"out"``.
    reports : list[DetectionReport]
        Stored Bell-analysis and timeout reports in handling order.

    Notes
    -----
    The analyzer accepts ``ACTION_RUN_BELL_ANALYSIS`` on either quantum input
    port and ``ACTION_COINCIDENCE_TIMEOUT`` for internally scheduled stale
    buffers. Arrival events must carry ``PortDelivery`` payloads whose signal
    resolves to exactly one qstate target. Matching uses ``pairing_key`` when
    either side has a key; otherwise FIFO fallback may match an opposite-side
    input within ``coincidence_window_ticks``.

    A matched pair is measured through the qstate manager using named
    ``"left"`` and ``"right"`` targets. ``ideal`` mode reports the BSM decision
    directly. ``linear_optical`` mode also maps successful decisions to
    configured detector click patterns and evaluates detector-channel physics
    before resolving the observed outcome.

    If either side has a non-``None`` pair key, only equal non-``None`` keys
    match; an input with a key does not FIFO-match an unkeyed input. If neither
    side has a key, FIFO fallback may match when enabled. Timeouts are scheduled
    at ``arrival_time + coincidence_window_ticks + 1`` so a boundary-time
    partner can still match first. Stale timeout events are ignored when their
    buffer has already matched.

    ``consume_matched_inputs`` discards both matched targets after analysis
    when enabled. ``consume_unmatched_inputs`` discards a buffered target only
    when its timeout report is produced.

    Bound RNG streams are timeline-owned and declared during ``bind`` for
    measurement selection, qstate Bell measurement, heralding decisions,
    physical pattern choice, and per-detector channel sampling.
    """

    device_id: str

    bsm_model: object = field(default_factory=BSMModel)
    detectors: Sequence[SinglePhotonDetector] = ()
    click_patterns: Sequence[ClickPattern] = ()

    measurement: object = field(
        default_factory=lambda: Measure.bell(
            targets=(_LEFT, _RIGHT),
            collapse=True,
        )
    )

    pairing_key: str | None = "bsa_pair_id"
    allow_fifo_fallback: bool = True
    coincidence_window_ticks: int = 0

    consume_matched_inputs: bool = True
    consume_unmatched_inputs: bool = True

    gate_model: GateModel = field(default_factory=AlwaysOpenGate)
    detection_window_ticks: int = 1
    detector_meta: tuple[tuple[str, object], ...] = ()

    output_latency_ticks: int = 0
    output_priority: int = 0
    timeout_priority: int = 10_000

    left_input_port: Port = field(init=False)
    right_input_port: Port = field(init=False)
    output_port: Port = field(init=False)

    reports: list[DetectionReport] = field(init=False, default_factory=list)

    _measurement: Measure = field(init=False, repr=False)
    _bsm_model: BSMModel = field(init=False, repr=False)
    _bound_timeline_id: int | None = field(init=False, default=None, repr=False)

    _measurement_choice_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _qstate_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _bsm_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pattern_choice_rng: object | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _detector_rngs: dict[str, DetectorRNGStreams] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    _buffers: dict[str, list[BufferedBellInput]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.device_id, field_name="device_id")

        self._measurement = Measure.from_spec(self.measurement)
        self._bsm_model = bsm_model_from_spec(self.bsm_model)
        self.bsm_model = self._bsm_model
        self.detectors = normalize_detectors(
            self.detectors,
            require_non_empty=False,
        )
        self.click_patterns = _normalize_click_patterns(
            self.click_patterns,
            detector_ids=tuple(detector.detector_id for detector in self.detectors),
            require_physical=self._bsm_model.kind == "linear_optical",
        )
        validate_gate_model(self.gate_model)
        validate_meta(
            self.detector_meta,
            field_name="detector_meta",
            require_hashable=False,
        )

        if self.pairing_key is not None:
            ensure_nonempty_id(self.pairing_key, field_name="pairing_key")

        validate_bool(self.allow_fifo_fallback, field_name="allow_fifo_fallback")

        if type(self.coincidence_window_ticks) is not int:
            raise TypeError("coincidence_window_ticks must be int")
        if self.coincidence_window_ticks < 0:
            raise ValueError("coincidence_window_ticks must be non-negative")

        validate_bool(
            self.consume_matched_inputs,
            field_name="consume_matched_inputs",
        )

        validate_bool(
            self.consume_unmatched_inputs,
            field_name="consume_unmatched_inputs",
        )

        if type(self.detection_window_ticks) is not int:
            raise TypeError("detection_window_ticks must be int")
        if self.detection_window_ticks <= 0:
            raise ValueError("detection_window_ticks must be positive")

        if type(self.output_latency_ticks) is not int:
            raise TypeError("output_latency_ticks must be int")
        if self.output_latency_ticks < 0:
            raise ValueError("output_latency_ticks must be non-negative")

        if type(self.output_priority) is not int:
            raise TypeError("output_priority must be int")

        if type(self.timeout_priority) is not int:
            raise TypeError("timeout_priority must be int")

        self.left_input_port = Port(
            name=_LEFT,
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )
        self.right_input_port = Port(
            name=_RIGHT,
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.QUANTUM,
            direction=PortDirection.INGRESS,
        )
        self.output_port = Port(
            name="out",
            owner=self,
            owner_id=self.device_id,
            port_kind=PortKind.CLASSICAL,
            direction=PortDirection.EGRESS,
        )

        self._buffers = {
            _LEFT: [],
            _RIGHT: [],
        }

    def bind(self, context: BindingContext) -> None:
        """
        Bind the analyzer to one timeline and declare deterministic RNG streams.

        Binding is idempotent for the same timeline and rejects a different
        timeline. Per-detector streams are declared even when no detector
        channel is configured.
        """

        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "bell state analyzer is already bound to another timeline"
                )
            return

        self._measurement_choice_rng = timeline.rng(
            self.device_id,
            "bell_state_analyzer",
            "measurement_choice",
        )
        self._qstate_rng = timeline.rng(
            self.device_id,
            "bell_state_analyzer",
            "qstate_measurement",
        )
        self._bsm_rng = timeline.rng(
            self.device_id,
            "bell_state_analyzer",
            "bsm_decision",
        )
        self._pattern_choice_rng = timeline.rng(
            self.device_id,
            "bell_state_analyzer",
            "pattern_choice",
        )
        self._detector_rngs = bind_detector_rngs(
            timeline=timeline,
            device_id=self.device_id,
            namespace="bell_state_analyzer",
            detectors=cast(tuple[SinglePhotonDetector, ...], self.detectors),
        )

        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.detectors.bell_state_analyzer.ready",
            "bell state analyzer ready",
            meta={
                "device_id": self.device_id,
                "measurement_label": self._measurement.label,
                "bsm_model": self._bsm_model.kind,
                "heralding_efficiency": self._bsm_model.heralding_efficiency,
                "detector_count": len(self.detectors),
                "click_pattern_count": len(self.click_patterns),
                "coincidence_window_ticks": self.coincidence_window_ticks,
                "detection_window_ticks": self.detection_window_ticks,
                "pairing_key": self.pairing_key,
                "output_latency_ticks": self.output_latency_ticks,
                "output_priority": self.output_priority,
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """
        Handle one Bell-analyzer event dispatched by the timeline.

        Parameters
        ----------
        event : Event
            ``ACTION_RUN_BELL_ANALYSIS`` with ``PortDelivery`` payload or
            ``ACTION_COINCIDENCE_TIMEOUT`` with ``CoincidenceTimeout`` payload.
        timeline : Timeline
            Bound timeline executing the event.
        """

        self._validate_event_context(event=event, timeline=timeline)

        if event.action == ACTION_RUN_BELL_ANALYSIS:
            delivery = self._require_input_delivery(event.payload_ref)
            signal = self._require_signal_payload(delivery.payload)

            self._handle_arrival(
                event=event,
                delivery=delivery,
                signal=signal,
                timeline=timeline,
            )
            return

        if event.action == ACTION_COINCIDENCE_TIMEOUT:
            timeout = self._require_timeout(event.payload_ref)
            self._handle_timeout(
                timeout=timeout,
                timeline=timeline,
                event_id=event.event_id,
                action=event.action,
            )
            return

        raise ValueError(
            f"unsupported event action for BellStateAnalyzer: {event.action!r}"
        )

    def _validate_event_context(
        self,
        *,
        event: Event,
        timeline: Timeline,
    ) -> None:
        if not isinstance(event, Event):
            raise TypeError("event must be Event")

        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")

        if self._bound_timeline_id is None:
            raise RuntimeError(
                "bell state analyzer must be bound before handling events"
            )

        if self._bound_timeline_id != id(timeline):
            raise RuntimeError(
                "bell state analyzer received event for a different timeline"
            )

        if event.target_ref is not self:
            raise ValueError("event target_ref must be this BellStateAnalyzer")

    def _require_input_delivery(self, payload: object) -> PortDelivery:
        if not isinstance(payload, PortDelivery):
            raise TypeError("ACTION_RUN_BELL_ANALYSIS payload_ref must be PortDelivery")

        if payload.target_port not in (self.left_input_port, self.right_input_port):
            raise ValueError(
                "PortDelivery target_port must be this analyzer's left or right port"
            )

        return payload

    def _require_signal_payload(self, payload: object) -> Signal:
        if not isinstance(payload, Signal):
            raise TypeError("PortDelivery payload must be Signal")
        return payload

    def _require_timeout(self, payload: object) -> CoincidenceTimeout:
        if not isinstance(payload, CoincidenceTimeout):
            raise TypeError(
                "ACTION_COINCIDENCE_TIMEOUT payload_ref must be CoincidenceTimeout"
            )
        return payload

    def _handle_arrival(
        self,
        *,
        event: Event,
        delivery: PortDelivery,
        signal: Signal,
        timeline: Timeline,
    ) -> None:
        side = self._side_from_delivery(delivery)
        targets = qstate_targets_from_signal(signal)

        buffered = BufferedBellInput(
            buffer_id=self._make_buffer_id(
                side=side,
                signal=signal,
                event=event,
                time=timeline.current_time,
            ),
            side=side,
            signal=signal,
            targets=targets,
            arrival_time=timeline.current_time,
            pair_key=self._resolve_pair_key(signal),
            source_port_name=delivery.source_port.name,
            connection_id=delivery.connection_id,
            event_id=event.event_id,
        )

        match = self._pop_match_for(buffered)

        if match is None:
            self._buffer_input(buffered)
            self._schedule_timeout(buffered=buffered, timeline=timeline)
            return

        left, right = _ordered_pair(buffered, match)

        report, emit_base_time = self._run_bell_analysis(
            left=left,
            right=right,
            timeline=timeline,
        )

        if self.consume_matched_inputs:
            timeline.qstate.discard(targets=(left.targets[0], right.targets[0]))

        self._store_report(
            report=report,
            timeline=timeline,
            emit_base_time=emit_base_time,
            event_id=event.event_id,
            action=event.action,
            left=left,
            right=right,
        )

    def _side_from_delivery(self, delivery: PortDelivery) -> str:
        if delivery.target_port is self.left_input_port:
            return _LEFT
        if delivery.target_port is self.right_input_port:
            return _RIGHT
        raise ValueError("delivery target_port is not a BSA input port")

    def _make_buffer_id(
        self,
        *,
        side: str,
        signal: Signal,
        event: Event,
        time: int,
    ) -> str:
        return (
            f"{self.device_id}:{side}:{time}:"
            f"{_signal_id(signal)}:{event.event_id}:{self._buffer_count()}"
        )

    def _buffer_count(self) -> int:
        return len(self._buffers[_LEFT]) + len(self._buffers[_RIGHT])

    def _resolve_pair_key(self, signal: Signal) -> object | None:
        if self.pairing_key is None:
            return None

        for key, value in _signal_meta_items(signal):
            if key == self.pairing_key:
                return value

        return None

    def _buffer_input(self, buffered: BufferedBellInput) -> None:
        self._buffers[buffered.side].append(buffered)

    def _pop_match_for(
        self,
        incoming: BufferedBellInput,
    ) -> BufferedBellInput | None:
        opposite_side = _opposite_side(incoming.side)
        candidates = self._buffers[opposite_side]

        for index, candidate in enumerate(candidates):
            delta = abs(incoming.arrival_time - candidate.arrival_time)
            if delta > self.coincidence_window_ticks:
                continue

            has_key = incoming.pair_key is not None or candidate.pair_key is not None

            # Keyed inputs require matching non-None keys; they do not fall
            # back to FIFO against unkeyed inputs.
            if has_key:
                if (
                    incoming.pair_key is not None
                    and incoming.pair_key == candidate.pair_key
                ):
                    return candidates.pop(index)
                continue

            if self.allow_fifo_fallback:
                return candidates.pop(index)

        return None

    def _schedule_timeout(
        self,
        *,
        buffered: BufferedBellInput,
        timeline: Timeline,
    ) -> None:
        # Add one tick so a partner arriving exactly at the configured boundary
        # can be processed before this stale-buffer timeout.
        timeout_time = timeline.current_time + self.coincidence_window_ticks + 1

        timeline.schedule(
            Event(
                time=timeout_time,
                priority=self.timeout_priority,
                target_ref=self,
                action=ACTION_COINCIDENCE_TIMEOUT,
                payload_ref=CoincidenceTimeout(
                    buffer_id=buffered.buffer_id,
                    side=buffered.side,
                ),
                source=self,
                subsystem_id="components",
                meta={
                    "device_id": self.device_id,
                    "buffer_id": buffered.buffer_id,
                    "side": buffered.side,
                    "signal_id": _signal_id(buffered.signal),
                    "bell_state_analyzer": self.device_id,
                },
            )
        )
        timeline.log(
            LogLevel.TRACE,
            "components.detectors.bell_state_analyzer.buffer",
            "bell input buffered",
            event_id=buffered.event_id,
            action=ACTION_RUN_BELL_ANALYSIS,
            meta={
                "device_id": self.device_id,
                "side": buffered.side,
                "signal_id": _signal_id(buffered.signal),
                "buffer_id": buffered.buffer_id,
                "pair_key": buffered.pair_key,
                "coincidence_window_ticks": self.coincidence_window_ticks,
                "timeout_time": timeout_time,
            },
        )

    def _handle_timeout(
        self,
        *,
        timeout: CoincidenceTimeout,
        timeline: Timeline,
        event_id: int | None,
        action: str,
    ) -> None:
        buffered = self._pop_buffer_by_id(
            side=timeout.side,
            buffer_id=timeout.buffer_id,
        )

        if buffered is None:
            return

        if self.consume_unmatched_inputs:
            timeline.qstate.discard(targets=buffered.targets)

        report = self._make_timeout_report(
            time=timeline.current_time,
            buffered=buffered,
        )

        self._store_report(
            report=report,
            timeline=timeline,
            emit_base_time=timeline.current_time,
            event_id=event_id,
            action=action,
            timeout_input=buffered,
        )

    def _pop_buffer_by_id(
        self,
        *,
        side: str,
        buffer_id: str,
    ) -> BufferedBellInput | None:
        candidates = self._buffers[side]

        for index, candidate in enumerate(candidates):
            if candidate.buffer_id == buffer_id:
                return candidates.pop(index)

        return None

    def _run_bell_analysis(
        self,
        *,
        left: BufferedBellInput,
        right: BufferedBellInput,
        timeline: Timeline,
    ) -> tuple[DetectionReport, int]:
        context = self._build_measurement_context(
            time=timeline.current_time,
            left=left,
            right=right,
        )

        call = self._measurement.choose(
            context,
            rng=self._require_measurement_choice_rng(),
        )

        if call.method != "bell":
            raise ValueError("BellStateAnalyzer measurement must resolve to bell")

        # Bell analysis reports a concrete joint outcome, so this component
        # requires the qstate boundary to collapse at measurement time.
        if not call.collapse:
            raise ValueError("BellStateAnalyzer requires collapse=True")

        qstate_result = execute_measurement_call(
            call=call,
            context=context,
            qstate=timeline.qstate,
            rng=self._require_qstate_rng(),
        )

        decision = self._decide_bsm(qstate_result=qstate_result)

        if self._bsm_model.kind == "linear_optical":
            return self._run_linear_optical_bsm(
                time=timeline.current_time,
                left=left,
                right=right,
                call=call,
                qstate_result=qstate_result,
                decision=decision,
            )

        return (
            self._make_bell_report(
                time=timeline.current_time,
                left=left,
                right=right,
                call=call,
                qstate_result=qstate_result,
                decision=decision,
            ),
            timeline.current_time,
        )

    def _run_linear_optical_bsm(
        self,
        *,
        time: int,
        left: BufferedBellInput,
        right: BufferedBellInput,
        call: MeasurementCall,
        qstate_result: object | None,
        decision: BSMDecision,
    ) -> tuple[DetectionReport, int]:
        # A successful decision selects the intended detector pair. All
        # detectors are still evaluated, so unselected channels may dark-count;
        # final success comes from resolving observed clicks back to a pattern.
        selected_pattern = self._select_signal_pattern(decision)
        exposures = self._linear_optical_exposures(selected_pattern=selected_pattern)
        raw_clicks, detection_complete_time = self._evaluate_detector_physics(
            time=time,
            exposures=exposures,
            measurement_call=call,
            selected_pattern=selected_pattern,
        )
        observed_outcome = resolve_click_pattern(
            raw_clicks=raw_clicks,
            patterns=cast(tuple[ClickPattern, ...], self.click_patterns),
        )

        report = self._make_physical_bell_report(
            time=time,
            left=left,
            right=right,
            call=call,
            qstate_result=qstate_result,
            decision=decision,
            selected_pattern=selected_pattern,
            observed_outcome=observed_outcome,
            raw_clicks=raw_clicks,
        )

        return report, detection_complete_time

    def _decide_bsm(self, *, qstate_result: object | None) -> BSMDecision:
        true_label = result_label(qstate_result)
        detectable = self._bsm_model.is_detectable(true_label)

        survived_efficiency = False
        failure_reason: str | None = None

        if detectable:
            survived_efficiency = self._sample_heralding_efficiency()
            if not survived_efficiency:
                failure_reason = "heralding_efficiency_miss"
        else:
            failure_reason = "undetectable_bell_label"

        success = detectable and survived_efficiency
        reported_label = true_label if success else None

        return BSMDecision(
            true_label=true_label,
            detectable=detectable,
            survived_efficiency=survived_efficiency,
            success=success,
            reported_label=reported_label,
            failure_reason=None if success else failure_reason,
            flags=() if success else (FLAG_NO_OUTCOME,),
        )

    def _select_signal_pattern(
        self,
        decision: BSMDecision,
    ) -> ClickPattern | None:
        if not decision.success:
            return None

        patterns = tuple(
            pattern
            for pattern in self.click_patterns
            if pattern.outcome == decision.reported_label
        )

        if not patterns:
            raise RuntimeError(
                f"no click pattern configured for {decision.reported_label!r}"
            )

        return _choose_pattern(patterns, self._require_pattern_choice_rng())

    def _linear_optical_exposures(
        self,
        *,
        selected_pattern: ClickPattern | None,
    ) -> tuple[DetectorExposure, ...]:
        selected_detector_ids = (
            frozenset()
            if selected_pattern is None
            else frozenset(selected_pattern.detector_ids)
        )
        outcome_label = (
            None if selected_pattern is None else str(selected_pattern.outcome)
        )

        return tuple(
            DetectorExposure(
                detector_id=detector.detector_id,
                signal_present=detector.detector_id in selected_detector_ids,
                outcome_label=outcome_label,
            )
            for detector in self.detectors
        )

    def _evaluate_detector_physics(
        self,
        *,
        time: int,
        exposures: tuple[DetectorExposure, ...],
        measurement_call: MeasurementCall,
        selected_pattern: ClickPattern | None,
    ) -> tuple[tuple[RawClick, ...], int]:
        selected_detector_ids = (
            () if selected_pattern is None else tuple(selected_pattern.detector_ids)
        )

        return evaluate_detector_windows(
            device_id=self.device_id,
            time=time,
            detectors=cast(tuple[SinglePhotonDetector, ...], self.detectors),
            exposures=exposures,
            detector_rngs=self._detector_rngs,
            detection_window_ticks=self.detection_window_ticks,
            gate_model=self.gate_model,
            measurement_label=measurement_call.label,
            fallback_complete_time=time + self.detection_window_ticks,
            click_meta=(
                (
                    ("selected_pattern_outcome", selected_pattern.outcome)
                    if selected_pattern is not None
                    else ("selected_pattern_outcome", None)
                ),
                ("selected_pattern_detector_ids", selected_detector_ids),
                *self.detector_meta,
            ),
        )

    def _sample_heralding_efficiency(self) -> bool:
        efficiency = self._bsm_model.heralding_efficiency

        if efficiency <= 0.0:
            return False

        if efficiency >= 1.0:
            return True

        rng = self._require_bsm_rng()

        if not hasattr(rng, "random"):
            raise TypeError("BSM RNG must provide random()")

        return float(cast(_RandomSource, rng).random()) < efficiency

    def _build_measurement_context(
        self,
        *,
        time: int,
        left: BufferedBellInput,
        right: BufferedBellInput,
    ) -> MeasurementContext:
        pair_key = left.pair_key if left.pair_key is not None else right.pair_key

        return MeasurementContext(
            device_id=self.device_id,
            time=time,
            signal=None,
            signal_targets=(left.targets[0], right.targets[0]),
            input_port_name=None,
            detector_meta=(
                ("bsm_model", self._bsm_model.kind),
                ("readout_model", "label"),
                ("left_signal_id", _signal_id(left.signal)),
                ("right_signal_id", _signal_id(right.signal)),
                ("left_arrival_time", left.arrival_time),
                ("right_arrival_time", right.arrival_time),
                ("arrival_delta_ticks", right.arrival_time - left.arrival_time),
                ("coincidence_window_ticks", self.coincidence_window_ticks),
                ("pair_key", pair_key),
            ),
            named_targets=(
                (_LEFT, left.targets[0]),
                (_RIGHT, right.targets[0]),
            ),
        )

    def _make_bell_report(
        self,
        *,
        time: int,
        left: BufferedBellInput,
        right: BufferedBellInput,
        call: MeasurementCall,
        qstate_result: object | None,
        decision: BSMDecision,
    ) -> DetectionReport:
        pair_key = left.pair_key if left.pair_key is not None else right.pair_key
        reported_outcome = decision.reported_label if decision.success else None

        return DetectionReport(
            report_id=(
                f"{self.device_id}:bell:{time}:"
                f"{_signal_id(left.signal)}:{_signal_id(right.signal)}:"
                f"{len(self.reports)}"
            ),
            device_id=self.device_id,
            time=time,
            success=decision.success,
            outcome=reported_outcome,
            raw_clicks=(),
            qstate_result=qstate_result,
            measurement_method=call.method,
            measurement_label=call.label,
            selection_index=call.selection_index,
            selection_probability=call.selection_probability,
            selection_label=call.selection_label,
            signal_id=(_signal_id(left.signal), _signal_id(right.signal)),
            flags=decision.flags,
            meta=(
                ("bsm_model", self._bsm_model.kind),
                ("readout_model", "label"),
                ("true_bell_label", decision.true_label),
                ("reported_bell_label", decision.reported_label),
                ("detectable_bell_label", decision.detectable),
                (
                    "survived_heralding_efficiency",
                    decision.survived_efficiency,
                ),
                ("heralding_efficiency", self._bsm_model.heralding_efficiency),
                ("bsm_failure_reason", decision.failure_reason),
                ("left_signal_id", _signal_id(left.signal)),
                ("right_signal_id", _signal_id(right.signal)),
                ("left_target", str(left.targets[0])),
                ("right_target", str(right.targets[0])),
                ("left_arrival_time", left.arrival_time),
                ("right_arrival_time", right.arrival_time),
                ("arrival_delta_ticks", right.arrival_time - left.arrival_time),
                ("coincidence_window_ticks", self.coincidence_window_ticks),
                ("left_correlation_id", left.signal.correlation_id),
                ("right_correlation_id", right.signal.correlation_id),
                ("pair_key", pair_key),
                ("consume_matched_inputs", self.consume_matched_inputs),
                *call.meta,
            ),
        )

    def _make_physical_bell_report(
        self,
        *,
        time: int,
        left: BufferedBellInput,
        right: BufferedBellInput,
        call: MeasurementCall,
        qstate_result: object | None,
        decision: BSMDecision,
        selected_pattern: ClickPattern | None,
        observed_outcome: object | None,
        raw_clicks: tuple[RawClick, ...],
    ) -> DetectionReport:
        pair_key = left.pair_key if left.pair_key is not None else right.pair_key
        success = observed_outcome is not None
        click_failure_reason = _click_failure_reason(
            raw_clicks=raw_clicks,
            success=success,
        )
        selected_detector_ids = (
            () if selected_pattern is None else tuple(selected_pattern.detector_ids)
        )

        return DetectionReport(
            report_id=(
                f"{self.device_id}:bell:{time}:"
                f"{_signal_id(left.signal)}:{_signal_id(right.signal)}:"
                f"{len(self.reports)}"
            ),
            device_id=self.device_id,
            time=time,
            success=success,
            outcome=observed_outcome,
            raw_clicks=raw_clicks,
            qstate_result=qstate_result,
            measurement_method=call.method,
            measurement_label=call.label,
            selection_index=call.selection_index,
            selection_probability=call.selection_probability,
            selection_label=call.selection_label,
            signal_id=(_signal_id(left.signal), _signal_id(right.signal)),
            flags=_physical_report_flags(
                raw_clicks=raw_clicks,
                success=success,
            ),
            meta=(
                ("bsm_model", self._bsm_model.kind),
                ("readout_model", "linear_optical_click_patterns"),
                ("true_bell_label", decision.true_label),
                ("reported_bell_label", observed_outcome),
                ("click_pattern_outcome", observed_outcome),
                ("detectable_bell_label", decision.detectable),
                (
                    "survived_heralding_efficiency",
                    decision.survived_efficiency,
                ),
                ("heralding_efficiency", self._bsm_model.heralding_efficiency),
                (
                    "bsm_failure_reason",
                    (
                        None
                        if success
                        else decision.failure_reason or click_failure_reason
                    ),
                ),
                ("click_failure_reason", click_failure_reason),
                (
                    "selected_pattern_outcome",
                    None if selected_pattern is None else selected_pattern.outcome,
                ),
                ("selected_pattern_detector_ids", selected_detector_ids),
                (
                    "detector_ids",
                    tuple(detector.detector_id for detector in self.detectors),
                ),
                ("left_signal_id", _signal_id(left.signal)),
                ("right_signal_id", _signal_id(right.signal)),
                ("left_target", str(left.targets[0])),
                ("right_target", str(right.targets[0])),
                ("left_arrival_time", left.arrival_time),
                ("right_arrival_time", right.arrival_time),
                ("arrival_delta_ticks", right.arrival_time - left.arrival_time),
                ("coincidence_window_ticks", self.coincidence_window_ticks),
                ("left_correlation_id", left.signal.correlation_id),
                ("right_correlation_id", right.signal.correlation_id),
                ("pair_key", pair_key),
                ("consume_matched_inputs", self.consume_matched_inputs),
                *call.meta,
            ),
        )

    def _make_timeout_report(
        self,
        *,
        time: int,
        buffered: BufferedBellInput,
    ) -> DetectionReport:
        return DetectionReport(
            report_id=(
                f"{self.device_id}:timeout:{time}:"
                f"{_signal_id(buffered.signal)}:{len(self.reports)}"
            ),
            device_id=self.device_id,
            time=time,
            success=False,
            outcome=None,
            raw_clicks=(),
            qstate_result=None,
            measurement_method="bell",
            measurement_label=self._measurement.label,
            selection_index=None,
            selection_probability=None,
            selection_label=None,
            signal_id=_signal_id(buffered.signal),
            flags=(FLAG_TIMEOUT,),
            meta=(
                ("bsm_model", self._bsm_model.kind),
                ("readout_model", "label"),
                ("reason", "no_coincident_partner"),
                ("side", buffered.side),
                ("signal_id", _signal_id(buffered.signal)),
                ("target", str(buffered.targets[0])),
                ("arrival_time", buffered.arrival_time),
                ("timeout_time", time),
                ("coincidence_window_ticks", self.coincidence_window_ticks),
                ("pair_key", buffered.pair_key),
                ("consume_unmatched_inputs", self.consume_unmatched_inputs),
            ),
        )

    def _store_report(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        emit_base_time: int,
        event_id: int | None,
        action: str,
        left: BufferedBellInput | None = None,
        right: BufferedBellInput | None = None,
        timeout_input: BufferedBellInput | None = None,
    ) -> None:
        self.reports.append(report)
        if left is not None and right is not None:
            self._log_analysis_report(
                report=report,
                timeline=timeline,
                event_id=event_id,
                action=action,
                left=left,
                right=right,
            )
        elif timeout_input is not None:
            self._log_timeout_report(
                report=report,
                timeline=timeline,
                event_id=event_id,
                action=action,
                buffered=timeout_input,
            )
        self._emit_report_if_connected(
            report=report,
            timeline=timeline,
            emit_base_time=emit_base_time,
        )

    def _log_analysis_report(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        event_id: int | None,
        action: str,
        left: BufferedBellInput,
        right: BufferedBellInput,
    ) -> None:
        pair_key = left.pair_key if left.pair_key is not None else right.pair_key
        timeline.log(
            LogLevel.DEBUG,
            "components.detectors.bell_state_analyzer.analysis",
            "bell analysis reported",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "report_id": report.report_id,
                "measurement_label": report.measurement_label,
                "left_signal_id": _signal_id(left.signal),
                "right_signal_id": _signal_id(right.signal),
                "pair_key": pair_key,
                "bsm_model": self._bsm_model.kind,
                "readout_model": _report_meta_value(report, "readout_model"),
                "success": report.success,
                "outcome": report.outcome,
                "true_bell_label": _report_meta_value(report, "true_bell_label"),
                "arrival_delta_ticks": abs(right.arrival_time - left.arrival_time),
                "click_count": len(report.raw_clicks),
                "flags": report.flags,
            },
        )

    def _log_timeout_report(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        event_id: int | None,
        action: str,
        buffered: BufferedBellInput,
    ) -> None:
        timeline.log(
            LogLevel.TRACE,
            "components.detectors.bell_state_analyzer.timeout",
            "bell input timed out",
            event_id=event_id,
            action=action,
            meta={
                "device_id": self.device_id,
                "side": buffered.side,
                "signal_id": _signal_id(buffered.signal),
                "buffer_id": buffered.buffer_id,
                "report_id": report.report_id,
                "measurement_label": report.measurement_label,
                "success": report.success,
                "pair_key": buffered.pair_key,
                "arrival_time": buffered.arrival_time,
                "timeout_time": report.time,
                "coincidence_window_ticks": self.coincidence_window_ticks,
                "flags": report.flags,
            },
        )

    def _emit_report_if_connected(
        self,
        *,
        report: DetectionReport,
        timeline: Timeline,
        emit_base_time: int,
    ) -> None:
        connection: PortConnection | None = self.output_port.connection

        if connection is None:
            return

        ready_time = _report_ready_time(
            report=report,
            fallback_time=emit_base_time,
        )

        connection.transmit(
            report,
            timeline,
            time=max(timeline.current_time, ready_time) + self.output_latency_ticks,
            priority=self.output_priority,
            source=self,
            subsystem_id="components",
            meta={
                "device_id": self.device_id,
                "output_port": self.output_port.name,
                "report_id": report.report_id,
                "signal_id": report.signal_id,
                "bell_state_analyzer": self.device_id,
            },
        )

    def _require_measurement_choice_rng(self) -> object:
        if self._measurement_choice_rng is None:
            raise RuntimeError("measurement choice RNG is not bound")
        return self._measurement_choice_rng

    def _require_qstate_rng(self) -> object:
        if self._qstate_rng is None:
            raise RuntimeError("qstate measurement RNG is not bound")
        return self._qstate_rng

    def _require_bsm_rng(self) -> object:
        if self._bsm_rng is None:
            raise RuntimeError("BSM RNG is not bound")
        return self._bsm_rng

    def _require_pattern_choice_rng(self) -> object:
        if self._pattern_choice_rng is None:
            raise RuntimeError("pattern choice RNG is not bound")
        return self._pattern_choice_rng


def _normalize_click_patterns(
    patterns: Sequence[ClickPattern],
    *,
    detector_ids: tuple[str, ...],
    require_physical: bool,
) -> tuple[ClickPattern, ...]:
    if isinstance(patterns, (str, bytes)) or not isinstance(patterns, Sequence):
        raise TypeError("click_patterns must be a sequence of ClickPattern")

    if require_physical and not detector_ids:
        raise ValueError("linear_optical BSM requires detectors")

    detector_id_set = frozenset(detector_ids)
    normalized = tuple(
        _normalize_click_pattern(pattern, detector_ids=detector_id_set)
        for pattern in patterns
    )
    _validate_unambiguous_click_patterns(normalized)

    if require_physical:
        if not normalized:
            raise ValueError("linear_optical BSM requires click_patterns")

        outcomes = {pattern.outcome for pattern in normalized}
        for label in _LINEAR_OPTICAL_LABELS:
            if label not in outcomes:
                raise ValueError(
                    "linear_optical BSM requires psi+ and psi- click patterns"
                )

    return normalized


def _validate_unambiguous_click_patterns(
    patterns: tuple[ClickPattern, ...],
) -> None:
    outcome_by_detector_pair: dict[tuple[str, str], object] = {}
    seen_patterns: set[tuple[object, tuple[str, str]]] = set()

    for pattern in patterns:
        detector_a, detector_b = sorted(pattern.detector_ids)
        detector_pair = (detector_a, detector_b)
        pattern_key = (pattern.outcome, detector_pair)

        if pattern_key in seen_patterns:
            raise ValueError(
                "click_patterns must not contain duplicate outcome/detector pair "
                "mappings"
            )

        seen_patterns.add(pattern_key)
        previous_outcome = outcome_by_detector_pair.get(detector_pair)

        if previous_outcome is not None and previous_outcome != pattern.outcome:
            raise ValueError(
                "click_patterns must not map the same detector pair "
                "to different outcomes"
            )

        outcome_by_detector_pair[detector_pair] = pattern.outcome


def _normalize_click_pattern(
    pattern: ClickPattern,
    *,
    detector_ids: frozenset[str],
) -> ClickPattern:
    if not isinstance(pattern, ClickPattern):
        raise TypeError("click_patterns must contain ClickPattern")

    if pattern.outcome not in _LINEAR_OPTICAL_LABELS:
        raise ValueError("linear_optical click pattern outcome must be psi+ or psi-")

    if not isinstance(pattern.detector_ids, tuple):
        raise TypeError("click pattern detector_ids must be tuple")

    if len(pattern.detector_ids) != 2:
        raise ValueError("linear_optical click patterns must contain two detectors")

    if len(set(pattern.detector_ids)) != 2:
        raise ValueError("linear_optical click pattern detectors must be distinct")

    for detector_id in pattern.detector_ids:
        ensure_nonempty_id(detector_id, field_name="click pattern detector_id")
        if detector_id not in detector_ids:
            raise ValueError(f"unknown click pattern detector_id: {detector_id!r}")

    return ClickPattern(
        outcome=pattern.outcome,
        detector_ids=tuple(pattern.detector_ids),
    )


def _choose_pattern(
    patterns: tuple[ClickPattern, ...],
    rng: object,
) -> ClickPattern:
    if hasattr(rng, "choice"):
        return rng.choice(patterns)

    if not hasattr(rng, "random"):
        raise TypeError("BSM RNG must provide choice() or random()")

    draw = float(cast(_RandomSource, rng).random())
    index = min(int(draw * len(patterns)), len(patterns) - 1)
    return patterns[index]


def _physical_report_flags(
    *,
    raw_clicks: tuple[RawClick, ...],
    success: bool,
) -> tuple[str, ...]:
    # Two clicks can still fail when they do not match a configured pattern.
    # More than two clicks is reported as a double-click style failure.
    if success:
        return _raw_click_flags(raw_clicks)

    if not raw_clicks:
        return (FLAG_NO_CLICK, FLAG_NO_OUTCOME)

    flags = [FLAG_NO_OUTCOME]
    if len(raw_clicks) > 2:
        flags.append(FLAG_DOUBLE_CLICK)

    return tuple(flags)


def _raw_click_flags(raw_clicks: tuple[RawClick, ...]) -> tuple[str, ...]:
    flags: list[str] = []

    for click in raw_clicks:
        for flag in click.flags:
            if flag not in flags:
                flags.append(flag)

    return tuple(flags)


def _click_failure_reason(
    *,
    raw_clicks: tuple[RawClick, ...],
    success: bool,
) -> str | None:
    # Mirrors _physical_report_flags: no clicks, too many clicks, or a two-click
    # pattern that did not resolve to a configured Bell outcome.
    if success:
        return None

    if not raw_clicks:
        return "no_click"

    if len(raw_clicks) > 2:
        return "too_many_clicks"

    return "unresolved_click_pattern"


def _report_ready_time(
    *,
    report: DetectionReport,
    fallback_time: int,
) -> int:
    if type(fallback_time) is not int:
        raise TypeError("fallback_time must be int")
    if fallback_time < 0:
        raise ValueError("fallback_time must be non-negative")

    if report.raw_clicks:
        return max(click.time for click in report.raw_clicks)

    # Label-only BSM reports use measurement/timeout time; physical reports
    # without clicks use the detector-window completion fallback.
    return fallback_time


def _opposite_side(side: str) -> str:
    if side == _LEFT:
        return _RIGHT
    if side == _RIGHT:
        return _LEFT
    raise ValueError("side must be 'left' or 'right'")


def _ordered_pair(
    first: BufferedBellInput,
    second: BufferedBellInput,
) -> tuple[BufferedBellInput, BufferedBellInput]:
    if first.side == _LEFT and second.side == _RIGHT:
        return first, second

    if first.side == _RIGHT and second.side == _LEFT:
        return second, first

    raise ValueError("Bell pair must contain one left and one right input")


def _signal_id(signal: Signal) -> object:
    signal_id = getattr(signal, "signal_id", None)
    if signal_id is not None:
        return signal_id
    return signal.id


def _report_meta_value(report: DetectionReport, key: str) -> object | None:
    for meta_key, value in report.meta:
        if meta_key == key:
            return value
    return None


def _signal_meta_items(signal: Signal):
    # Pair-key lookup uses first-match precedence across increasingly specific
    # signal metadata channels.
    for items in (
        signal.protocol_params,
        signal.meta,
        signal.correlation_meta,
        signal.timing_meta,
    ):
        if items is None:
            continue

        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                continue

            key, value = item
            if isinstance(key, str):
                yield key, value


__all__ = [
    "BSMDecision",
    "BSMModel",
    "BellStateAnalyzer",
    "BufferedBellInput",
    "CoincidenceTimeout",
    "bsm_model_from_spec",
]
