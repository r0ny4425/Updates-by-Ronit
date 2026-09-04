"""Quantum-signal transport component.

``QuantumChannel`` receives qstate-backed ``Signal`` payloads, applies
channel-level survival, optional qstate noise, timing metadata, and delayed
delivery through its quantum output port. Qstate math stays in the qstate layer;
the channel only invokes configured noise models on explicit signal targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import floor
from typing import TYPE_CHECKING

from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.units import ticks_to_seconds
from simyuj.primitives.validation import (
    require_non_negative_real,
    require_positive_real,
)
from simyuj.qstate.noise import NoiseModel
from simyuj.runtime.binding import BindingContext
from simyuj.signal import Signal
from simyuj.tracing.levels import LogLevel

from ..coherent_optics import attenuated, phase_shifted
from ..connections import PortDelivery, require_connection
from ..ports import Port, PortKind
from ..quantum_targets import qstate_payload_role, qstate_targets_from_signal
from ._common import _create_channel_ports, _non_negative_int, _resolve_delay_ticks

if TYPE_CHECKING:
    from simyuj.engine.rng_manager import DeterministicRNG
    from simyuj.engine.timeline import Timeline


ACTION_TRANSMIT_QUANTUM = "transmit_quantum"
DEFAULT_FIBER_LIGHT_SPEED_M_PER_S = 2.0e8


def _survival_probability_from_loss(
    *,
    length_m: float,
    attenuation_db_per_km: float,
    fixed_insertion_loss_db: float,
) -> float:
    """Return fiber survival probability from length and insertion loss.

    The implemented model is:

    .. math::

       10^{-L_{\\mathrm{total}} / 10}

    where:

    .. math::

       L_{\\mathrm{total}} =
       a_{\\mathrm{dB/km}} L_{\\mathrm{km}} + L_{\\mathrm{fixed}}
    """
    length_km = length_m / 1000.0
    total_loss_db = attenuation_db_per_km * length_km + fixed_insertion_loss_db
    return 10.0 ** (-total_loss_db / 10.0)


@dataclass(slots=True)
class QuantumChannel(Component):
    """Transport qstate-backed quantum signals through component ports.

    ``QuantumChannel`` accepts a port-routed ``Signal`` on its quantum input
    port, samples channel loss, optionally applies configured qstate noise
    models to the signal target, appends channel metadata to surviving signals,
    and schedules downstream delivery from its quantum output port.

    Parameters
    ----------
    channel_id : str
        Non-empty channel identifier used for port ownership, logs, metadata,
        and RNG stream names.
    length_m : float, default=0.0
        Fiber length in meters used for length-derived delay and attenuation.
    delay_ticks : int, optional
        Explicit non-negative propagation delay in simulation ticks. When set,
        it overrides the length-derived delay.
    propagation_speed_m_per_s : float, default=DEFAULT_FIBER_LIGHT_SPEED_M_PER_S
        Positive propagation speed used for length-derived delay.
    attenuation_db_per_km : float, default=0.0
        Non-negative attenuation coefficient in decibels per kilometer.
    fixed_insertion_loss_db : float, default=0.0
        Non-negative fixed insertion loss in decibels.
    noise_models : Sequence[NoiseModel], default=()
        Qstate noise models applied to surviving signals after jitter is
        sampled and before downstream delivery is scheduled.
    timing_jitter_stddev_ticks : float, default=0.0
        Non-negative standard deviation for one-sided timing jitter sampling in
        simulation ticks.
    session_id : str, optional
        Optional session identifier copied into downstream event metadata.
    delivery_priority : int, default=0
        Priority used for the scheduled downstream event.
    phase_noise_stddev_rad : float, default=0.0
        Non-negative standard deviation of the per-pulse optical phase shift
        applied on the coherent-amplitude path. Zero consumes no randomness and
        is the default. Ignored by qstate-backed signals, which carry no
        optical phase.

    Attributes
    ----------
    input_port : Port
        Quantum input port named ``"in"``.
    output_port : Port
        Quantum output port named ``"out"``.

    Notes
    -----
    ``ACTION_TRANSMIT_QUANTUM`` is the only action handled by this component.
    Its ``payload_ref`` must be a ``PortDelivery`` whose payload is a
    qstate-backed ``Signal`` and whose target port is this channel's input
    port.

    Propagation is represented by the scheduled time of the downstream event.
    The channel does not schedule an internal propagation event.

    The survival model is a Bernoulli event with probability
    :math:`10^{-L_{\\mathrm{total}} / 10}`, where
    :math:`L_{\\mathrm{total}}` combines fiber attenuation over ``length_m``
    and fixed insertion loss. Lost signals are not forwarded; their qstate
    targets are discarded through the timeline qstate manager.

    Surviving signals preserve their qstate subsystem identity. The dataclass
    record is replaced only to append ``meta`` and ``timing_meta`` entries for
    channel identity, survival probability, delay, jitter, duration, and
    arrival time.

    Timing jitter is sampled from a normal draw, rounded to ticks, and clamped
    at zero. It can only delay arrival; it never makes a signal arrive early.

    Configured ``noise_models`` are applied in order to surviving signal
    targets before downstream delivery is scheduled.

    Fully wire the channel before execution. For surviving signals, this
    component applies qstate effects before checking the output connection, so
    the output port is assumed to be connected in a valid run setup.

    ``bind()`` declares timeline-owned RNG streams
    ``(channel_id, "quantum_channel", "loss")``,
    ``(channel_id, "quantum_channel", "timing")`` and
    ``(channel_id, "quantum_channel", "phase")`` before execution. A survival
    probability of one avoids consuming the loss stream, zero jitter avoids
    consuming the timing stream, and zero phase noise avoids consuming the
    phase stream.

    Coherent-amplitude transport
    ----------------------------

    A ``Signal`` carrying a ``coherent_state`` takes a second path, chosen by
    the **role** of its qstate record rather than by whether it has one -- see
    ``qstate_payload_role``. On that path:

    - ``eta`` is a **power transmission** rather than a survival probability,
      so ``alpha -> sqrt(eta) * alpha`` and ``mu -> eta * mu``. It is the same
      :math:`10^{-L/10}`; one fibre property with two correct consequences for
      two different input states, and there is no second loss field.
    - Attenuation is deterministic. The loss stream is never consumed, so an
      all-amplitude run replays identically at any ``master_seed`` whatever the
      configured loss.
    - Nothing is discarded, so ``lost_count`` stays 0 and
      ``delivered_count == received_count``. ``attenuated_count`` is the
      counter that moves.
    - ``timing_jitter_stddev_ticks`` must be zero, and ``noise_models`` must be
      empty unless the signal carries a ``"mode"`` record. Both are rejected at
      event time, because a channel cannot know at construction what it will
      carry.
    - ``phase_noise_stddev_rad`` applies an optical phase shift per pulse.

    A ``"mode"`` record beside the amplitude -- a polarization state occupying
    the pulse's mode -- takes ``noise_models`` through the ordinary Kraus path
    and is **not** discarded by loss.

    This is a compact signal-level fiber model. It does not model pulse shape,
    dispersion, polarization drift, multi-photon optical modes, congestion, or
    detector behavior. Per-pulse phase noise is independent rather than
    correlated between neighbours; see ``CAPABILITY_MAP.md`` section 5.
    """

    channel_id: str

    length_m: float = 0.0
    delay_ticks: int | None = None
    propagation_speed_m_per_s: float = DEFAULT_FIBER_LIGHT_SPEED_M_PER_S

    attenuation_db_per_km: float = 0.0
    fixed_insertion_loss_db: float = 0.0

    noise_models: Sequence[NoiseModel] = field(default_factory=tuple)

    timing_jitter_stddev_ticks: float = 0.0

    session_id: str | None = None
    delivery_priority: int = 0

    # Appended after the existing fields rather than grouped with the other
    # physics knobs: this one applies only to the coherent-amplitude path.
    phase_noise_stddev_rad: float = 0.0

    input_port: Port = field(init=False)
    output_port: Port = field(init=False)

    _resolved_delay_ticks: int = field(init=False)
    _survival_probability_value: float = field(init=False)

    _received_count: int = field(init=False, default=0)
    _delivered_count: int = field(init=False, default=0)
    _lost_count: int = field(init=False, default=0)
    _attenuated_count: int = field(init=False, default=0)

    _bound_timeline_id: int | None = field(init=False, default=None)
    _loss_rng: DeterministicRNG | None = field(init=False, default=None)
    _timing_rng: DeterministicRNG | None = field(init=False, default=None)
    _phase_rng: DeterministicRNG | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        (
            length_m,
            propagation_speed_m_per_s,
            attenuation_db_per_km,
            fixed_insertion_loss_db,
        ) = self._validate_config()

        self._initialize_derived_values(
            length_m=length_m,
            propagation_speed_m_per_s=propagation_speed_m_per_s,
            attenuation_db_per_km=attenuation_db_per_km,
            fixed_insertion_loss_db=fixed_insertion_loss_db,
        )
        if isinstance(self.noise_models, (str, bytes)) or not isinstance(
            self.noise_models,
            Sequence,
        ):
            raise TypeError("QuantumChannel.noise_models must be a sequence")

        self.noise_models = tuple(self.noise_models)

        self.input_port, self.output_port = _create_channel_ports(
            owner=self,
            owner_id=self.channel_id,
            port_kind=PortKind.QUANTUM,
        )

    def _validate_config(self) -> tuple[float, float, float, float]:
        ensure_nonempty_id(self.channel_id, field_name="channel_id")

        length_m = require_non_negative_real(
            self.length_m,
            field_name="length_m",
            type_name="numeric",
        )

        if self.delay_ticks is not None:
            _non_negative_int("delay_ticks", self.delay_ticks)

        propagation_speed_m_per_s = require_positive_real(
            self.propagation_speed_m_per_s,
            field_name="propagation_speed_m_per_s",
            type_name="numeric",
        )

        attenuation_db_per_km = require_non_negative_real(
            self.attenuation_db_per_km,
            field_name="attenuation_db_per_km",
            type_name="numeric",
        )
        fixed_insertion_loss_db = require_non_negative_real(
            self.fixed_insertion_loss_db,
            field_name="fixed_insertion_loss_db",
            type_name="numeric",
        )

        require_non_negative_real(
            self.timing_jitter_stddev_ticks,
            field_name="timing_jitter_stddev_ticks",
            type_name="numeric",
        )

        require_non_negative_real(
            self.phase_noise_stddev_rad,
            field_name="phase_noise_stddev_rad",
            type_name="numeric",
        )

        if self.session_id is not None:
            ensure_nonempty_id(self.session_id, field_name="session_id")

        if type(self.delivery_priority) is not int:
            raise TypeError("delivery_priority must be int")

        return (
            length_m,
            propagation_speed_m_per_s,
            attenuation_db_per_km,
            fixed_insertion_loss_db,
        )

    def _initialize_derived_values(
        self,
        *,
        length_m: float,
        propagation_speed_m_per_s: float,
        attenuation_db_per_km: float,
        fixed_insertion_loss_db: float,
    ) -> None:
        self._survival_probability_value = _survival_probability_from_loss(
            length_m=length_m,
            attenuation_db_per_km=attenuation_db_per_km,
            fixed_insertion_loss_db=fixed_insertion_loss_db,
        )
        self._resolved_delay_ticks = _resolve_delay_ticks(
            self.delay_ticks,
            length_m=length_m,
            speed_m_per_s=propagation_speed_m_per_s,
        )

    @property
    def received_count(self) -> int:
        """Number of quantum transmission events accepted by the channel."""
        return self._received_count

    @property
    def delivered_count(self) -> int:
        """Number of surviving signals scheduled for downstream delivery."""
        return self._delivered_count

    @property
    def lost_count(self) -> int:
        """Number of accepted signals lost and discarded by the channel.

        Only qubit carriers are ever counted here. A coherent pulse faces no
        Bernoulli trial, so ``lost_count == 0`` on an all-amplitude run means
        "nothing was discarded", **not** "the fibre was lossless" -- see
        :attr:`attenuated_count`.
        """
        return self._lost_count

    @property
    def attenuated_count(self) -> int:
        """Number of coherent pulses scaled by the fibre's power transmission.

        The amplitude counterpart of :attr:`lost_count`. Loss is invisible in
        the delivered/received counters on that path, because attenuation is
        arithmetic rather than a discard.
        """
        return self._attenuated_count

    @property
    def resolved_delay_ticks(self) -> int:
        """Resolved non-negative base propagation delay in simulation ticks."""
        return self._resolved_delay_ticks

    @property
    def survival_probability(self) -> float:
        """Cached Bernoulli survival probability for each accepted signal."""
        return self._survival_probability_value

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
        first transmission event executes. The declared loss and timing streams
        keep channel loss and jitter reproducible under fixed seeds and fixed
        configuration.
        """
        if not isinstance(context, BindingContext):
            raise TypeError("context must be BindingContext")

        timeline = context.timeline
        timeline_id = id(timeline)

        if self._bound_timeline_id is not None:
            if self._bound_timeline_id != timeline_id:
                raise RuntimeError(
                    "quantum channel is already bound to another timeline"
                )
            return

        self._loss_rng = timeline.rng(
            self.channel_id,
            "quantum_channel",
            "loss",
        )
        self._timing_rng = timeline.rng(
            self.channel_id,
            "quantum_channel",
            "timing",
        )
        # Declared unconditionally, and never drawn at the 0.0 default.
        # ``Timeline.rng`` refuses a new stream once execution begins, and
        # declaring one perturbs no existing replay: RNGManager seeds each
        # stream from its path, never from creation order or stream count.
        self._phase_rng = timeline.rng(
            self.channel_id,
            "quantum_channel",
            "phase",
        )
        self._bound_timeline_id = timeline_id
        timeline.log(
            LogLevel.INFO,
            "components.channels.quantum.ready",
            "quantum channel ready",
            meta={
                "channel_id": self.channel_id,
                "delay_ticks": self._resolved_delay_ticks,
                "survival_probability": self._survival_probability_value,
                "timing_jitter_stddev_ticks": float(self.timing_jitter_stddev_ticks),
                "delivery_priority": self.delivery_priority,
            },
        )

    def handle_event(self, event: Event, timeline: Timeline) -> None:
        """Handle a port-routed quantum transmission event.

        Parameters
        ----------
        event : Event
            Timeline event with action ``ACTION_TRANSMIT_QUANTUM`` and
            ``PortDelivery`` payload.
        timeline : Timeline
            Timeline currently executing the event.

        Raises
        ------
        RuntimeError
            If the channel has not been bound before execution.
        TypeError
            If the payload wrapper or wrapped signal has the wrong type.
        ValueError
            If the action is unsupported or the delivery targets another port.
        """
        if self._bound_timeline_id is None:
            raise RuntimeError("quantum channel must be bound before execution")

        if event.action != ACTION_TRANSMIT_QUANTUM:
            raise ValueError(
                f"unsupported event action for quantum channel: {event.action!r}"
            )

        if not isinstance(event.payload_ref, PortDelivery):
            raise TypeError("ACTION_TRANSMIT_QUANTUM payload_ref must be PortDelivery")

        delivery = event.payload_ref

        if delivery.target_port is not self.input_port:
            raise ValueError("quantum channel received delivery on unknown port")

        if not isinstance(delivery.payload, Signal):
            raise TypeError("delivery payload must be Signal")

        self._transmit_now(
            timeline,
            signal=delivery.payload,
            event_id=event.event_id,
            action=event.action,
        )

    def _transmit_now(
        self,
        timeline: Timeline,
        *,
        signal: Signal,
        event_id: int | None,
        action: str,
    ) -> None:
        """Apply channel loss, qstate noise, metadata, and delivery scheduling.

        Notes
        -----
        The two transport models are split on the **role** a signal's qstate
        record plays, not on whether it has one -- see ``qstate_payload_role``.
        """
        role = qstate_payload_role(signal)

        if signal.coherent_state is not None:
            self._transmit_coherent_now(
                timeline,
                signal=signal,
                role=role,
                event_id=event_id,
                action=action,
            )
            return

        if role == "mode":
            raise ValueError(
                "signal carries a mode descriptor with no amplitude occupying "
                "it; a mode record describes the mode of a coherent_state and "
                "cannot travel alone"
            )

        self._received_count += 1

        targets = qstate_targets_from_signal(signal)

        if self._is_lost():
            self._lost_count += 1
            timeline.qstate.discard(targets=targets)
            timeline.log(
                LogLevel.DEBUG,
                "components.channels.quantum.signal_lost",
                "quantum signal lost",
                event_id=event_id,
                action=action,
                meta={
                    "channel_id": self.channel_id,
                    "signal_id": signal.id,
                    "signal_kind": signal.signal_kind.value,
                    "received_index": self._received_count,
                    "survival_probability": self._survival_probability_value,
                },
            )
            return

        jitter_ticks = self._sample_timing_jitter_ticks()
        duration_ticks = self._resolved_delay_ticks + jitter_ticks
        duration_s = ticks_to_seconds(duration_ticks)
        arrival_time = timeline.current_time + duration_ticks

        if self.noise_models:
            timeline.qstate.apply_noise_models(
                self.noise_models,
                targets=targets,
                duration_s=duration_s,
            )

        updated_signal = self._with_channel_metadata(
            signal,
            arrival_time=arrival_time,
            duration_ticks=duration_ticks,
            duration_s=duration_s,
            timing_jitter_ticks=jitter_ticks,
        )
        event_meta = {
            "channel_id": self.channel_id,
            "signal_id": signal.id,
        }
        if self.session_id is not None:
            event_meta["session_id"] = self.session_id

        output_connection = require_connection(self.output_port)
        timeline.log(
            LogLevel.DEBUG,
            "components.channels.quantum.signal_forwarded",
            "quantum signal forwarded",
            event_id=event_id,
            action=action,
            meta={
                "channel_id": self.channel_id,
                "signal_id": signal.id,
                "signal_kind": signal.signal_kind.value,
                "received_index": self._received_count,
                "connection_id": output_connection.connection_id,
                "delay_ticks": self._resolved_delay_ticks,
                "jitter_ticks": jitter_ticks,
                "duration_ticks": duration_ticks,
                "arrival_time": arrival_time,
            },
        )
        output_connection.transmit(
            updated_signal,
            timeline,
            time=arrival_time,
            priority=self.delivery_priority,
            source=self,
            subsystem_id="components",
            meta=event_meta,
        )

        self._delivered_count += 1

    def _sample_timing_jitter_ticks(self) -> int:
        """Sample non-negative timing jitter in simulation ticks."""
        stddev = float(self.timing_jitter_stddev_ticks)
        if stddev == 0.0:
            return 0

        timing_rng = self._timing_rng
        if timing_rng is None:
            raise RuntimeError("quantum channel must be bound before execution")

        draw = timing_rng.normal(loc=0.0, scale=stddev)
        return max(0, floor(draw + 0.5))

    def _is_lost(self) -> bool:
        """Return whether the current transmission is lost."""
        if self._survival_probability_value >= 1.0:
            return False

        loss_rng = self._loss_rng
        if loss_rng is None:
            raise RuntimeError("quantum channel must be bound before execution")

        return loss_rng.random() >= self._survival_probability_value

    def _sample_phase_noise_rad(self) -> float:
        """Sample an optical phase shift in radians for one coherent pulse."""
        stddev = float(self.phase_noise_stddev_rad)
        if stddev == 0.0:
            return 0.0

        phase_rng = self._phase_rng
        if phase_rng is None:
            raise RuntimeError("quantum channel must be bound before execution")

        return float(phase_rng.normal(loc=0.0, scale=stddev))

    def _require_coherent_transport_supported(
        self,
        *,
        role: str | None,
    ) -> None:
        """Reject configurations this channel cannot honour for an amplitude.

        Notes
        -----
        Checked at event time rather than at construction, because a channel
        cannot know at construction which payloads it will be asked to carry.
        """
        if role == "qubit":
            raise ValueError(
                "signal carries both a qubit carrier and an optical amplitude; "
                "the carrier must be one or the other. A qstate record that "
                "describes the mode of a coherent pulse belongs on a handle "
                "with kind='mode'"
            )

        if float(self.timing_jitter_stddev_ticks) > 0.0:
            raise ValueError(
                "timing_jitter_stddev_ticks is not supported for "
                "coherent-amplitude transport: an independent draw per pulse "
                "makes adjacent-pulse spacing wander and can reorder pulses, "
                "which a phase-encoded receiver depends on. Set "
                "timing_jitter_stddev_ticks=0.0 on a channel carrying pulses"
            )

        if self.noise_models and role is None:
            raise ValueError(
                "noise_models cannot act on an optical amplitude alone. A "
                "noise model is a set of Kraus operators shaped "
                "(2**arity, 2**arity) for qubit axes; a coherent amplitude is "
                "one complex number and has no such axis, so there is nothing "
                "for them to act on and applying them silently would be worse "
                "than refusing. A fibre configured for single photons "
                "therefore cannot be reused unchanged for pulses: drop "
                "noise_models and set phase_noise_stddev_rad instead, which is "
                "the optical-phase equivalent. Polarization noise on a pulse "
                "reaches its qstate record through the ordinary noise_models "
                "path and needs a handle with kind='mode'"
            )

    def _transmit_coherent_now(
        self,
        timeline: Timeline,
        *,
        signal: Signal,
        role: str | None,
        event_id: int | None,
        action: str,
    ) -> None:
        """Attenuate one coherent pulse and schedule its delivery.

        Notes
        -----
        Attenuation is **arithmetic, not a trial**: the loss stream is never
        consumed, so an all-amplitude run replays identically whatever the fibre
        loss is. ``eta`` is one fibre property with two correct consequences --
        a survival probability for a photon, a power transmission for an
        amplitude -- so there is no second loss field.

        Nothing is discarded, so ``lost_count`` stays 0 and
        ``delivered_count == received_count`` however lossy the fibre is.
        Reading ``lost_count == 0`` as "lossless" is a trap.
        """
        self._received_count += 1
        self._require_coherent_transport_supported(role=role)

        incoming = signal.coherent_state
        assert incoming is not None
        power_transmission = self._survival_probability_value
        state = attenuated(incoming, power_transmission=power_transmission)

        # No jitter: rejected above for this path, so spacing is exactly the
        # emission spacing and adjacent pulses stay adjacent.
        duration_ticks = self._resolved_delay_ticks
        duration_s = ticks_to_seconds(duration_ticks)
        arrival_time = timeline.current_time + duration_ticks

        phase_noise_rad = self._sample_phase_noise_rad()
        if phase_noise_rad != 0.0:
            state = phase_shifted(state, phase_rad=phase_noise_rad)

        if role is not None and self.noise_models:
            timeline.qstate.apply_noise_models(
                self.noise_models,
                targets=qstate_targets_from_signal(signal),
                duration_s=duration_s,
            )

        updated_signal = signal._derived(
            coherent_state=state,
            meta=signal.meta
            + (
                ("quantum_channel_id", self.channel_id),
                # Not "survival_probability": a pulse that faced no Bernoulli
                # trial must not carry a record claiming it did.
                ("channel_power_transmission", power_transmission),
            ),
            timing_meta=signal.timing_meta
            + (
                ("channel_delay_ticks", self._resolved_delay_ticks),
                ("channel_timing_jitter_ticks", 0),
                ("channel_duration_ticks", duration_ticks),
                ("channel_duration_s", duration_s),
                ("channel_arrival_time", arrival_time),
            ),
        )

        event_meta = {
            "channel_id": self.channel_id,
            "signal_id": signal.id,
        }
        if self.session_id is not None:
            event_meta["session_id"] = self.session_id

        output_connection = require_connection(self.output_port)
        self._attenuated_count += 1
        timeline.log(
            LogLevel.DEBUG,
            # A separate topic from ``signal_forwarded``: the two records carry
            # different keys, and the qstate one is asserted by exact equality.
            "components.channels.quantum.pulse_forwarded",
            "coherent pulse forwarded",
            event_id=event_id,
            action=action,
            meta={
                "channel_id": self.channel_id,
                "signal_id": signal.id,
                "signal_kind": signal.signal_kind.value,
                "received_index": self._received_count,
                "connection_id": output_connection.connection_id,
                "delay_ticks": self._resolved_delay_ticks,
                "duration_ticks": duration_ticks,
                "arrival_time": arrival_time,
                # Loss is invisible in the counters on this path, so it has to
                # be visible here. Derived floats only -- the JSONL sink has no
                # complex case and would fall back to repr().
                "channel_power_transmission": power_transmission,
                "mean_photon_number_in": incoming.mean_photon_number,
                "mean_photon_number_out": state.mean_photon_number,
                "phase_noise_rad": phase_noise_rad,
                "qstate_payload_role": role,
            },
        )
        output_connection.transmit(
            updated_signal,
            timeline,
            time=arrival_time,
            priority=self.delivery_priority,
            source=self,
            subsystem_id="components",
            meta=event_meta,
        )

        self._delivered_count += 1

    def _with_channel_metadata(
        self,
        signal: Signal,
        *,
        arrival_time: int,
        duration_ticks: int,
        duration_s: float,
        timing_jitter_ticks: int,
    ) -> Signal:
        """Return ``signal`` with channel metadata appended."""
        return signal._with_metadata(
            meta=signal.meta
            + (
                ("quantum_channel_id", self.channel_id),
                ("survival_probability", self._survival_probability_value),
            ),
            timing_meta=signal.timing_meta
            + (
                ("channel_delay_ticks", self._resolved_delay_ticks),
                ("channel_timing_jitter_ticks", timing_jitter_ticks),
                ("channel_duration_ticks", duration_ticks),
                ("channel_duration_s", duration_s),
                ("channel_arrival_time", arrival_time),
            ),
        )


__all__ = [
    "ACTION_TRANSMIT_QUANTUM",
    "DEFAULT_FIBER_LIGHT_SPEED_M_PER_S",
    "QuantumChannel",
]
