"""Measurement-selection and qstate execution primitives for detectors.

Detector components use this module to choose a ``MeasurementCall``, resolve
target specifications against a local ``MeasurementContext``, and execute the
chosen operation through ``QuantumStateManager``. The module defines the qstate
boundary but does not own component ports or event scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from typing import Literal, Protocol, cast

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import validate_bool
from simyuj.qstate import QuantumStateManager, SubsystemId
from simyuj.qstate.measure import POVM, MeasurementBasis, basis_for
from simyuj.signal import Signal

MeasurementMethod = Literal["projective", "povm", "bell", "none"]
TargetSpec = object

_SIGNAL_TARGET = "signal"
_VALID_MEASUREMENT_METHODS = frozenset(("projective", "povm", "bell", "none"))


class _RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MeasurementCall:
    """
    Immutable qstate measurement request selected for one detector operation.

    Parameters
    ----------
    method : {"projective", "povm", "bell", "none"}
        Measurement method executed by ``execute_measurement_call``.
    operator : object or None, default=None
        Basis or POVM object required by projective and POVM measurements.
    targets : object, default="signal"
        Target spec resolved against ``MeasurementContext``. The sentinel
        ``"signal"`` means all signal targets.
    collapse : bool, default=True
        Whether qstate measurement collapses the measured subsystem(s).
    label : str or None, default=None
        Human-readable measurement label used by readout mappings and reports.
    discard : bool, default=False
        For ``method="none"``, whether the targets are discarded without a
        qstate measurement.
    selection_index, selection_probability, selection_label : optional
        Metadata filled when ``Measure.random`` selects among choices.
    meta : tuple[tuple[str, object], ...], default=()
        Extra metadata copied into downstream reports where the caller includes
        it.

    Notes
    -----
    This record describes what to do at the qstate boundary. It does not hold
    qstate ownership and does not execute measurement by itself. Operator
    shape depends on ``method``: projective measurements expect a basis name or
    ``MeasurementBasis``, POVM measurements expect ``POVM``, and ``"bell"`` or
    ``"none"`` calls do not use an operator. ``discard`` is only meaningful for
    ``method="none"``; measured-state lifetime for other methods is controlled
    by ``collapse``.
    """

    method: MeasurementMethod
    operator: object | None = None
    targets: object = _SIGNAL_TARGET
    collapse: bool = True
    label: str | None = None
    discard: bool = False

    selection_index: int | None = None
    selection_probability: float | None = None
    selection_label: str | None = None

    meta: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _validate_measurement_method(self.method)

    def _with_selection(
        self,
        *,
        selection_index: int,
        selection_probability: float,
        selection_label: str | None,
    ) -> "MeasurementCall":
        """Return an internally annotated call without revalidating fields."""
        call = object.__new__(type(self))
        object.__setattr__(call, "method", self.method)
        object.__setattr__(call, "operator", self.operator)
        object.__setattr__(call, "targets", self.targets)
        object.__setattr__(call, "collapse", self.collapse)
        object.__setattr__(call, "label", self.label)
        object.__setattr__(call, "discard", self.discard)
        object.__setattr__(call, "selection_index", selection_index)
        object.__setattr__(call, "selection_probability", selection_probability)
        object.__setattr__(call, "selection_label", selection_label)
        object.__setattr__(call, "meta", self.meta)
        return call


@dataclass(frozen=True, slots=True)
class MeasurementContext:
    """
    Context used to choose and execute a detector measurement.

    Parameters
    ----------
    device_id : str
        Detector component identifier.
    time : int
        Current simulation tick.
    signal : Signal or None
        Input signal when the detector operation came from a signal delivery.
    signal_targets : tuple[SubsystemId, ...]
        Qstate subsystem identifiers resolved from the signal or readout job.
    input_port_name : str or None, default=None
        Local ingress port name crossed by the input signal.
    detector_meta : tuple[tuple[str, object], ...], default=()
        Detector-level metadata available to metadata-selected measurements.
    named_targets : tuple[tuple[str, SubsystemId], ...], default=()
        Name-to-target bindings such as Bell analyzer ``"left"`` and
        ``"right"``.

    Notes
    -----
    Target resolution is explicit: targets are taken from ``signal_targets``,
    ``named_targets``, integer indexes, concrete ``SubsystemId`` objects, or a
    custom resolver. The helper does not discover targets by scanning the
    qstate manager.
    """

    device_id: str
    time: int
    signal: Signal | None
    signal_targets: tuple[SubsystemId, ...]
    input_port_name: str | None = None
    detector_meta: tuple[tuple[str, object], ...] = ()
    named_targets: tuple[tuple[str, SubsystemId], ...] = ()

    def __post_init__(self) -> None:
        _validate_measurement_context(self)


def _validate_measurement_method(method: object) -> None:
    if method not in _VALID_MEASUREMENT_METHODS:
        raise ValueError("unsupported measurement method")


def _validate_measurement_context(context: MeasurementContext) -> None:
    ensure_nonempty_id(context.device_id, field_name="device_id")
    _validate_time(context.time)
    _validate_signal(context.signal)
    _validate_signal_targets(context.signal_targets)
    _validate_input_port_name(context.input_port_name)
    _validate_detector_meta(context.detector_meta)
    _validate_named_targets(context.named_targets)


def _validate_time(time: object) -> None:
    if type(time) is not int:
        raise TypeError("time must be int")
    if time < 0:
        raise ValueError("time must be non-negative")


def _validate_signal(signal: object) -> None:
    if signal is not None and not isinstance(signal, Signal):
        raise TypeError("signal must be Signal or None")


def _validate_signal_targets(signal_targets: object) -> None:
    if not isinstance(signal_targets, tuple):
        raise TypeError("signal_targets must be tuple")

    for target in signal_targets:
        if not isinstance(target, SubsystemId):
            raise TypeError("signal_targets must contain SubsystemId")


def _validate_input_port_name(input_port_name: object) -> None:
    if input_port_name is not None and not isinstance(input_port_name, str):
        raise TypeError("input_port_name must be str or None")


def _validate_detector_meta(detector_meta: object) -> None:
    if not isinstance(detector_meta, tuple):
        raise TypeError("detector_meta must be tuple")

    for item in detector_meta:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("detector_meta entries must be 2-tuples")

        key, _value = item
        if not isinstance(key, str):
            raise TypeError("detector_meta keys must be str")


def _validate_named_targets(named_targets: object) -> None:
    if not isinstance(named_targets, tuple):
        raise TypeError("named_targets must be tuple")

    seen_names: set[str] = set()

    for item in named_targets:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("named_targets entries must be 2-tuples")

        name, target = item
        ensure_nonempty_id(name, field_name="named target name")

        if name in seen_names:
            raise ValueError("named target names must be unique")
        seen_names.add(name)

        if not isinstance(target, SubsystemId):
            raise TypeError("named target values must be SubsystemId")


def _named_target_map(context: MeasurementContext) -> dict[str, SubsystemId]:
    return {name: target for name, target in context.named_targets}


def _resolve_named_target(
    name: str,
    context: MeasurementContext,
) -> SubsystemId:
    named_targets = _named_target_map(context)

    if name not in named_targets:
        raise KeyError(f"unknown named measurement target: {name!r}")

    return named_targets[name]


def _resolve_target_index(
    index: int,
    context: MeasurementContext,
) -> SubsystemId:
    if index < 0:
        raise IndexError("measurement target index must be non-negative")

    try:
        return context.signal_targets[index]
    except IndexError:
        raise IndexError("measurement target index out of range") from None


def _resolve_one_target(
    target: object,
    context: MeasurementContext,
) -> SubsystemId:
    if isinstance(target, SubsystemId):
        return target

    if isinstance(target, str):
        if target == _SIGNAL_TARGET:
            raise ValueError("'signal' resolves to a target tuple, not one target")

        return _resolve_named_target(target, context)

    if type(target) is int:
        return _resolve_target_index(target, context)

    raise TypeError("unsupported measurement target spec")


def _check_unique_targets(
    resolved: tuple[SubsystemId, ...],
) -> tuple[SubsystemId, ...]:
    if not resolved:
        raise ValueError("measurement targets must be non-empty")

    if len(set(resolved)) != len(resolved):
        raise ValueError("measurement targets must be unique")

    return resolved


def _resolve_callable_targets(
    targets: Callable[[MeasurementContext], object],
    context: MeasurementContext,
) -> tuple[SubsystemId, ...]:
    resolved = targets(context)

    if isinstance(resolved, SubsystemId):
        return (resolved,)

    if not isinstance(resolved, tuple):
        raise TypeError("custom target resolver must return SubsystemId or tuple")

    for target in resolved:
        if not isinstance(target, SubsystemId):
            raise TypeError("custom target resolver tuple must contain SubsystemId")

    return _check_unique_targets(resolved)


def resolve_measurement_targets(
    targets: object,
    context: MeasurementContext,
) -> tuple[SubsystemId, ...]:
    """
    Resolve a detector measurement target specification.

    Parameters
    ----------
    targets : object
        ``"signal"``, a ``SubsystemId``, a named target string, an integer
        signal-target index, a tuple of those specs, or a callable returning a
        ``SubsystemId`` or tuple of ``SubsystemId`` values.
    context : MeasurementContext
        Measurement context containing signal and named target bindings.

    Returns
    -------
    tuple[SubsystemId, ...]
        Non-empty unique qstate targets.

    Notes
    -----
    ``"signal"`` resolves to the full ``context.signal_targets`` tuple and is
    therefore not valid where exactly one target is required internally. Target
    indexes require exact ``int`` values, so ``bool`` is rejected instead of
    being treated as ``0`` or ``1``.
    """

    if not isinstance(context, MeasurementContext):
        raise TypeError("context must be MeasurementContext")

    if callable(targets):
        return _resolve_callable_targets(targets, context)

    if targets == _SIGNAL_TARGET:
        return _check_unique_targets(context.signal_targets)

    if isinstance(targets, SubsystemId):
        return (targets,)

    if isinstance(targets, str):
        return (_resolve_one_target(targets, context),)

    if type(targets) is int:
        return (_resolve_one_target(targets, context),)

    if isinstance(targets, tuple):
        return _check_unique_targets(
            tuple(_resolve_one_target(target, context) for target in targets)
        )

    raise TypeError("unsupported measurement target spec")


@dataclass(frozen=True, slots=True)
class Measure:
    """
    Measurement-selection policy for detector components.

    Parameters
    ----------
    _chooser : callable
        Callable that returns a ``MeasurementCall`` for a
        ``MeasurementContext`` and optional RNG.
    label : str or None, default=None
        Policy label used in logs and reports.

    Notes
    -----
    ``Measure`` separates selection from execution. Components call
    ``choose(...)`` using a timeline-owned RNG stream, then pass the returned
    ``MeasurementCall`` to ``execute_measurement_call`` with the qstate manager.
    ``Measure.random`` consumes RNG only when choosing among configured
    measurements, and its probabilities must already sum to one; they are not
    normalized.
    """

    _chooser: Callable[[MeasurementContext, object | None], MeasurementCall]
    label: str | None = None

    def choose(
        self,
        context: MeasurementContext,
        *,
        rng: object | None = None,
    ) -> MeasurementCall:
        if not isinstance(context, MeasurementContext):
            raise TypeError("context must be MeasurementContext")

        call = self._chooser(context, rng)

        if not isinstance(call, MeasurementCall):
            raise TypeError("measurement chooser must return MeasurementCall")

        return call

    @staticmethod
    def _fixed(
        *,
        method: MeasurementMethod,
        operator: object | None = None,
        targets: object = _SIGNAL_TARGET,
        collapse: bool = True,
        label: str | None = None,
        discard: bool = False,
    ) -> "Measure":
        def choose(
            _context: MeasurementContext,
            _rng: object | None,
        ) -> MeasurementCall:
            return MeasurementCall(
                method=method,
                operator=operator,
                targets=targets,
                collapse=collapse,
                label=label,
                discard=discard,
            )

        return Measure(choose, label=label)

    @staticmethod
    def basis(
        basis: str | MeasurementBasis,
        *,
        targets: object = _SIGNAL_TARGET,
        collapse: bool = True,
        label: str | None = None,
    ) -> "Measure":
        resolved_basis = basis_for(basis)
        resolved_label = label if label is not None else resolved_basis.name

        return Measure._fixed(
            method="projective",
            operator=resolved_basis,
            targets=targets,
            collapse=collapse,
            label=resolved_label,
        )

    @staticmethod
    def povm(
        povm: POVM,
        *,
        targets: object = _SIGNAL_TARGET,
        collapse: bool = True,
        label: str | None = None,
    ) -> "Measure":
        if not isinstance(povm, POVM):
            raise TypeError("povm must be POVM")

        resolved_label = label if label is not None else povm.name

        return Measure._fixed(
            method="povm",
            operator=povm,
            targets=targets,
            collapse=collapse,
            label=resolved_label,
        )

    @staticmethod
    def bell(
        *,
        targets: object = _SIGNAL_TARGET,
        collapse: bool = True,
        label: str = "bell",
    ) -> "Measure":
        ensure_nonempty_id(label, field_name="label")

        return Measure._fixed(
            method="bell",
            targets=targets,
            collapse=collapse,
            label=label,
        )

    @staticmethod
    def none(
        *,
        targets: object = _SIGNAL_TARGET,
        discard: bool = False,
        label: str = "none",
    ) -> "Measure":
        ensure_nonempty_id(label, field_name="label")

        validate_bool(discard, field_name="discard")

        return Measure._fixed(
            method="none",
            targets=targets,
            collapse=False,
            label=label,
            discard=discard,
        )

    @staticmethod
    def random(
        choices: Mapping[object, float] | Sequence[tuple[object, float]],
        *,
        label: str = "random",
    ) -> "Measure":
        ensure_nonempty_id(label, field_name="label")
        normalized = _normalize_random_choices(choices)

        def choose(
            context: MeasurementContext,
            rng: object | None,
        ) -> MeasurementCall:
            selected_index, selected_measure, selected_probability = (
                _choose_random_measurement(normalized, rng)
            )
            call = selected_measure.choose(context, rng=rng)

            return call._with_selection(
                selection_index=selected_index,
                selection_probability=selected_probability,
                selection_label=selected_measure.label,
            )

        return Measure(choose, label=label)

    @staticmethod
    def by_meta(
        key: str,
        mapping: Mapping[object, object],
        *,
        default: object | None = None,
        label: str | None = None,
    ) -> "Measure":
        ensure_nonempty_id(key, field_name="metadata key")

        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be Mapping")

        normalized_mapping = {
            raw_key: Measure.from_spec(raw_measurement)
            for raw_key, raw_measurement in mapping.items()
        }

        default_measure = None if default is None else Measure.from_spec(default)
        resolved_label = label if label is not None else f"by_meta:{key}"

        def choose(
            context: MeasurementContext,
            rng: object | None,
        ) -> MeasurementCall:
            value = _lookup_measurement_meta(context, key)

            if value in normalized_mapping:
                return normalized_mapping[value].choose(context, rng=rng)

            if default_measure is not None:
                return default_measure.choose(context, rng=rng)

            raise KeyError(f"metadata key {key!r} had unmapped value {value!r}")

        return Measure(choose, label=resolved_label)

    @staticmethod
    def custom(
        chooser: Callable[[MeasurementContext, object | None], MeasurementCall],
        *,
        label: str | None = None,
    ) -> "Measure":
        if not callable(chooser):
            raise TypeError("chooser must be callable")

        return Measure(chooser, label=label)

    @staticmethod
    def from_spec(value: object) -> "Measure":
        if isinstance(value, Measure):
            return value

        if isinstance(value, (str, MeasurementBasis)):
            return Measure.basis(value)

        if isinstance(value, POVM):
            return Measure.povm(value)

        if callable(value):
            return Measure.custom(value)

        raise TypeError("cannot convert value to Measure")


def _choose_random_measurement(
    choices: tuple[tuple[Measure, float], ...],
    rng: object | None,
) -> tuple[int, Measure, float]:
    if rng is None:
        raise ValueError("rng is required for random measurement selection")

    if not hasattr(rng, "random"):
        raise TypeError("rng must provide random()")

    random_source = cast(_RandomSource, rng)
    draw = float(random_source.random())

    selected_index = len(choices) - 1
    selected_measure = choices[-1][0]
    selected_probability = choices[-1][1]

    cumulative = 0.0
    for index, (candidate, probability) in enumerate(choices):
        cumulative += probability
        if draw < cumulative:
            selected_index = index
            selected_measure = candidate
            selected_probability = probability
            break

    return selected_index, selected_measure, selected_probability


def _normalize_random_choices(
    choices: Mapping[object, float] | Sequence[tuple[object, float]],
) -> tuple[tuple[Measure, float], ...]:
    if isinstance(choices, Mapping):
        raw_items = tuple(choices.items())
    else:
        if isinstance(choices, (str, bytes)):
            raise TypeError("choices must not be str or bytes")
        raw_items = tuple(choices)

    if not raw_items:
        raise ValueError("choices must be non-empty")

    normalized: list[tuple[Measure, float]] = []
    total = 0.0

    for item in raw_items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("choices entries must be 2-tuples")

        measurement_spec, probability_raw = item

        if type(probability_raw) is bool or not isinstance(
            probability_raw,
            (int, float),
        ):
            raise TypeError("choice probability must be numeric")

        probability = float(probability_raw)

        if probability <= 0.0:
            raise ValueError("choice probability must be positive")

        normalized.append((Measure.from_spec(measurement_spec), probability))
        total += probability

    if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("choice probabilities must sum to 1")

    return tuple(normalized)


def _measurement_meta_items(
    context: MeasurementContext,
) -> Iterator[tuple[str, object]]:
    # Metadata-selected measurements use first-match precedence:
    # detector-level metadata, then signal metadata, then signal timing metadata.
    yield from context.detector_meta

    if context.signal is not None:
        yield from context.signal.meta
        yield from context.signal.timing_meta


def _lookup_measurement_meta(
    context: MeasurementContext,
    key: str,
) -> object:
    for meta_key, value in _measurement_meta_items(context):
        if meta_key == key:
            return value

    raise KeyError(f"metadata key not found: {key!r}")


def _validate_execution_inputs(
    *,
    call: MeasurementCall,
    context: MeasurementContext,
    qstate: QuantumStateManager,
) -> None:
    if not isinstance(call, MeasurementCall):
        raise TypeError("call must be MeasurementCall")

    if not isinstance(context, MeasurementContext):
        raise TypeError("context must be MeasurementContext")

    if not isinstance(qstate, QuantumStateManager):
        raise TypeError("qstate must be QuantumStateManager")


def execute_measurement_call(
    *,
    call: MeasurementCall,
    context: MeasurementContext,
    qstate: QuantumStateManager,
    rng: object | None,
    discard_after: bool = False,
) -> object | None:
    """
    Execute one measurement call against the qstate manager.

    Parameters
    ----------
    call : MeasurementCall
        Chosen measurement request.
    context : MeasurementContext
        Context used to resolve qstate targets.
    qstate : QuantumStateManager
        Qstate manager that owns the target subsystems.
    rng : object or None
        RNG passed through to qstate measurement methods.

    Returns
    -------
    object or None
        Qstate measurement result, or ``None`` for ``method="none"``.

    Notes
    -----
    Projective, POVM, and Bell methods delegate to the corresponding qstate
    manager methods with ``collapse=call.collapse``. When ``discard_after`` is
    true for a projective call, the manager measures and discards the measured
    targets in one workflow. ``method="none"`` performs no measurement and only
    discards targets when ``call.discard`` is true. This function is the
    detector primitive that may mutate qstate through collapse or discard;
    selection and readout helpers remain descriptive around it.
    """

    _validate_execution_inputs(call=call, context=context, qstate=qstate)
    validate_bool(discard_after, field_name="discard_after")
    targets = resolve_measurement_targets(call.targets, context)

    if call.method == "projective":
        if not isinstance(call.operator, (str, MeasurementBasis)):
            raise TypeError(
                "projective measurement operator must be str or MeasurementBasis"
            )

        if discard_after:
            return qstate.measure_and_discard(
                targets=targets,
                basis=call.operator,
                rng=rng,
                collapse=call.collapse,
            )

        return qstate.measure(
            targets=targets,
            basis=call.operator,
            rng=rng,
            collapse=call.collapse,
        )

    if call.method == "povm":
        if not isinstance(call.operator, POVM):
            raise TypeError("povm measurement operator must be POVM")

        return qstate.measure_povm(
            call.operator,
            targets=targets,
            rng=rng,
            collapse=call.collapse,
        )

    if call.method == "bell":
        return qstate.measure_bell(
            targets=targets,
            rng=rng,
            collapse=call.collapse,
        )

    if call.method == "none":
        if call.discard:
            qstate.discard(targets=targets)
        return None

    raise ValueError(f"unsupported measurement method: {call.method!r}")


__all__ = [
    "Measure",
    "MeasurementCall",
    "MeasurementContext",
    "MeasurementMethod",
    "TargetSpec",
    "execute_measurement_call",
    "resolve_measurement_targets",
]
