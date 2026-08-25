"""Readout mapping primitives for detector components.

This module maps qstate measurement results to detector-channel exposures and
builds report-only qubit readout outcomes. It keeps detector readout logic
separate from detector-channel physics and from qstate measurement execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, cast, runtime_checkable

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import require_optional_probability, validate_bool
from simyuj.qstate import QuantumStateManager, SubsystemId
from simyuj.signal import Signal

from .measurement import (
    Measure,
    MeasurementCall,
    MeasurementContext,
    execute_measurement_call,
)
from .reports import FLAG_NO_OUTCOME, DetectionReport
from .result_labels import result_label


@dataclass(frozen=True, slots=True)
class DetectorExposure:
    """
    Per-detector readout meaning for one measurement context.

    Parameters
    ----------
    detector_id : str
        Detector channel to evaluate.
    signal_present : bool, default=True
        Whether the measured signal should expose this detector to a signal
        click. Dark counts may still be evaluated when this is false.
    outcome_label : str or None, default=None
        Logical click label assigned to this detector.
    time_offset_ticks : int, default=0
        Non-negative offset from the component's base detection tick.
    meta : tuple[tuple[str, object], ...], default=()
        Metadata appended to raw clicks produced from this exposure.
    signal_click_probability : float or None, default=None
        Probability that the exposed signal produces a click, **replacing** the
        detector's own ``params.efficiency`` for this exposure. ``None`` uses
        that efficiency and is exactly today's behaviour.

    Notes
    -----
    ``outcome_label`` is kept even when ``signal_present`` is false. That lets
    dark counts from unexposed detectors report the detector's logical outcome.
    ``time_offset_ticks`` shifts the exposure start before gate clipping and
    detector-window evaluation.

    **``signal_click_probability`` overrides ``params.efficiency``; it does not
    multiply it.** The field exists for a payload whose detection probability is
    not a bare efficiency -- a coherent pulse, where
    ``coherent_optics.click_probability`` returns ``1 - exp(-eta_d * mu)`` with
    the detector's own ``eta_d`` already inside the exponent. Multiplying would
    apply ``eta_d`` twice and lower every click rate by that factor, silently:
    at ``eta_d = 0.2`` a run yields a fifth of the clicks and nothing raises,
    because the result looks exactly like a lossier link. The field name invites
    the wrong reading, which is why it is stated here and again on
    ``SinglePhotonDetector.evaluate_window``.

    It governs the **signal** click alone. Dark counts and afterpulses keep
    using ``params`` on every path, which is why this is an override of one
    term rather than a replacement of the detector's parameters.
    """

    detector_id: str
    signal_present: bool = True
    outcome_label: str | None = None
    time_offset_ticks: int = 0
    meta: tuple[tuple[str, object], ...] = ()
    # Appended last so no existing keyword position moves and every current
    # construction site keeps today's behaviour by taking the default.
    signal_click_probability: float | None = None

    def __post_init__(self) -> None:
        ensure_nonempty_id(self.detector_id, field_name="detector_id")

        validate_bool(self.signal_present, field_name="signal_present")

        if self.outcome_label is not None and not isinstance(
            self.outcome_label,
            str,
        ):
            raise TypeError("outcome_label must be str or None")

        if type(self.time_offset_ticks) is not int:
            raise TypeError("time_offset_ticks must be int")
        if self.time_offset_ticks < 0:
            raise ValueError("time_offset_ticks must be non-negative")

        object.__setattr__(
            self,
            "signal_click_probability",
            require_optional_probability(
                self.signal_click_probability,
                field_name="signal_click_probability",
            ),
        )

        _validate_meta(self.meta)


@dataclass(frozen=True, slots=True)
class ReadoutContext:
    """
    Inputs needed to map a qstate result to detector exposures.

    Parameters
    ----------
    detector_ids : tuple[str, ...]
        Detector identifiers in component order.
    measurement_call : MeasurementCall
        Measurement request that produced ``qstate_result``.
    qstate_result : object or None
        Qstate result whose label is mapped to detector exposures.
    signal : Signal
        Input signal being detected.
    """

    detector_ids: tuple[str, ...]
    measurement_call: MeasurementCall
    qstate_result: object | None
    signal: Signal

    def __post_init__(self) -> None:
        _validate_detector_ids(self.detector_ids)

        if not isinstance(self.measurement_call, MeasurementCall):
            raise TypeError("measurement_call must be MeasurementCall")

        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be Signal")


@runtime_checkable
class ReadoutLayout(Protocol):
    """
    Protocol for mapping measurement results to detector exposures.

    Notes
    -----
    Implementations return the exposures they control. Callers normalize the
    tuple so every detector appears once and omitted detectors become unexposed
    default entries.
    """

    def resolve_exposures(
        self,
        context: ReadoutContext,
    ) -> tuple[DetectorExposure, ...]: ...


@dataclass(frozen=True, slots=True)
class FixedReadout:
    """
    Readout layout that exposes one detector for the measured outcome.

    Parameters
    ----------
    detector_id : str or None, default=None
        Detector to expose. When omitted, the readout context must contain
        exactly one detector.
    """

    detector_id: str | None = None

    def __post_init__(self) -> None:
        if self.detector_id is not None:
            ensure_nonempty_id(self.detector_id, field_name="detector_id")

    def resolve_exposures(
        self,
        context: ReadoutContext,
    ) -> tuple[DetectorExposure, ...]:
        context = _require_readout_context(context)

        detector_id = self.detector_id

        if detector_id is None:
            if len(context.detector_ids) != 1:
                raise ValueError(
                    "detector_id is required when more than one detector exists"
                )
            detector_id = context.detector_ids[0]

        if detector_id not in context.detector_ids:
            raise ValueError(f"unknown readout detector_id: {detector_id!r}")

        label = result_label(context.qstate_result)

        return (
            DetectorExposure(
                detector_id=detector_id,
                outcome_label=None if label is None else str(label),
            ),
        )


@dataclass(frozen=True, slots=True)
class OutcomeMapReadout:
    """
    Readout layout mapping qstate result labels to detector ids.

    Parameters
    ----------
    mapping : Mapping[object, str]
        Mapping from logical qstate/result label to detector identifier.

    Notes
    -----
    The returned exposures include one entry for every mapped detector. The
    matching label has ``signal_present=True`` and other mapped detectors are
    returned with ``signal_present=False`` so dark-count behavior can still be
    evaluated. A dark count from an unexposed mapped detector can therefore
    still carry that detector's logical outcome label.
    """

    mapping: Mapping[object, str]
    _entries: tuple[tuple[object, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_entries", _outcome_entries(self.mapping))

    def resolve_exposures(
        self,
        context: ReadoutContext,
    ) -> tuple[DetectorExposure, ...]:
        context = _require_readout_context(context)

        label = result_label(context.qstate_result)

        if not _has_mapped_label(label, self._entries):
            _raise_unmapped_outcome(label)

        exposures: list[DetectorExposure] = []

        for mapped_label, detector_id in self._entries:
            if detector_id not in context.detector_ids:
                raise ValueError(f"mapped detector_id does not exist: {detector_id!r}")

            exposures.append(
                DetectorExposure(
                    detector_id=detector_id,
                    signal_present=label == mapped_label,
                    outcome_label=_outcome_label(mapped_label),
                )
            )

        return tuple(exposures)


@dataclass(frozen=True, slots=True)
class BasisOutcomeMapReadout:
    """
    Readout layout selected by measurement label, then result label.

    Parameters
    ----------
    mapping : Mapping[object, Mapping[object, str]]
        Outer mapping keyed by ``MeasurementCall.label``. Inner mappings are
        interpreted as ``OutcomeMapReadout`` mappings. The outer key is the
        selected measurement label, not the measurement method or basis object.
    """

    mapping: Mapping[object, Mapping[object, str]]
    _entries: tuple[tuple[object, OutcomeMapReadout], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, Mapping):
            raise TypeError("mapping must be Mapping")

        if not self.mapping:
            raise ValueError("mapping must be non-empty")

        entries: list[tuple[object, OutcomeMapReadout]] = []

        for label, outcome_map in self.mapping.items():
            if not isinstance(outcome_map, Mapping):
                raise TypeError("basis readout entries must be mappings")
            entries.append((label, OutcomeMapReadout(outcome_map)))

        object.__setattr__(self, "_entries", tuple(entries))

    def resolve_exposures(
        self,
        context: ReadoutContext,
    ) -> tuple[DetectorExposure, ...]:
        context = _require_readout_context(context)

        label = context.measurement_call.label

        for mapped_label, readout in self._entries:
            if label == mapped_label:
                return readout.resolve_exposures(context)

        raise ValueError(f"unmapped readout measurement label: {label!r}")


@dataclass(frozen=True, slots=True)
class _CallableReadout:
    resolver: Callable[[ReadoutContext], tuple[DetectorExposure, ...]]

    def resolve_exposures(
        self,
        context: ReadoutContext,
    ) -> tuple[DetectorExposure, ...]:
        exposures = self.resolver(context)

        if not isinstance(exposures, tuple):
            raise TypeError("custom readout must return tuple")

        for exposure in exposures:
            if not isinstance(exposure, DetectorExposure):
                raise TypeError("custom readout tuple must contain DetectorExposure")

        return exposures


def readout_from_spec(
    value: object,
    detector_ids: tuple[str, ...],
) -> ReadoutLayout:
    """
    Convert a public readout specification into a readout layout.

    Parameters
    ----------
    value : object
        ``None``, mapping, existing ``ReadoutLayout``, or callable accepting a
        ``ReadoutContext``.
    detector_ids : tuple[str, ...]
        Detector ids available to the owning component.

    Returns
    -------
    ReadoutLayout
        Readout layout used by detector arrays.

    Notes
    -----
    ``None`` means fixed single-detector readout, a flat mapping means
    outcome-label to detector-id readout, and a nested mapping means
    measurement-label to outcome-label to detector-id readout.
    """

    checked_detector_ids = _validate_detector_ids(detector_ids)

    if value is None:
        if len(checked_detector_ids) != 1:
            raise ValueError("readout is required when more than one detector exists")
        return FixedReadout()

    if isinstance(value, Mapping):
        readout = _readout_from_mapping(value)
        _validate_mapped_detector_ids(readout, checked_detector_ids)
        return readout

    if isinstance(value, ReadoutLayout):
        return value

    if callable(value):
        resolver = cast(Callable[[ReadoutContext], tuple[DetectorExposure, ...]], value)
        return _CallableReadout(resolver)

    raise TypeError("cannot convert value to ReadoutLayout")


def normalize_readout_exposures(
    exposures: object,
    *,
    detector_ids: tuple[str, ...],
) -> tuple[DetectorExposure, ...]:
    """
    Return exactly one exposure per detector in detector order.

    Parameters
    ----------
    exposures : object
        Candidate tuple returned by a readout layout.
    detector_ids : tuple[str, ...]
        Component detector order.

    Returns
    -------
    tuple[DetectorExposure, ...]
        Exposure tuple ordered like ``detector_ids``. Missing detectors are
        filled as unexposed detectors, not ignored, so later detector-window
        evaluation may still sample their dark counts.
    """

    checked_detector_ids = _validate_detector_ids(detector_ids)

    if not isinstance(exposures, tuple):
        raise TypeError("readout must return tuple")

    exposure_by_detector_id: dict[str, DetectorExposure] = {}

    for exposure in exposures:
        if not isinstance(exposure, DetectorExposure):
            raise TypeError("readout exposures must contain DetectorExposure")

        if exposure.detector_id not in checked_detector_ids:
            raise ValueError(f"unknown readout detector_id: {exposure.detector_id!r}")

        if exposure.detector_id in exposure_by_detector_id:
            raise ValueError("readout exposures must not duplicate detector_id")

        exposure_by_detector_id[exposure.detector_id] = exposure

    return tuple(
        exposure_by_detector_id.get(
            detector_id,
            DetectorExposure(detector_id=detector_id, signal_present=False),
        )
        for detector_id in checked_detector_ids
    )


def run_qubit_readout(
    *,
    device_id: str,
    time: int,
    targets: tuple[SubsystemId, ...],
    measurement: object,
    collapse: bool | None,
    qstate: QuantumStateManager,
    measurement_choice_rng: object | None,
    qstate_rng: object | None,
    readout_rng: object | None,
    readout_model: object | None,
    signal: Signal | None = None,
    input_port_name: str | None = None,
    detector_meta: tuple[tuple[str, object], ...] = (),
    report_id: str | None = None,
) -> DetectionReport:
    """
    Run qstate measurement, readout distortion, and report construction.

    Parameters
    ----------
    device_id : str
        Detector device identifier.
    time : int
        Simulation tick reported on the resulting detection report.
    targets : tuple[SubsystemId, ...]
        Explicit qstate targets to measure or discard.
    measurement : object
        Measurement spec converted through ``Measure.from_spec``.
    collapse : bool or None
        Optional override for measurement collapse behavior.
    qstate : QuantumStateManager
        Qstate manager that owns the targets.
    measurement_choice_rng, qstate_rng, readout_rng : object or None
        RNG streams for random measurement selection, qstate measurement, and
        readout distortion.
    readout_model : object or None
        Object exposing ``report_outcome`` or ``None`` for identity readout.
    signal : Signal or None, default=None
        Optional source signal associated with the readout.
    input_port_name : str or None, default=None
        Optional component input port name associated with the readout.
    detector_meta : tuple[tuple[str, object], ...], default=()
        Metadata appended to the report.
    report_id : str or None, default=None
        Optional explicit report identifier.

    Returns
    -------
    DetectionReport
        Immutable report whose ``qstate_result`` records the true qstate
        measurement result and whose ``outcome`` records the readout-model
        output. These may differ when ``readout_model`` applies classical
        readout noise or confusion.

    Notes
    -----
    This primitive owns no ports, events, timelines, or device report storage.
    Callers pass explicit targets, time, RNGs, and qstate ownership. Qstate
    mutation is limited to the selected measurement or discard operation. With
    ``readout_model=None``, report metadata records the model type as
    ``"NoneType"`` because it stores the Python type name.
    """
    _validate_readout_inputs(
        device_id=device_id,
        time=time,
        targets=targets,
        qstate=qstate,
        signal=signal,
        input_port_name=input_port_name,
        detector_meta=detector_meta,
        report_id=report_id,
    )

    context = MeasurementContext(
        device_id=device_id,
        time=time,
        signal=signal,
        signal_targets=targets,
        input_port_name=input_port_name,
        detector_meta=detector_meta,
    )
    call = _choose_measurement_call(
        measurement=measurement,
        context=context,
        collapse=collapse,
        rng=measurement_choice_rng,
    )
    qstate_result = execute_measurement_call(
        call=call,
        context=context,
        qstate=qstate,
        rng=qstate_rng,
    )
    true_label = result_label(qstate_result)
    reported_label = _report_outcome(
        readout_model=readout_model,
        true_label=true_label,
        qstate_result=qstate_result,
        measurement_call=call,
        context=context,
        rng=readout_rng,
    )

    return _build_detection_report(
        report_id=report_id,
        device_id=device_id,
        time=time,
        targets=targets,
        signal=signal,
        call=call,
        qstate_result=qstate_result,
        true_label=true_label,
        reported_label=reported_label,
        readout_model=readout_model,
        input_port_name=input_port_name,
        detector_meta=detector_meta,
    )


def _readout_from_mapping(
    mapping: Mapping[object, object],
) -> OutcomeMapReadout | BasisOutcomeMapReadout:
    values = tuple(mapping.values())

    if all(isinstance(value, str) for value in values):
        str_mapping = cast(Mapping[object, str], mapping)
        return OutcomeMapReadout(str_mapping)

    if all(isinstance(value, Mapping) for value in values):
        basis_mapping = cast(Mapping[object, Mapping[object, str]], mapping)
        return BasisOutcomeMapReadout(basis_mapping)

    raise TypeError("readout mapping values must be all str or all Mapping")


def _choose_measurement_call(
    *,
    measurement: object,
    context: MeasurementContext,
    collapse: bool | None,
    rng: object | None,
) -> MeasurementCall:
    measurement_spec = Measure.from_spec(measurement)
    call = measurement_spec.choose(context, rng=rng)

    if collapse is not None and call.method != "none":
        return replace(call, collapse=collapse)

    return call


def _report_outcome(
    *,
    readout_model: object | None,
    true_label: object | None,
    qstate_result: object | None,
    measurement_call: MeasurementCall,
    context: MeasurementContext,
    rng: object | None,
) -> object | None:
    if readout_model is None:
        return true_label

    report_outcome = getattr(readout_model, "report_outcome", None)
    if not callable(report_outcome):
        raise TypeError("readout_model must expose report_outcome")

    return report_outcome(
        true_outcome=true_label,
        qstate_result=qstate_result,
        measurement_call=measurement_call,
        context=context,
        rng=rng,
    )


def _build_detection_report(
    *,
    report_id: str | None,
    device_id: str,
    time: int,
    targets: tuple[SubsystemId, ...],
    signal: Signal | None,
    call: MeasurementCall,
    qstate_result: object | None,
    true_label: object | None,
    reported_label: object | None,
    readout_model: object | None,
    input_port_name: str | None,
    detector_meta: tuple[tuple[str, object], ...],
) -> DetectionReport:
    success = reported_label is not None

    return DetectionReport(
        report_id=(f"{device_id}:readout:{time}" if report_id is None else report_id),
        device_id=device_id,
        time=time,
        success=success,
        outcome=reported_label,
        raw_clicks=(),
        qstate_result=qstate_result,
        measurement_method=call.method,
        measurement_label=call.label,
        selection_index=call.selection_index,
        selection_probability=call.selection_probability,
        selection_label=call.selection_label,
        signal_id=None if signal is None else signal.id,
        flags=() if success else (FLAG_NO_OUTCOME,),
        meta=(
            ("device_id", device_id),
            ("targets", tuple(str(target) for target in targets)),
            ("measurement", call.method),
            ("measurement_label", call.label),
            ("collapse", call.collapse),
            ("true_label", true_label),
            ("reported_label", reported_label),
            ("qstate_probability", getattr(qstate_result, "probability", None)),
            ("qstate_probabilities", getattr(qstate_result, "probabilities", None)),
            ("qstate_state_ref", getattr(qstate_result, "state_ref", None)),
            (
                "qstate_post_state_ref",
                getattr(qstate_result, "post_state_ref", None),
            ),
            ("input_signal_id", None if signal is None else signal.id),
            ("input_port_name", input_port_name),
            ("readout_model", type(readout_model).__name__),
            *detector_meta,
        ),
    )


def _validate_readout_inputs(
    *,
    device_id: str,
    time: int,
    targets: tuple[SubsystemId, ...],
    qstate: QuantumStateManager,
    signal: Signal | None,
    input_port_name: str | None,
    detector_meta: tuple[tuple[str, object], ...],
    report_id: str | None,
) -> None:
    ensure_nonempty_id(device_id, field_name="device_id")
    if type(time) is not int:
        raise TypeError("time must be int")
    if time < 0:
        raise ValueError("time must be non-negative")
    if not isinstance(targets, tuple):
        raise TypeError("targets must be tuple[SubsystemId, ...]")
    if not targets:
        raise ValueError("targets must be non-empty")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    for target in targets:
        if not isinstance(target, SubsystemId):
            raise TypeError("targets must contain SubsystemId")
    if not isinstance(qstate, QuantumStateManager):
        raise TypeError("qstate must be QuantumStateManager")
    if signal is not None and not isinstance(signal, Signal):
        raise TypeError("signal must be Signal or None")
    if input_port_name is not None:
        ensure_nonempty_id(input_port_name, field_name="input_port_name")
    _validate_meta(detector_meta)
    if report_id is not None:
        ensure_nonempty_id(report_id, field_name="report_id")


def _outcome_label(label: object) -> str | None:
    if label is None:
        return None
    return str(label)


def _outcome_entries(mapping: Mapping[object, str]) -> tuple[tuple[object, str], ...]:
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be Mapping")

    if not mapping:
        raise ValueError("mapping must be non-empty")

    entries: list[tuple[object, str]] = []

    for label, detector_id in mapping.items():
        if not isinstance(detector_id, str):
            raise TypeError("mapped detector_id must be str")
        ensure_nonempty_id(detector_id, field_name="mapped detector_id")
        entries.append((label, detector_id))

    _check_unique_detector_ids(entries)

    return tuple(entries)


def _require_readout_context(context: object) -> ReadoutContext:
    if not isinstance(context, ReadoutContext):
        raise TypeError("context must be ReadoutContext")
    return context


def _has_mapped_label(
    label: object | None,
    entries: tuple[tuple[object, str], ...],
) -> bool:
    return any(label == mapped_label for mapped_label, _detector_id in entries)


def _raise_unmapped_outcome(label: object | None) -> None:
    if label is None:
        raise ValueError("qstate result is None and no None readout mapping exists")

    raise ValueError(f"unmapped readout outcome label: {label!r}")


def _validate_detector_ids(detector_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(detector_ids, tuple):
        raise TypeError("detector_ids must be tuple")

    if not detector_ids:
        raise ValueError("detector_ids must be non-empty")

    seen: set[str] = set()

    for detector_id in detector_ids:
        ensure_nonempty_id(detector_id, field_name="detector_id")

        if detector_id in seen:
            raise ValueError("detector_ids must be unique")

        seen.add(detector_id)

    return detector_ids


def _validate_meta(meta: object) -> None:
    if not isinstance(meta, tuple):
        raise TypeError("meta must be tuple")

    for item in meta:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("meta entries must be 2-tuples")

        key, _value = item
        if not isinstance(key, str):
            raise TypeError("meta keys must be str")


def _validate_mapped_detector_ids(
    readout: OutcomeMapReadout | BasisOutcomeMapReadout,
    detector_ids: tuple[str, ...],
) -> None:
    if isinstance(readout, OutcomeMapReadout):
        entries = readout._entries
    else:
        entries = tuple(
            entry
            for _basis_label, outcome_readout in readout._entries
            for entry in outcome_readout._entries
        )

    for _label, detector_id in entries:
        if detector_id not in detector_ids:
            raise ValueError(f"mapped detector_id does not exist: {detector_id!r}")


def _check_unique_detector_ids(entries: list[tuple[object, str]]) -> None:
    seen: set[str] = set()

    for _label, detector_id in entries:
        if detector_id in seen:
            raise ValueError("mapped detector_id values must be unique")

        seen.add(detector_id)


__all__ = [
    "BasisOutcomeMapReadout",
    "DetectorExposure",
    "FixedReadout",
    "MeasurementContext",
    "OutcomeMapReadout",
    "ReadoutContext",
    "ReadoutLayout",
    "normalize_readout_exposures",
    "readout_from_spec",
    "run_qubit_readout",
]
