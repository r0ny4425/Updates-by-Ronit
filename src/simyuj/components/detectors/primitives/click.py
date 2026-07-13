"""Click-resolution primitives for detector reports.

This module converts low-level ``RawClick`` records into ``DetectionReport``
objects. It does not evaluate detector physics, mutate qstate, or schedule
timeline events; detector components call these resolvers after measurement,
readout, and window evaluation are complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import validate_bool
from simyuj.signal import Signal

from .measurement import MeasurementCall
from .reports import FLAG_DOUBLE_CLICK, FLAG_NO_CLICK, DetectionReport, RawClick
from .result_labels import result_label

DoubleClickPolicy = Literal["fail", "first", "random"]


class ClickPatternResolver(Protocol):
    """
    Protocol for converting raw detector clicks into a report.

    Notes
    -----
    Resolvers receive already-evaluated ``RawClick`` records and optional
    qstate measurement metadata. They do not measure qstate and do not schedule
    events. A resolver may consume ``rng`` for policies such as random
    double-click selection.
    """

    def resolve(
        self,
        *,
        device_id: str,
        time: int,
        signal: Signal | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall | None,
        raw_clicks: tuple[RawClick, ...],
        rng: object | None = None,
    ) -> DetectionReport: ...


@dataclass(frozen=True, slots=True)
class ClickPattern:
    """
    Detector-click pattern mapped to a logical outcome.

    Parameters
    ----------
    outcome : object
        Logical outcome assigned when exactly these detectors click.
    detector_ids : tuple[str, ...]
        Detector identifiers that define the pattern. Pattern matching sorts
        identifiers, so pair order is not significant.

    Notes
    -----
    Pattern records are passive data and do not validate ambiguity or duplicate
    detector ids. Components that accept patterns own that configuration
    validation.
    """

    outcome: object
    detector_ids: tuple[str, ...]


def resolve_click_pattern(
    *,
    raw_clicks: tuple[RawClick, ...],
    patterns: tuple[ClickPattern, ...],
) -> object | None:
    """
    Resolve raw clicks against configured detector patterns.

    Parameters
    ----------
    raw_clicks : tuple[RawClick, ...]
        Clicks observed in one detection window.
    patterns : tuple[ClickPattern, ...]
        Candidate mappings from exact sorted detector-id tuples to logical
        outcomes.

    Returns
    -------
    object or None
        Matching pattern outcome, or ``None`` when the clicked detector tuple
        has no configured mapping.

    Notes
    -----
    Matching is exact after sorting detector identifiers. Duplicate raw clicks
    on the same detector remain duplicate identifiers; they are not collapsed
    into a set.
    """

    clicked = tuple(sorted(click.detector_id for click in raw_clicks))

    for pattern in patterns:
        if clicked == tuple(sorted(pattern.detector_ids)):
            return pattern.outcome

    return None


def _validate_resolve_inputs(
    *,
    device_id: str,
    time: int,
    signal: Signal | None,
    measurement_call: MeasurementCall | None,
    raw_clicks: tuple[RawClick, ...],
) -> None:
    ensure_nonempty_id(device_id, field_name="device_id")

    if type(time) is not int:
        raise TypeError("time must be int")
    if time < 0:
        raise ValueError("time must be non-negative")

    if signal is not None and not isinstance(signal, Signal):
        raise TypeError("signal must be Signal or None")

    if measurement_call is not None and not isinstance(
        measurement_call,
        MeasurementCall,
    ):
        raise TypeError("measurement_call must be MeasurementCall or None")

    if not isinstance(raw_clicks, tuple):
        raise TypeError("raw_clicks must be tuple")

    for click in raw_clicks:
        if not isinstance(click, RawClick):
            raise TypeError("raw_clicks must contain RawClick")


def _signal_id(signal: Signal | None) -> object | None:
    if signal is None:
        return None
    return getattr(signal, "signal_id", signal.id)


def _report_id(
    *,
    device_id: str,
    time: int,
    signal: Signal | None,
    suffix: str,
) -> str:
    signal_part = _signal_id(signal)
    if signal_part is None:
        signal_part = "no-signal"
    return f"{device_id}:report:{time}:{signal_part}:{suffix}"


def _sort_clicks(raw_clicks: tuple[RawClick, ...]) -> tuple[RawClick, ...]:
    return tuple(
        sorted(
            raw_clicks,
            key=lambda click: (click.time, click.detector_id, click.trigger),
        )
    )


@dataclass(frozen=True, slots=True)
class ThresholdClickResolver:
    """
    Resolve threshold-detector clicks using ``RawClick.outcome_label``.

    Parameters
    ----------
    double_click_policy : {"fail", "first", "random"}, default="fail"
        Policy for multiple raw clicks in the same report. ``"fail"`` produces
        an unsuccessful double-click report, ``"first"`` chooses the earliest
        sorted click, and ``"random"`` chooses one click using the supplied RNG.

    Notes
    -----
    This resolver assumes the readout layer has already assigned each
    detector's logical click label to ``RawClick.outcome_label``. Raw clicks are
    sorted by click time, detector id, and trigger before resolution. The
    ``"first"`` policy chooses the first click after that sort, not the first
    click in caller-provided order. The ``"random"`` policy chooses uniformly
    from the sorted click tuple and consumes the supplied resolver RNG only for
    double-click resolution.
    """

    double_click_policy: DoubleClickPolicy = "fail"

    def __post_init__(self) -> None:
        if self.double_click_policy not in {"fail", "first", "random"}:
            raise ValueError("unsupported double-click policy")

    def resolve(
        self,
        *,
        device_id: str,
        time: int,
        signal: Signal | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall | None,
        raw_clicks: tuple[RawClick, ...],
        rng: object | None = None,
    ) -> DetectionReport:
        _validate_resolve_inputs(
            device_id=device_id,
            time=time,
            signal=signal,
            measurement_call=measurement_call,
            raw_clicks=raw_clicks,
        )

        sorted_clicks = _sort_clicks(raw_clicks)

        if not sorted_clicks:
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=False,
                outcome=None,
                raw_clicks=(),
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix="no-click",
                flags=(FLAG_NO_CLICK,),
            )

        if len(sorted_clicks) == 1:
            click = sorted_clicks[0]
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=True,
                outcome=click.outcome_label,
                raw_clicks=sorted_clicks,
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix=f"click:{click.detector_id}",
                flags=click.flags,
            )

        return self._resolve_double_click(
            device_id=device_id,
            time=time,
            signal=signal,
            qstate_result=qstate_result,
            measurement_call=measurement_call,
            raw_clicks=sorted_clicks,
            rng=rng,
        )

    def _resolve_double_click(
        self,
        *,
        device_id: str,
        time: int,
        signal: Signal | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall | None,
        raw_clicks: tuple[RawClick, ...],
        rng: object | None,
    ) -> DetectionReport:
        if self.double_click_policy == "fail":
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=False,
                outcome=None,
                raw_clicks=raw_clicks,
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix="double-click",
                flags=(FLAG_DOUBLE_CLICK,),
            )

        if self.double_click_policy == "first":
            selected = raw_clicks[0]
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=True,
                outcome=selected.outcome_label,
                raw_clicks=raw_clicks,
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix=f"double-click:first:{selected.detector_id}",
                flags=(FLAG_DOUBLE_CLICK, *selected.flags),
                meta=(("selected_detector_id", selected.detector_id),),
            )

        if self.double_click_policy == "random":
            selected = _choose_random_click(raw_clicks, rng)
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=True,
                outcome=selected.outcome_label,
                raw_clicks=raw_clicks,
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix=f"double-click:random:{selected.detector_id}",
                flags=(FLAG_DOUBLE_CLICK, *selected.flags),
                meta=(("selected_detector_id", selected.detector_id),),
            )

        raise ValueError("unsupported double-click policy")


def _choose_random_click(
    raw_clicks: tuple[RawClick, ...],
    rng: object | None,
) -> RawClick:
    if rng is None:
        raise ValueError("rng is required for random double-click policy")

    if hasattr(rng, "integers"):
        index = int(rng.integers(0, len(raw_clicks)))
        return raw_clicks[index]

    if hasattr(rng, "randrange"):
        index = int(rng.randrange(len(raw_clicks)))
        return raw_clicks[index]

    if hasattr(rng, "random"):
        draw = float(rng.random())
        index = min(int(draw * len(raw_clicks)), len(raw_clicks) - 1)
        return raw_clicks[index]

    raise TypeError("rng must provide integers(), randrange(), or random()")


def _make_report(
    *,
    device_id: str,
    time: int,
    signal: Signal | None,
    success: bool,
    outcome: object | None,
    raw_clicks: tuple[RawClick, ...],
    qstate_result: object | None,
    measurement_call: MeasurementCall | None,
    suffix: str,
    flags: tuple[str, ...] = (),
    meta: tuple[tuple[str, object], ...] = (),
) -> DetectionReport:
    measurement_method = None
    measurement_label = None
    selection_index = None
    selection_probability = None
    selection_label = None

    if measurement_call is not None:
        measurement_method = measurement_call.method
        measurement_label = measurement_call.label
        selection_index = measurement_call.selection_index
        selection_probability = measurement_call.selection_probability
        selection_label = measurement_call.selection_label

    return DetectionReport(
        report_id=_report_id(
            device_id=device_id,
            time=time,
            signal=signal,
            suffix=suffix,
        ),
        device_id=device_id,
        time=time,
        success=success,
        outcome=outcome,
        raw_clicks=raw_clicks,
        qstate_result=qstate_result,
        measurement_method=measurement_method,
        measurement_label=measurement_label,
        selection_index=selection_index,
        selection_probability=selection_probability,
        selection_label=selection_label,
        signal_id=_signal_id(signal),
        flags=flags,
        meta=meta,
    )


@dataclass(frozen=True, slots=True)
class POVMLabelClickResolver:
    """
    Resolve reports from the qstate measurement result label.

    Parameters
    ----------
    require_click : bool, default=True
        Whether a missing raw click makes the report fail with ``no_click``.

    Notes
    -----
    Use this when the POVM or qstate result itself is the receiver outcome. It
    is intentionally separate from physical click-label resolution, so detector
    click labels do not overwrite the qstate result label. When
    ``require_click=False``, a missing qstate result label still produces an
    unsuccessful report because the resolver has no logical outcome to report.
    """

    require_click: bool = True

    def __post_init__(self) -> None:
        validate_bool(self.require_click, field_name="require_click")

    def resolve(
        self,
        *,
        device_id: str,
        time: int,
        signal: Signal | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall | None,
        raw_clicks: tuple[RawClick, ...],
        rng: object | None = None,
    ) -> DetectionReport:
        _validate_resolve_inputs(
            device_id=device_id,
            time=time,
            signal=signal,
            measurement_call=measurement_call,
            raw_clicks=raw_clicks,
        )

        sorted_clicks = _sort_clicks(raw_clicks)

        if self.require_click and not sorted_clicks:
            return _make_report(
                device_id=device_id,
                time=time,
                signal=signal,
                success=False,
                outcome=None,
                raw_clicks=(),
                qstate_result=qstate_result,
                measurement_call=measurement_call,
                suffix="povm:no-click",
                flags=(FLAG_NO_CLICK,),
            )

        outcome = result_label(qstate_result)

        return _make_report(
            device_id=device_id,
            time=time,
            signal=signal,
            success=outcome is not None,
            outcome=outcome,
            raw_clicks=sorted_clicks,
            qstate_result=qstate_result,
            measurement_call=measurement_call,
            suffix=f"povm:{outcome}",
            flags=(),
        )


__all__ = [
    "ClickPattern",
    "ClickPatternResolver",
    "DoubleClickPolicy",
    "POVMLabelClickResolver",
    "ThresholdClickResolver",
    "resolve_click_pattern",
]
