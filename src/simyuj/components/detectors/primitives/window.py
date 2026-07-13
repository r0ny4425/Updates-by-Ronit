"""Shared detector-window evaluation helpers.

This module binds per-channel RNG streams, normalizes detector ordering, clips
exposure windows against gate models, and asks ``SinglePhotonDetector`` objects
to evaluate active windows. It is used by detector components but does not
schedule events or touch qstate.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..single_photon import SinglePhotonDetector
from .readout import DetectorExposure
from .reports import RawClick
from .rng import DetectorRNGStreams


def normalize_detectors(
    detectors: Sequence[SinglePhotonDetector],
    *,
    require_non_empty: bool,
) -> tuple[SinglePhotonDetector, ...]:
    """
    Validate and freeze a detector sequence.

    Parameters
    ----------
    detectors : Sequence[SinglePhotonDetector]
        Candidate detector channels.
    require_non_empty : bool
        Whether an empty sequence is rejected.

    Returns
    -------
    tuple[SinglePhotonDetector, ...]
        Detectors in caller-provided order.

    Notes
    -----
    Detector order is part of the readout contract. Window evaluation expects
    readout exposures to be normalized into the same order.
    """

    if isinstance(detectors, (str, bytes)) or not isinstance(detectors, Sequence):
        raise TypeError("detectors must be a sequence of SinglePhotonDetector")

    normalized = tuple(detectors)

    if require_non_empty and not normalized:
        raise ValueError("detectors must be non-empty")

    seen: set[str] = set()
    for detector in normalized:
        if not isinstance(detector, SinglePhotonDetector):
            raise TypeError("detectors must contain SinglePhotonDetector")

        if detector.detector_id in seen:
            raise ValueError("detector_id values must be unique")

        seen.add(detector.detector_id)

    return normalized


def validate_gate_model(gate_model: object) -> None:
    for method_name in (
        "is_open",
        "window_containing",
        "active_duration_between",
    ):
        method = getattr(gate_model, method_name, None)
        if not callable(method):
            raise TypeError(
                "gate_model must provide "
                "is_open(...), window_containing(...), and "
                "active_duration_between(...)"
            )


def bind_detector_rngs(
    *,
    timeline,
    device_id: str,
    namespace: str,
    detectors: tuple[SinglePhotonDetector, ...],
) -> dict[str, DetectorRNGStreams]:
    """
    Bind deterministic RNG streams for every detector channel.

    Parameters
    ----------
    timeline : Timeline
        Timeline that owns deterministic RNG streams.
    device_id : str
        Detector component identifier.
    namespace : str
        Component-local stream namespace, such as ``"detector_array"``.
    detectors : tuple[SinglePhotonDetector, ...]
        Detector channels to bind.

    Returns
    -------
    dict[str, DetectorRNGStreams]
        Streams keyed by detector identifier.

    Notes
    -----
    Component ``bind`` methods call this before event execution so detection is
    reproducible under fixed timeline seeds and fixed configuration. Streams
    are bound eagerly for every detector even when parameter edge cases may not
    consume every stream during a run.
    """

    return {
        detector.detector_id: DetectorRNGStreams(
            efficiency=timeline.rng(
                device_id,
                namespace,
                detector.detector_id,
                "efficiency",
            ),
            dark=timeline.rng(
                device_id,
                namespace,
                detector.detector_id,
                "dark",
            ),
            jitter=timeline.rng(
                device_id,
                namespace,
                detector.detector_id,
                "jitter",
            ),
            afterpulse=timeline.rng(
                device_id,
                namespace,
                detector.detector_id,
                "afterpulse",
            ),
        )
        for detector in detectors
    }


def require_detector_rngs(
    detector_rngs: dict[str, DetectorRNGStreams],
    detector: SinglePhotonDetector,
) -> DetectorRNGStreams:
    try:
        return detector_rngs[detector.detector_id]
    except KeyError:
        raise RuntimeError(
            f"RNG streams are not bound for detector {detector.detector_id!r}"
        ) from None


def active_detection_duration_at_arrival(
    *,
    time: int,
    detection_window_ticks: int,
    gate_model,
) -> int:
    """
    Return the active portion of a detection window at an arrival tick.

    Parameters
    ----------
    time : int
        Arrival tick for the detector exposure.
    detection_window_ticks : int
        Requested detector observation length in simulation ticks.
    gate_model : GateModel
        Gate model queried for active intervals.

    Returns
    -------
    int
        Number of active ticks available from ``time`` before the configured
        detection window or enclosing gate closes.

    Notes
    -----
    The requested detection window is clipped to active gate time from the
    exposure start. An arrival outside a gate returns ``0``.
    """

    requested_end = time + detection_window_ticks

    active_duration = gate_model.active_duration_between(
        time,
        requested_end,
    )

    if active_duration >= detection_window_ticks:
        return detection_window_ticks

    gate_window = gate_model.window_containing(time)

    if gate_window is None:
        return 0

    return max(
        0,
        min(
            detection_window_ticks,
            gate_window.end - time,
        ),
    )


def evaluate_detector_windows(
    *,
    device_id: str,
    time: int,
    detectors: tuple[SinglePhotonDetector, ...],
    exposures: tuple[DetectorExposure, ...],
    detector_rngs: dict[str, DetectorRNGStreams],
    detection_window_ticks: int,
    gate_model,
    measurement_label: str | None,
    fallback_complete_time: int,
    click_meta: tuple[tuple[str, object], ...] = (),
) -> tuple[tuple[RawClick, ...], int]:
    """
    Evaluate one normalized exposure window per detector channel.

    Parameters
    ----------
    device_id : str
        Owning detector component identifier used in click metadata.
    time : int
        Base detection tick.
    detectors : tuple[SinglePhotonDetector, ...]
        Detector channels in readout order.
    exposures : tuple[DetectorExposure, ...]
        Normalized exposures in the same order as ``detectors``.
    detector_rngs : dict[str, DetectorRNGStreams]
        Bound deterministic RNG streams keyed by detector identifier.
    detection_window_ticks : int
        Requested detector window length in simulation ticks.
    gate_model : GateModel
        Gate model that may shorten or close each exposure window.
    measurement_label : str or None
        Measurement label added to raw-click metadata.
    fallback_complete_time : int
        Completion tick used when no active exposure extends the time.
    click_meta : tuple[tuple[str, object], ...], default=()
        Extra metadata appended to every raw click.

    Returns
    -------
    tuple[tuple[RawClick, ...], int]
        Raw clicks and the latest tick at which evaluated active windows
        complete.

    Notes
    -----
    This helper owns no timeline or qstate behavior. It only evaluates detector
    physics for already-resolved exposures and preserves exposure order.
    Exposures must already be normalized into detector order. Unexposed
    detectors are still evaluated when their gate window is active, because
    dark counts may occur without a signal. The returned completion time is the
    latest evaluated active-window end, falling back to ``fallback_complete_time``
    when no active exposure extends it.
    """

    if len(exposures) != len(detectors):
        raise RuntimeError("exposure length must match number of detectors")

    raw_clicks: list[RawClick] = []
    detection_complete_time = fallback_complete_time

    for index, (detector, exposure) in enumerate(zip(detectors, exposures)):
        if exposure.detector_id != detector.detector_id:
            raise RuntimeError("exposure detector_id must match detector order")

        exposure_time = time + exposure.time_offset_ticks
        exposure_window_duration_ticks = active_detection_duration_at_arrival(
            time=exposure_time,
            detection_window_ticks=detection_window_ticks,
            gate_model=gate_model,
        )

        if exposure_window_duration_ticks <= 0:
            continue

        detection_complete_time = max(
            detection_complete_time,
            exposure_time + exposure_window_duration_ticks,
        )

        raw_clicks.extend(
            detector.evaluate_window(
                time=exposure_time,
                signal_present=exposure.signal_present,
                window_duration_ticks=exposure_window_duration_ticks,
                rngs=require_detector_rngs(detector_rngs, detector),
                outcome_label=exposure.outcome_label,
                meta=(
                    ("device_id", device_id),
                    ("detector_index", index),
                    ("measurement_label", measurement_label),
                    ("readout_signal_present", exposure.signal_present),
                    ("readout_time_offset_ticks", exposure.time_offset_ticks),
                    ("readout_outcome_label", exposure.outcome_label),
                    ("window_duration_ticks", exposure_window_duration_ticks),
                    (
                        "configured_detection_window_ticks",
                        detection_window_ticks,
                    ),
                    *click_meta,
                    *exposure.meta,
                ),
            )
        )

    return tuple(raw_clicks), detection_complete_time


__all__ = [
    "active_detection_duration_at_arrival",
    "bind_detector_rngs",
    "evaluate_detector_windows",
    "normalize_detectors",
    "require_detector_rngs",
    "validate_gate_model",
]
