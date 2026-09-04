"""Source-side preparation reports.

Source reports describe successful qstate preparation choices. They are local
control-plane payloads and do not represent inter-node classical messages. Any
qstate identifiers in a report are descriptive handles for correlation and
follow-up requests; they do not transfer qstate ownership to the receiving
agent.

Two report types live here because there are two kinds of source.
:class:`SourcePreparationReport` describes a qstate-backed preparation and
carries a ``state_ref`` and sampler choice. :class:`CoherentPulsePreparationReport`
describes an optical preparation and carries the classical choices that produced
the amplitude instead. Preparing an amplitude creates no quantum state; an
unpolarized pulse therefore reaches no qstate at all, while a polarized one
additionally has one record per pulse prepared for the mode it occupies. Neither
report borrows the other's fields: a report claiming a sampler that does not
exist, or a state reference that was never allocated, is false in the record an
agent reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeVar, Union

from simyuj.components.ports import Port
from simyuj.engine.timeline import Timeline
from simyuj.primitives.coherent_state import CoherentState
from simyuj.qstate import StateRef, SubsystemId


@dataclass(frozen=True, slots=True)
class SourcePreparationReport:
    """
    Immutable report for one successful source preparation.

    Parameters
    ----------
    report_id : str
        Stable identifier chosen by the source component.
    device_id : str
        Source device identifier.
    time : int
        Simulation tick at which the prepared signal or pair is emitted.
    attempt_index : int
        One-based hardware attempt index.
    emission_index : int or None
        One-based successful emission index, such as a photon or pair index.
    signal_ids : tuple[str, ...]
        Emitted signal identifiers for this preparation.
    sampler_index, sampler_label : int, str or None
        State-sampler choice selected by the source.
    state_ref : int
        Qstate reference containing the prepared subsystem targets. This is a
        descriptive identifier, not ownership transfer.
    state_targets : tuple[SubsystemId, ...]
        Qstate subsystem IDs prepared by the source. Agents must not mutate
        these qstate subsystems directly; they should schedule explicit
        simulator requests or component events instead.
    emission_slot_tick, emission_delay_ticks : int
        Nominal source slot and sampled delay used for the emission event.
    meta : tuple[tuple[str, object], ...]
        Additional immutable metadata.
    """

    report_id: str
    device_id: str
    time: int
    attempt_index: int
    emission_index: int | None
    signal_ids: tuple[str, ...]
    sampler_index: int
    sampler_label: str | None
    state_ref: StateRef
    state_targets: tuple[SubsystemId, ...]
    emission_slot_tick: int
    emission_delay_ticks: int
    meta: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class CoherentPulsePreparationReport:
    """
    Immutable report for one emitted coherent pulse.

    Parameters
    ----------
    report_id : str
        Stable identifier chosen by the source component.
    device_id : str
        Source device identifier.
    time : int
        Simulation tick at which the pulse is emitted.
    pulse_index : int
        One-based pulse index. There is no separate attempt index: a coherent
        source has no emission Bernoulli, so every active slot emits and the two
        counters would always agree.
    signal_ids : tuple[str, ...]
        Emitted signal identifiers for this preparation.
    coherent_state : CoherentState
        The amplitude actually emitted. Mean photon number and phase are derived
        from it rather than stored again.
    emission_slot_tick, emission_delay_ticks : int
        Nominal source slot and sampled delay used for the emission event.
    mean_photon_number : float
        Mean photon number selected for this pulse.
    intensity_index : int
        Position of that value in the intensity selector's alphabet.
    carrier_phase_rad : float
        Source carrier phase for this pulse.
    encoding_phase_rad : float
        Deliberate encoding phase for this pulse.
    encoding_phase_index : int
        Position of that phase in the encoding selector's alphabet. This is what
        a protocol agent decodes; for a two-phase DPS alphabet ``0`` is phase
        ``0`` and ``1`` is phase ``pi``.
    polarization : tuple[complex, complex] or None, default=None
        Jones vector selected for this pulse, when polarization is modelled.
    polarization_index : int or None, default=None
        Position of that state in the polarization selector's alphabet.
    meta : tuple[tuple[str, object], ...], default=()
        Additional immutable metadata.

    Notes
    -----
    There is no ``sampler_*`` field: nothing here samples a photon number, so a
    report naming a sampler choice would misdescribe what happened.

    There is also no ``state_ref`` and no ``state_targets``, and that is a
    narrower statement than it looks. Preparing an amplitude allocates nothing,
    so for an unpolarized pulse there is no reference such a field could carry.
    A *polarized* pulse does allocate one record per pulse for the mode it
    occupies, and this report does not name it -- an agent reading a polarized
    preparation holds the Jones vector and its alphabet index, and reaches the
    record only through the emitted signal's ``state_ref``.

    The carrier and encoding phases are recorded **separately** rather than
    pre-summed, so a later analysis can attribute a visibility loss to carrier
    drift rather than encoding.

    The applied encoding phase is **not** recoverable from the amplitude:
    ``CoherentState.phase_rad`` is the total wrapped phase. Protocol code must
    read ``encoding_phase_index`` from this report, never ``arg(alpha)`` from a
    received signal.
    """

    report_id: str
    device_id: str
    time: int
    pulse_index: int
    signal_ids: tuple[str, ...]
    coherent_state: CoherentState
    emission_slot_tick: int
    emission_delay_ticks: int
    mean_photon_number: float
    intensity_index: int
    carrier_phase_rad: float
    encoding_phase_rad: float
    encoding_phase_index: int
    polarization: Optional[tuple[complex, complex]] = None
    polarization_index: Optional[int] = None
    meta: tuple[tuple[str, object], ...] = ()


SourceReport = Union[SourcePreparationReport, CoherentPulsePreparationReport]
"""Any report a source component may store and transmit."""

_ReportT = TypeVar("_ReportT", bound=SourceReport)
"""One concrete source report type, tying ``reports`` to ``report``.

``list`` is invariant, so binding the element type is what accepts each source's
own homogeneous list while rejecting a mixed one.
"""


def store_source_report(
    *,
    reports: list[_ReportT],
    report_port: Port,
    report: _ReportT,
    timeline: Timeline,
    source: object,
) -> None:
    """
    Store a source report and optionally emit it through the report port.

    Reports are always appended to the source-local ``reports`` list. They are
    transmitted only when the source's classical ``report`` port is connected.
    """
    reports.append(report)

    connection = report_port.connection
    if connection is None:
        return

    connection.transmit(
        report,
        timeline,
        time=timeline.current_time,
        source=source,
        subsystem_id="components",
        meta={
            "device_id": report.device_id,
            "report_id": report.report_id,
            "report_kind": "source_preparation",
        },
    )


__all__ = [
    "CoherentPulsePreparationReport",
    "SourcePreparationReport",
    "SourceReport",
    "store_source_report",
]
