"""Source-side preparation reports.

Source reports describe successful qstate preparation choices. They are local
control-plane payloads and do not represent inter-node classical messages. Any
qstate identifiers in a report are descriptive handles for correlation and
follow-up requests; they do not transfer qstate ownership to the receiving
agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from simyuj.components.ports import Port
from simyuj.engine.timeline import Timeline
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


def store_source_report(
    *,
    reports: list[SourcePreparationReport],
    report_port: Port,
    report: SourcePreparationReport,
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


__all__ = ["SourcePreparationReport", "store_source_report"]
