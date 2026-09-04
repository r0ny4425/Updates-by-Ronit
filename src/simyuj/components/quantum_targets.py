"""Adapters for resolving qstate subsystem targets from component signals.

This module serves signal-level component code. It is not a general qstate
layout resolver for memories or multi-qubit layouts.

A signal may carry a qstate record, and that record may or may not be the thing
that propagates. :func:`qstate_targets_from_signal` answers *which subsystem*;
:func:`qstate_payload_role` answers *what the subsystem is for*;
:func:`qubit_carrier_targets_from_signal` asks both and requires the record to be
the carrier.

For a photon signal the record is the carrier and the two questions have the same
answer. They diverge for a polarized coherent pulse, whose record describes the
mode its amplitude occupies rather than the thing that propagates.
"""

from __future__ import annotations

from typing import Literal, Optional

from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import Signal

QstatePayloadRole = Literal["qubit", "mode"]
"""Role a signal's qstate record plays for transport code."""


def _single_state_handle(signal: Signal) -> SubsystemHandle:
    """Return the one ``SubsystemHandle`` a qstate-backed signal must carry."""
    if signal.state_ref is None:
        raise ValueError("quantum signal must carry state_ref")

    if len(signal.state_targets) != 1:
        raise ValueError(
            "qstate-backed signal operations currently require exactly one "
            "state target; multi-target support is future work"
        )

    handle = signal.state_targets[0]
    if not isinstance(handle, SubsystemHandle):
        raise TypeError("signal.state_targets must contain SubsystemHandle")

    return handle


def qstate_payload_role(signal: Signal) -> Optional[QstatePayloadRole]:
    """Return the role a signal's qstate record plays, or ``None``.

    Parameters
    ----------
    signal : Signal
        Any signal. A signal with ``state_ref`` must carry exactly one
        ``SubsystemHandle``.

    Returns
    -------
    {"qubit", "mode"} or None
        ``"qubit"``
            The record **is** the propagating carrier. Loss destroys it, noise
            acts on it.
        ``"mode"``
            The record **describes** the mode a classical amplitude occupies, as
            a polarization state beside a ``coherent_state`` does. Loss scales
            the amplitude and leaves the record alone; noise still acts on it.
        ``None``
            No qstate record at all. A bare coherent pulse.

    Raises
    ------
    ValueError
        If the signal carries ``state_ref`` but not exactly one state target.
    TypeError
        If the single target is not a ``SubsystemHandle``.

    Notes
    -----
    Gate noise on ``role is not None``, which applies to either role, and loss on
    ``role == "qubit"``, which does not. Gating loss on ``state_ref is not None``
    instead would subject a polarization state to probabilistic annihilation
    rather than scaling an amplitude, and the result looks exactly like an
    ordinary lossy link.

    ``SubsystemHandle.kind`` defaults to ``"qubit"``, so an unstamped handle is a
    carrier.

    Examples
    --------
    >>> from simyuj.primitives.subsystems import SubsystemHandle
    >>> from simyuj.signal import EncodingScheme, Signal, SignalKind
    >>> photon = Signal(
    ...     id="p1",
    ...     signal_kind=SignalKind.PHOTON,
    ...     encoding_scheme=EncodingScheme.POLARIZATION,
    ...     emission_time=0,
    ...     origin="src",
    ...     state_ref=3,
    ...     state_targets=(SubsystemHandle(label="src:photon:1"),),
    ... )
    >>> qstate_payload_role(photon)
    'qubit'
    """
    if signal.state_ref is None:
        return None
    return _single_state_handle(signal).kind


def qstate_targets_from_signal(signal: Signal) -> tuple[SubsystemId, ...]:
    """Return qstate subsystem targets encoded in a signal.

    Exactly one target per signal is supported; multi-target handling is future
    work. ``SubsystemHandle.metadata`` key ``"qstate_subsystem"`` takes
    precedence over the handle label when present.

    Parameters
    ----------
    signal : Signal
        Qstate-backed signal carrying ``state_ref`` and exactly one
        ``SubsystemHandle`` in ``state_targets``.

    Returns
    -------
    tuple[SubsystemId, ...]
        Single resolved qstate subsystem target.

    Raises
    ------
    ValueError
        If the signal has no ``state_ref`` or does not carry exactly one state
        target.
    TypeError
        If the single target is not a ``SubsystemHandle``.

    Notes
    -----
    Resolves identity only: it does not check whether the subsystem exists in a
    qstate manager and does not mutate qstate.

    Role-agnostic -- a mode record needs its subsystem resolved just as a carrier
    does, so that noise can be applied to it. Callers that need to know whether
    loss may destroy the record ask :func:`qstate_payload_role` separately.

    Memory and layout code should use their own qstate interfaces rather than
    this adapter.
    """
    handle = _single_state_handle(signal)

    for key, value in handle.metadata:
        if key == "qstate_subsystem":
            return (SubsystemId(str(value)),)

    return (SubsystemId(handle.label),)


def qubit_carrier_targets_from_signal(signal: Signal) -> tuple[SubsystemId, ...]:
    """Return a signal's qstate targets, requiring the record to be the carrier.

    The guarded form of :func:`qstate_targets_from_signal`, for components that
    measure, collapse, or store the record they are handed.

    Parameters
    ----------
    signal : Signal
        Signal whose qstate record must be the propagating carrier.

    Returns
    -------
    tuple[SubsystemId, ...]
        Single resolved qstate subsystem target.

    Raises
    ------
    ValueError
        If the signal carries no ``state_ref``, does not carry exactly one
        state target, or carries a record whose role is ``"mode"``.
    TypeError
        If the single target is not a ``SubsystemHandle``.

    Notes
    -----
    Presence is not enough. A bare coherent pulse has no ``state_ref`` and is
    already refused by the underlying resolver. A *polarized* coherent pulse has
    one, so it passes that check and would reach a measurement treating the
    polarization record as the thing that propagates -- routing an entire pulse
    to one detector, and completing with plausible numbers.

    Use :func:`qstate_targets_from_signal` directly where the role does not
    matter, as a channel does when applying noise to a mode record.
    """
    role = qstate_payload_role(signal)

    if role == "mode":
        raise ValueError(
            f"signal {signal.id!r} carries a qstate record with kind='mode': "
            "the record describes the optical mode a coherent amplitude "
            "occupies rather than the propagating carrier, so measuring it "
            "would treat the whole pulse as a single photon and route it to "
            "one detector; optical detection of a coherent pulse is not "
            "implemented yet, so no component here can receive this signal"
        )

    return qstate_targets_from_signal(signal)


__all__ = [
    "QstatePayloadRole",
    "qstate_payload_role",
    "qstate_targets_from_signal",
    "qubit_carrier_targets_from_signal",
]
