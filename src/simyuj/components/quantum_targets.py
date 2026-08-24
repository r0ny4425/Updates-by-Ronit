"""Adapters for resolving qstate subsystem targets from component signals.

This module serves signal-level component code. It is not a general qstate
layout resolver for memories or multi-qubit layouts.

Two questions, not one
----------------------

A signal may carry a qstate record, and that record may or may not be the thing
that propagates. :func:`qstate_targets_from_signal` answers *which subsystem*;
:func:`qstate_payload_role` answers *what the subsystem is for*. Transport code
needs both, because only the second decides whether channel loss may destroy the
record.

Every signal in the simulator today carries a ``"qubit"`` record, so the two
questions currently have the same answer and ``role == "qubit"`` is equivalent
to ``state_ref is not None``. They diverge for a coherent pulse that carries a
polarization state beside its amplitude, which is why the role exists.

:func:`qubit_carrier_targets_from_signal` asks both at once, for the components
that only work when the answers agree. A qstate-measuring component -- a
detector array, a Bell analyzer, a memory -- measures, collapses, or stores the
record it is handed, which is correct only when that record *is* the carrier.
Handed a mode record it would silently treat a whole coherent pulse as one
photon, so those callers resolve targets through the requiring form rather than
the role-agnostic one.
"""

from __future__ import annotations

from typing import Literal, Optional

from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import Signal

QstatePayloadRole = Literal["qubit", "mode"]
"""Role a signal's qstate record plays for transport code."""


def _single_state_handle(signal: Signal) -> SubsystemHandle:
    """Return the one ``SubsystemHandle`` a qstate-backed signal must carry.

    Shared by :func:`qstate_targets_from_signal` and
    :func:`qstate_payload_role` so the arity rule and its message exist once.
    """
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
            The record **is** the propagating carrier. Channel loss destroys
            it; channel noise acts on it. Every signal in the simulator today.
        ``"mode"``
            The record **describes** the mode a classical amplitude occupies --
            a polarization state beside a ``coherent_state``, for example. Loss
            scales the amplitude and leaves the record alone; noise still acts
            on it.
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
    **Presence and role are different questions, and conflating them is a bug
    waiting to happen.** ``state_ref is not None`` says a record exists;
    only the role says whether loss may destroy it. Gating a Bernoulli survival
    trial on presence would subject a polarization state to probabilistic
    annihilation instead of scaling an amplitude -- which produces plausible
    numbers, because it looks exactly like an ordinary lossy link. Use
    ``role is not None`` to gate noise, which applies to either role, and
    ``role == "qubit"`` to gate loss, which does not.

    **The one-handle rule is load-bearing, not incidental.** The role is read
    from the single handle in ``state_targets``, so a signal carrying both a
    carrier qubit *and* a mode descriptor would need two handles and would have
    no single role. Nothing constructs such a signal and nothing is planned to;
    if one is ever needed, this function is the thing that has to change, not
    its callers.

    ``SubsystemHandle.kind`` defaults to ``"qubit"``, so a handle built without
    an explicit role is a carrier. That direction is deliberate: forgetting to
    stamp a role gives today's behaviour, never a silently unprotected record.

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

    This shared adapter is used by qstate-backed components that consume
    ``Signal.state_targets``. Current simulator behavior supports exactly one
    target per signal; multi-target and multi-qubit signal handling is future
    work.

    When present, ``SubsystemHandle.metadata`` key ``"qstate_subsystem"``
    takes precedence over the handle label.

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
    This helper resolves identity only. It does not check whether the returned
    subsystem currently exists in a qstate manager and does not mutate qstate.

    It is **role-agnostic** on purpose: a mode record needs its subsystem
    resolved just as a carrier does, so that noise can be applied to it. Callers
    that need to know whether loss may destroy the record ask
    :func:`qstate_payload_role` separately.

    The one-target restriction matches current signal-level component behavior.
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
        Single resolved qstate subsystem target, exactly as
        :func:`qstate_targets_from_signal` returns it.

    Raises
    ------
    ValueError
        If the signal carries no ``state_ref``, does not carry exactly one
        state target, or carries a record whose role is ``"mode"``.
    TypeError
        If the single target is not a ``SubsystemHandle``.

    Notes
    -----
    **Presence is not enough, which is the whole reason this exists.** A bare
    coherent pulse has no ``state_ref`` and is already refused by the underlying
    resolver. A *polarized* coherent pulse has one, so it passes that check and
    reaches a measurement that would treat the polarization record as the thing
    that propagates -- routing an entire pulse to one detector, making the
    double-click rate identically zero at every mean photon number and the click
    rate independent of it. The run completes and the numbers look reasonable,
    which is what makes the loud failure here worth the line.

    Use :func:`qstate_targets_from_signal` directly where the role genuinely
    does not matter: a channel resolves a mode record's subsystem so that noise
    can act on it, and must keep doing so.
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
