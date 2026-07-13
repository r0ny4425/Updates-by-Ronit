"""Adapters for resolving qstate subsystem targets from component signals.

This module serves signal-level component code. It is not a general qstate
layout resolver for memories or multi-qubit layouts.
"""

from __future__ import annotations

from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import Signal


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

    The one-target restriction matches current signal-level component behavior.
    Memory and layout code should use their own qstate interfaces rather than
    this adapter.
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

    for key, value in handle.metadata:
        if key == "qstate_subsystem":
            return (SubsystemId(str(value)),)

    return (SubsystemId(handle.label),)


__all__ = [
    "qstate_targets_from_signal",
]
