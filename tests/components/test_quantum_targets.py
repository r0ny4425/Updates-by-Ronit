from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from simyuj.components.quantum_targets import qstate_targets_from_signal
from simyuj.primitives.subsystems import SubsystemHandle
from simyuj.qstate import SubsystemId
from simyuj.signal import EncodingScheme, Signal, SignalKind


def _signal(
    *,
    state_ref: int | None = 1,
    state_targets: tuple[SubsystemHandle, ...] | None = None,
) -> Signal:
    if state_targets is None:
        state_targets = (
            SubsystemHandle(
                label="fallback:q0",
                metadata=(("qstate_subsystem", "metadata:q0"),),
            ),
        )

    return Signal(
        id="s1",
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="alice",
        state_ref=state_ref,
        state_targets=state_targets,
    )


def test_signal_without_state_ref_raises_value_error() -> None:
    with pytest.raises(ValueError, match="state_ref"):
        qstate_targets_from_signal(_signal(state_ref=None))


def test_signal_with_zero_state_targets_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="currently require exactly one state target",
    ):
        qstate_targets_from_signal(_signal(state_targets=()))


def test_signal_with_two_state_targets_raises_value_error() -> None:
    target = SubsystemHandle(label="q0")

    with pytest.raises(
        ValueError,
        match="currently require exactly one state target",
    ):
        qstate_targets_from_signal(_signal(state_targets=(target, target)))


def test_non_subsystem_handle_target_raises_type_error() -> None:
    signal = replace(
        _signal(),
        state_targets=cast(tuple[SubsystemHandle, ...], ("not a handle",)),
        validation_flag=False,
    )

    with pytest.raises(TypeError, match="SubsystemHandle"):
        qstate_targets_from_signal(signal)


def test_subsystem_handle_metadata_qstate_subsystem_wins() -> None:
    signal = _signal(
        state_targets=(
            SubsystemHandle(
                label="fallback:q0",
                metadata=(("qstate_subsystem", "metadata:q0"),),
            ),
        )
    )

    assert qstate_targets_from_signal(signal) == (SubsystemId("metadata:q0"),)


def test_absent_qstate_subsystem_metadata_falls_back_to_handle_label() -> None:
    signal = _signal(
        state_targets=(
            SubsystemHandle(
                label="fallback:q0",
                metadata=(("other", "metadata:q0"),),
            ),
        )
    )

    assert qstate_targets_from_signal(signal) == (SubsystemId("fallback:q0"),)
