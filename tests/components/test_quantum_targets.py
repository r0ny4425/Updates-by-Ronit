from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from simyuj.components.quantum_targets import (
    qstate_payload_role,
    qstate_targets_from_signal,
)
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


def test_role_is_none_without_a_qstate_record() -> None:
    # A bare coherent pulse. The role function is total where
    # qstate_targets_from_signal raises, because transport code has to ask
    # before it knows whether it may resolve targets at all.
    assert qstate_payload_role(_signal(state_ref=None)) is None


def test_role_defaults_to_qubit_so_an_unstamped_handle_is_a_carrier() -> None:
    # SubsystemHandle.kind defaults to "qubit". Forgetting to stamp a role must
    # give today's behaviour -- a protected carrier -- never a silently
    # unprotected mode record.
    assert SubsystemHandle(label="q0").kind == "qubit"
    assert qstate_payload_role(_signal()) == "qubit"


def test_role_matches_the_presence_check_for_every_signal_today() -> None:
    # The property that lets the discriminator land before polarization exists:
    # while no handle is ever "mode", role == "qubit" is exactly
    # state_ref is not None, so a channel branching on either behaves the same.
    for signal in (_signal(), _signal(state_ref=None)):
        presence = signal.state_ref is not None
        assert (qstate_payload_role(signal) == "qubit") is presence


def test_role_reports_mode_when_the_handle_says_so() -> None:
    # The case the presence check cannot express: a record that exists but is
    # not the carrier. Loss must scale the sibling amplitude and leave this
    # record alone.
    signal = _signal(state_targets=(SubsystemHandle(label="pol:0", kind="mode"),))

    assert qstate_payload_role(signal) == "mode"
    assert signal.state_ref is not None  # a presence check would say "carrier"


def test_role_resolves_targets_for_a_mode_record_too() -> None:
    # qstate_targets_from_signal is role-agnostic on purpose: a mode record
    # needs its subsystem resolved so that channel noise can be applied to it.
    signal = _signal(
        state_targets=(
            SubsystemHandle(
                label="pol:0",
                kind="mode",
                metadata=(("qstate_subsystem", "alice:pol:0"),),
            ),
        )
    )

    assert qstate_payload_role(signal) == "mode"
    assert qstate_targets_from_signal(signal) == (SubsystemId("alice:pol:0"),)


@pytest.mark.parametrize(
    "resolve",
    [qstate_targets_from_signal, qstate_payload_role],
    ids=["targets", "role"],
)
def test_both_resolvers_share_one_arity_rule(resolve) -> None:
    target = SubsystemHandle(label="q0")

    with pytest.raises(
        ValueError,
        match="currently require exactly one state target",
    ):
        resolve(_signal(state_targets=(target, target)))

    with pytest.raises(
        ValueError,
        match="currently require exactly one state target",
    ):
        resolve(_signal(state_targets=()))
