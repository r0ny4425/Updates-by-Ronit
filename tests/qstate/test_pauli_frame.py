from __future__ import annotations

import pytest

from simyuj.qstate.ops.frame import (
    PauliFrame,
    correction_for_bell,
    correction_for_entanglement_swap,
    identity_frame,
    update_after_bsm,
    update_after_swap,
    update_after_teleport,
)


def test_pauli_frame_validates_tuple_bits_and_sizes() -> None:
    frame = PauliFrame((0, 1), (1, 0))

    assert frame.x == (0, 1)
    assert frame.z == (1, 0)
    assert frame.size == 2

    with pytest.raises(TypeError, match="x"):
        PauliFrame([0], (0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="0 or 1"):
        PauliFrame((2,), (0,))
    with pytest.raises(ValueError, match="0 or 1"):
        PauliFrame((True,), (0,))
    with pytest.raises(ValueError, match="same length"):
        PauliFrame((0, 1), (0,))


def test_identity_frame_builds_zero_frame() -> None:
    assert identity_frame(3) == PauliFrame((0, 0, 0), (0, 0, 0))
    assert identity_frame(0) == PauliFrame((), ())

    with pytest.raises(TypeError, match="size"):
        identity_frame(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        identity_frame(-1)


def test_pauli_frame_compose_xors_matching_frames() -> None:
    left = PauliFrame((1, 0, 1), (0, 1, 1))
    right = PauliFrame((0, 1, 1), (1, 1, 0))

    assert left.compose(right) == PauliFrame((1, 1, 0), (1, 0, 1))

    with pytest.raises(TypeError, match="PauliFrame"):
        left.compose("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same size"):
        left.compose(identity_frame(2))


def test_pauli_frame_with_correction_composes_one_qubit_bits() -> None:
    frame = PauliFrame((1, 0), (0, 1))

    assert frame.with_correction(1, x=1, z=1) == PauliFrame((1, 1), (0, 0))

    with pytest.raises(TypeError, match="qubit"):
        frame.with_correction(0.0)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="out of range"):
        frame.with_correction(2)
    with pytest.raises(ValueError, match="0 or 1"):
        frame.with_correction(0, x=2)


def test_correction_for_bell_uses_bell_label_bits() -> None:
    assert correction_for_bell("phi+") == (0, 0)
    assert correction_for_bell("phi-") == (0, 1)
    assert correction_for_bell("psi+") == (1, 0)
    assert correction_for_bell("psi-") == (1, 1)
    assert correction_for_bell("psi_minus") == (1, 1)


def test_update_after_bsm_and_teleport_apply_corrections_to_frame() -> None:
    frame = identity_frame(2)

    assert update_after_bsm(frame, "psi-", qubit=1) == PauliFrame((0, 1), (0, 1))
    assert update_after_teleport(frame, "phi-", qubit=0) == PauliFrame(
        (0, 0),
        (1, 0),
    )

    with pytest.raises(TypeError, match="PauliFrame"):
        update_after_bsm("bad", "phi+")  # type: ignore[arg-type]


def test_update_after_swap_xors_two_bell_outcomes() -> None:
    assert update_after_swap("phi+", "phi+") == (0, 0)
    assert update_after_swap("psi-", "phi-") == (1, 0)
    assert update_after_swap("psi+", "psi-") == (0, 1)


def test_correction_for_entanglement_swap_includes_input_pair_frames() -> None:
    assert correction_for_entanglement_swap("phi+", "phi+", "psi-") == (1, 1)
    assert correction_for_entanglement_swap("phi+", "phi-", "psi-") == (1, 0)
    assert correction_for_entanglement_swap("psi+", "phi-", "psi-") == (0, 0)

    with pytest.raises(ValueError, match="unsupported"):
        correction_for_entanglement_swap("phi+", "omega+", "psi-")
