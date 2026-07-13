from __future__ import annotations

import pytest

from simyuj.primitives.subsystems import SubsystemHandle


def test_subsystem_handle_preserves_identity_fields_and_metadata() -> None:
    backend_hint = {"opaque": True}
    handle = SubsystemHandle(
        label="q0",
        kind="qubit",
        index=0,
        metadata=(("backend_hint", backend_hint),),
    )

    assert handle.label == "q0"
    assert handle.kind == "qubit"
    assert handle.index == 0
    assert handle.metadata == (("backend_hint", backend_hint),)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"label": ""}, ValueError, "label must be a non-empty str"),
        (
            {"label": "q0", "kind": "ancilla"},
            ValueError,
            "kind must be 'qubit' or 'mode'",
        ),
        ({"label": "q0", "index": True}, TypeError, "index must be int or None"),
        ({"label": "q0", "index": -1}, ValueError, "index must be non-negative"),
        (
            {"label": "q0", "metadata": ("not-a-pair",)},
            TypeError,
            "metadata must contain",
        ),
    ],
)
def test_subsystem_handles_reject_ambiguous_public_identity(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SubsystemHandle(**kwargs)  # type: ignore[arg-type]
