import pytest

from simyuj.qstate import StateLayout, SubsystemId, SubsystemNotFoundError
from simyuj.qstate.space.target import resolve_one, resolve_targets, resolve_two


def test_resolves_targets_by_id_name_and_axis() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    layout = StateLayout((q0, q1), (2, 2))

    assert resolve_one(layout, q0) == 0
    assert resolve_one(layout, "q1") == 1
    assert resolve_one(layout, 1) == 1
    assert resolve_targets(layout, (q1, q0)) == (1, 0)
    assert resolve_two(layout, (q0, q1)) == (0, 1)


def test_rejects_duplicate_resolved_targets() -> None:
    q0 = SubsystemId("q0")
    layout = StateLayout((q0,), (2,))

    with pytest.raises(ValueError):
        resolve_targets(layout, (q0, "q0"))


def test_rejects_missing_targets() -> None:
    q0 = SubsystemId("q0")
    layout = StateLayout((q0,), (2,))

    with pytest.raises(SubsystemNotFoundError):
        resolve_one(layout, "q1")

    with pytest.raises(SubsystemNotFoundError):
        resolve_one(layout, 4)
