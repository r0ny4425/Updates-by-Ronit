import pytest

from simyuj.qstate import InvalidLayoutError, StateLayout, SubsystemId


def test_layout_axis_and_dimension_lookup() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    layout = StateLayout((q0, q1), (2, 3))

    assert layout.size == 2
    assert layout.hilbert_dim == 6
    assert layout.axis_of(q0) == 0
    assert layout.axis_of(q1) == 1
    assert layout.axes_of((q1, q0)) == (1, 0)
    assert layout.dim_of(q1) == 3


def test_layout_requires_unique_subsystems() -> None:
    q0 = SubsystemId("q0")

    with pytest.raises(InvalidLayoutError):
        StateLayout((q0, q0), (2, 2))


def test_layout_without_reorder_and_combine() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    q2 = SubsystemId("q2")
    left = StateLayout((q0, q1), (2, 3))
    right = StateLayout((q2,), (5,))

    without_q0 = left.without((q0,))
    assert without_q0.subsystems == (q1,)
    assert without_q0.dims == (3,)
    assert left.reorder((q1, q0)).dims == (3, 2)
    assert left.combine(right).subsystems == (q0, q1, q2)
    assert left.combine(right).dims == (2, 3, 5)


def test_layout_combine_rejects_duplicate_subsystem() -> None:
    q0 = SubsystemId("q0")
    left = StateLayout((q0,), (2,))
    right = StateLayout((q0,), (2,))

    with pytest.raises(InvalidLayoutError):
        left.combine(right)
