import pytest

from simyuj.primitives.ids import (
    as_device_id,
    as_link_id,
    as_node_id,
    as_round_id,
    as_session_id,
    build_round_id,
    ensure_nonempty_id,
)


@pytest.mark.parametrize(
    ("value", "field_name"),
    [
        ("", "id"),
        ("   ", "session_id"),
    ],
)
def test_ensure_nonempty_id_rejects_blank_values(value: str, field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be non-empty"):
        ensure_nonempty_id(value, field_name=field_name)


@pytest.mark.parametrize("value", [None, 123, 1.5, b"abc"])
def test_ensure_nonempty_id_rejects_non_string_values(value: object) -> None:
    with pytest.raises(TypeError, match="id must be str"):
        ensure_nonempty_id(value)


def test_identifier_validation_preserves_exact_public_name() -> None:
    original = "  node-a  "
    assert ensure_nonempty_id(original) == original


@pytest.mark.parametrize(
    ("wrapper", "value"),
    [
        (as_node_id, "node-a"),
        (as_link_id, "link-a-b"),
        (as_device_id, "memory-0"),
        (as_session_id, "session-1"),
        (as_round_id, "session-1:round:0"),
    ],
)
def test_typed_id_wrappers_preserve_public_identifier(wrapper, value: str) -> None:
    typed_id = wrapper(value)

    assert typed_id == value
    assert isinstance(typed_id, str)


@pytest.mark.parametrize(
    ("session_id", "index", "expected"),
    [
        ("s1", 0, "s1:round:0"),
        ("session-a", 12, "session-a:round:12"),
    ],
)
def test_round_ids_are_deterministic_session_scoped_names(
    session_id: str,
    index: int,
    expected: str,
) -> None:
    assert build_round_id(session_id, index) == expected
    assert build_round_id(session_id, index) == expected


@pytest.mark.parametrize("session_id", ["", "   "])
def test_build_round_id_rejects_blank_session_id(session_id: str) -> None:
    with pytest.raises(ValueError, match="session_id must be non-empty"):
        build_round_id(session_id, 0)


@pytest.mark.parametrize(
    ("index", "error_type", "message"),
    [
        (-1, ValueError, "index must be non-negative"),
        (1.2, TypeError, "index must be int"),
        ("1", TypeError, "index must be int"),
        (True, TypeError, "index must be int"),
    ],
)
def test_round_id_index_must_be_a_non_negative_integer(
    index: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        build_round_id("s1", index)  # type: ignore[arg-type]
