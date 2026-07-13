import pytest

from simyuj.primitives.meta import freeze_meta, validate_meta


def test_freeze_meta_preserves_public_metadata_order_and_duplicates() -> None:
    assert freeze_meta({"a": 1, "b": "x"}) == (("a", 1), ("b", "x"))
    assert freeze_meta((("a", 1), ("a", 2))) == (("a", 1), ("a", 2))
    assert freeze_meta(None) == ()


def test_metadata_values_are_hashable_for_hashable_records_by_default() -> None:
    with pytest.raises(TypeError, match="meta values must be hashable"):
        validate_meta((("bad", []),))


def test_metadata_can_carry_opaque_backend_hints_when_allowed() -> None:
    validate_meta((("bad", []),), require_hashable=False)
