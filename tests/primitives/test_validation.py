from __future__ import annotations

import math

import pytest

from simyuj.primitives.validation import (
    require_finite_real,
    require_non_negative_int,
    require_positive_int,
    require_probability,
    validate_slot_key,
)


def test_physical_numeric_boundaries_accept_finite_ints_and_floats() -> None:
    assert require_finite_real(1, field_name="value") == pytest.approx(1.0)
    assert require_finite_real(1.5, field_name="value") == pytest.approx(1.5)


def test_slot_indices_are_exact_non_negative_integers() -> None:
    assert require_non_negative_int(0, field_name="index") == 0
    assert require_non_negative_int(2, field_name="index") == 2

    with pytest.raises(TypeError, match="index must be int"):
        require_non_negative_int(True, field_name="index")


def test_slot_keys_accept_strings_and_exact_ints_without_range_check() -> None:
    validate_slot_key("memory-a", field_name="slot_key")
    validate_slot_key(0, field_name="slot_key")
    validate_slot_key(-1, field_name="slot_key")

    with pytest.raises(ValueError, match="slot_key must be non-empty"):
        validate_slot_key("", field_name="slot_key")
    with pytest.raises(TypeError, match="slot_key must be str or int"):
        validate_slot_key(True, field_name="slot_key")


def test_positive_counts_reject_zero() -> None:
    assert require_positive_int(1, field_name="count") == 1
    with pytest.raises(ValueError, match="count must be positive"):
        require_positive_int(0, field_name="count")


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (True, TypeError, "value must be numeric"),
        (object(), TypeError, "value must be numeric"),
        (math.inf, ValueError, "value must be finite"),
        (-math.inf, ValueError, "value must be finite"),
        (math.nan, ValueError, "value must be finite"),
    ],
)
def test_physical_numeric_boundaries_reject_ambiguous_values(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        require_finite_real(value, field_name="value", type_name="numeric")


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
def test_probabilities_must_stay_in_closed_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match="probability must be in \\[0.0, 1.0\\]"):
        require_probability(value, field_name="probability")
