"""
Scalar and tuple validation helpers for public boundary objects.

These helpers are intentionally generic and side-effect free so payload/record
dataclasses can share consistent validation semantics. Functions named
``require_*`` return the validated value, often converted to a narrower runtime
form. Functions named ``validate_*`` only raise on invalid input and otherwise
return ``None``.
"""

from __future__ import annotations

import math

from .ids import ensure_nonempty_id


def require_real(value: object, *, field_name: str, type_name: str = "float") -> float:
    """Return `value` as ``float`` after numeric type validation.

    Parameters
    ----------
    value : object
        Candidate numeric value. ``int`` and ``float`` are accepted; ``bool``
        is rejected.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float
        Numeric value converted with ``float(value)``.

    Raises
    ------
    TypeError
        If `value` is not an ``int`` or ``float``, or if it is ``bool``.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be {type_name}")
    return float(value)


def require_finite_real(
    value: object,
    *,
    field_name: str,
    type_name: str = "float",
) -> float:
    """Return a finite real value as ``float``.

    Parameters
    ----------
    value : object
        Candidate numeric value.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float
        Finite numeric value.

    Raises
    ------
    TypeError
        If `value` is not accepted by :func:`require_real`.
    ValueError
        If `value` converts to ``nan`` or infinity.
    """
    resolved = require_real(value, field_name=field_name, type_name=type_name)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def require_non_negative_real(
    value: object,
    *,
    field_name: str,
    type_name: str = "float",
) -> float:
    """Return a finite real value in ``[0.0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate numeric value.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float
        Finite non-negative value.

    Raises
    ------
    TypeError
        If `value` is not accepted by :func:`require_real`.
    ValueError
        If `value` is not finite or is negative.
    """
    resolved = require_finite_real(
        value,
        field_name=field_name,
        type_name=type_name,
    )
    if resolved < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return resolved


def require_positive_real(
    value: object,
    *,
    field_name: str,
    type_name: str = "float",
) -> float:
    """Return a finite real value in ``(0.0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate numeric value.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float
        Finite positive value.

    Raises
    ------
    TypeError
        If `value` is not accepted by :func:`require_real`.
    ValueError
        If `value` is not finite or is less than or equal to zero.
    """
    resolved = require_finite_real(
        value,
        field_name=field_name,
        type_name=type_name,
    )
    if resolved <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def require_probability(value: object, *, field_name: str) -> float:
    """Return a numeric probability in ``[0.0, 1.0]``.

    Parameters
    ----------
    value : object
        Candidate probability. Non-finite float values pass the numeric type
        check but fail the bounds check.
    field_name : str
        Name used in validation error messages.

    Returns
    -------
    float
        Probability value converted to ``float``.

    Raises
    ------
    TypeError
        If `value` is not accepted by :func:`require_real`.
    ValueError
        If the converted value is outside ``[0.0, 1.0]``.
    """
    probability = require_real(value, field_name=field_name)
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"{field_name} must be in [0.0, 1.0]")
    return probability


def require_optional_positive_real(
    value: object | None,
    *,
    field_name: str,
    type_name: str = "float",
) -> float | None:
    """Return ``None`` or a finite positive real value.

    Parameters
    ----------
    value : object or None
        Optional candidate value.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float or None
        ``None`` when `value` is ``None``; otherwise the validated positive
        float.
    """
    if value is None:
        return None
    return require_positive_real(
        value,
        field_name=field_name,
        type_name=type_name,
    )


def require_optional_non_negative_real(
    value: object | None,
    *,
    field_name: str,
    type_name: str = "float",
) -> float | None:
    """Return ``None`` or a finite non-negative real value.

    Parameters
    ----------
    value : object or None
        Optional candidate value.
    field_name : str
        Name used in validation error messages.
    type_name : str, default="float"
        User-facing type name used in the ``TypeError`` message.

    Returns
    -------
    float or None
        ``None`` when `value` is ``None``; otherwise the validated
        non-negative float.
    """
    if value is None:
        return None
    return require_non_negative_real(
        value,
        field_name=field_name,
        type_name=type_name,
    )


def require_optional_probability(
    value: object | None,
    *,
    field_name: str,
) -> float | None:
    """Return ``None`` or a probability in ``[0.0, 1.0]``.

    Parameters
    ----------
    value : object or None
        Optional candidate probability.
    field_name : str
        Name used in validation error messages.

    Returns
    -------
    float or None
        ``None`` when `value` is ``None``; otherwise the validated probability.
    """
    if value is None:
        return None
    return require_probability(value, field_name=field_name)


def require_non_negative_int(
    value: object,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> int:
    """Return an integer in ``[0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate integer value.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer. The default rejects ``bool`` to
        avoid accepting flags where counts or positions are expected.

    Returns
    -------
    int
        The original integer value.

    Raises
    ------
    TypeError
        If `value` is not an integer under the selected bool policy.
    ValueError
        If `value` is negative.
    """
    if allow_bool:
        if not isinstance(value, int):
            raise TypeError(f"{field_name} must be int")
    elif type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def validate_non_negative_int(
    value: object,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> None:
    """Validate an integer in ``[0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate integer value.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer.
    """
    require_non_negative_int(value, field_name=field_name, allow_bool=allow_bool)


def validate_optional_non_negative_int(
    value: object | None,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> None:
    """Validate ``None`` or an integer in ``[0, +inf)``.

    Parameters
    ----------
    value : object or None
        Optional candidate integer.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer when `value` is not ``None``.
    """
    if value is None:
        return
    validate_non_negative_int(value, field_name=field_name, allow_bool=allow_bool)


def require_positive_int(
    value: object,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> int:
    """Return an integer in ``(0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate integer value.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer.

    Returns
    -------
    int
        The original positive integer value.

    Raises
    ------
    TypeError
        If `value` is not an integer under the selected bool policy.
    ValueError
        If `value` is less than or equal to zero.
    """
    if allow_bool:
        if not isinstance(value, int):
            raise TypeError(f"{field_name} must be int")
    elif type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def validate_positive_int(
    value: object,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> None:
    """Validate an integer in ``(0, +inf)``.

    Parameters
    ----------
    value : object
        Candidate integer value.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer.
    """
    require_positive_int(value, field_name=field_name, allow_bool=allow_bool)


def validate_optional_positive_int(
    value: object | None,
    *,
    field_name: str,
    allow_bool: bool = False,
) -> None:
    """Validate ``None`` or an integer in ``(0, +inf)``.

    Parameters
    ----------
    value : object or None
        Optional candidate integer.
    field_name : str
        Name used in validation error messages.
    allow_bool : bool, default=False
        Whether ``bool`` may pass as an integer when `value` is not ``None``.
    """
    if value is None:
        return
    validate_positive_int(value, field_name=field_name, allow_bool=allow_bool)


def validate_non_empty_str(value: str, *, field_name: str) -> None:
    """Validate a non-empty string.

    Parameters
    ----------
    value : str
        Candidate string. The stripped form must be non-empty, but the value is
        not modified.
    field_name : str
        Name used in validation error messages.
    """
    ensure_nonempty_id(value, field_name=field_name)


def validate_optional_non_empty_str(value: str | None, *, field_name: str) -> None:
    """Validate ``None`` or a non-empty string.

    Parameters
    ----------
    value : str or None
        Optional candidate string.
    field_name : str
        Name used in validation error messages.
    """
    if value is None:
        return
    validate_non_empty_str(value, field_name=field_name)


def require_bool(value: object, *, field_name: str) -> bool:
    """Return `value` after requiring exact ``bool`` type.

    Parameters
    ----------
    value : object
        Candidate boolean.
    field_name : str
        Name used in validation error messages.

    Returns
    -------
    bool
        The original boolean value.

    Raises
    ------
    TypeError
        If `value` is not exactly ``bool``.
    """
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool")
    return value


def validate_bool(value: object, *, field_name: str) -> None:
    """Validate exact ``bool`` type.

    Parameters
    ----------
    value : object
        Candidate boolean.
    field_name : str
        Name used in validation error messages.
    """
    require_bool(value, field_name=field_name)


def validate_bit(value: int, *, field_name: str) -> None:
    """Validate a single bit value.

    Parameters
    ----------
    value : int
        Candidate bit. Must be exactly ``0`` or ``1``.
    field_name : str
        Name used in validation error messages.

    Raises
    ------
    TypeError
        If `value` is not exactly ``int``.
    ValueError
        If `value` is not ``0`` or ``1``.
    """
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    if value not in (0, 1):
        raise ValueError(f"{field_name} must be 0 or 1")


def validate_bits(bits: tuple[int, ...], *, field_name: str) -> None:
    """Validate a tuple of bit values.

    Parameters
    ----------
    bits : tuple[int, ...]
        Candidate bit sequence. Each item must be accepted by
        :func:`validate_bit`.
    field_name : str
        Name used in validation error messages.
    """
    if not isinstance(bits, tuple):
        raise TypeError(f"{field_name} must be tuple[int, ...]")
    for bit in bits:
        validate_bit(bit, field_name=field_name)


def validate_probability(value: float, *, field_name: str) -> None:
    """Validate a numeric probability in ``[0.0, 1.0]``.

    Parameters
    ----------
    value : float
        Candidate probability.
    field_name : str
        Name used in validation error messages.
    """
    require_probability(value, field_name=field_name)


def validate_optional_probability(value: float | None, *, field_name: str) -> None:
    """Validate ``None`` or a probability in ``[0.0, 1.0]``.

    Parameters
    ----------
    value : float or None
        Optional candidate probability.
    field_name : str
        Name used in validation error messages.
    """
    if value is None:
        return
    validate_probability(value, field_name=field_name)


def validate_index_tuple(values: tuple[int, ...], *, field_name: str) -> None:
    """Validate a tuple of non-negative integer indices.

    Parameters
    ----------
    values : tuple[int, ...]
        Candidate index tuple. Each element must be a non-negative integer.
    field_name : str
        Name used in validation error messages.
    """
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be tuple[int, ...]")
    for value in values:
        validate_non_negative_int(value, field_name=field_name)


def validate_slot_key(value: str | int, *, field_name: str) -> None:
    """Validate a memory slot key.

    Parameters
    ----------
    value : str or int
        Slot key. String keys must be non-empty; integer keys may be any exact
        ``int`` value accepted by the implementation.
        Integer keys are not range-checked here; callers decide whether
        negative integer keys are meaningful.
    field_name : str
        Name used in validation error messages.
    """
    if isinstance(value, str):
        validate_non_empty_str(value, field_name=field_name)
        return
    if type(value) is not int:
        raise TypeError(f"{field_name} must be str or int")


__all__ = [
    "require_bool",
    "require_finite_real",
    "require_non_negative_int",
    "require_non_negative_real",
    "require_optional_non_negative_real",
    "require_optional_positive_real",
    "require_optional_probability",
    "require_positive_int",
    "require_positive_real",
    "require_probability",
    "require_real",
    "validate_bool",
    "validate_bit",
    "validate_bits",
    "validate_index_tuple",
    "validate_non_empty_str",
    "validate_non_negative_int",
    "validate_optional_non_empty_str",
    "validate_optional_non_negative_int",
    "validate_optional_positive_int",
    "validate_optional_probability",
    "validate_positive_int",
    "validate_probability",
    "validate_slot_key",
]
