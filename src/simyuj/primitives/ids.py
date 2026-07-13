"""
Typed identifier primitives for simulation objects.

The aliases in this module provide lightweight type markers for common
identifier roles. They remain plain strings at runtime. Validation is limited
to type and non-empty checks; callers keep ownership of naming schemes and any
normalization policy.
"""

from __future__ import annotations

from typing import NewType
from uuid import UUID

NodeId = NewType("NodeId", str)
LinkId = NewType("LinkId", str)
DeviceId = NewType("DeviceId", str)
SessionId = NewType("SessionId", str)
RoundId = NewType("RoundId", str)


def ensure_nonempty_id(value: object, *, field_name: str = "id") -> str:
    """Validate and return a non-empty string identifier.

    Parameters
    ----------
    value : object
        Candidate identifier. The value must be a ``str`` whose stripped form
        is non-empty.
    field_name : str, default="id"
        Name used in validation error messages.

    Returns
    -------
    str
        The original string object. Whitespace is checked but not stripped.

    Raises
    ------
    TypeError
        If `value` is not a string.
    ValueError
        If `value` is empty or contains only whitespace.

    Notes
    -----
    This function intentionally does not normalize identifiers. Preserving the
    input string makes accidental whitespace or casing differences visible to
    the caller instead of silently changing public names.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_optional_correlation_id(
    value: object,
    *,
    field_name: str = "correlation_id",
) -> int | UUID | None:
    """Validate an optional signal-plane correlation identifier.

    Signal records use ``int`` or ``UUID`` correlation IDs. Classical and
    control payloads validate their own ``str`` or ``int`` correlation IDs
    separately.
    """
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    if type(value) is int:
        return value
    raise TypeError(f"{field_name} must be int, UUID, or None")


def as_node_id(value: object) -> NodeId:
    """Validate `value` and return it with the ``NodeId`` type marker.

    Parameters
    ----------
    value : object
        Candidate node identifier.

    Returns
    -------
    NodeId
        Validated identifier string marked as a node identifier for type
        checkers.
    """
    return NodeId(ensure_nonempty_id(value, field_name="node_id"))


def as_link_id(value: object) -> LinkId:
    """Validate `value` and return it with the ``LinkId`` type marker.

    Parameters
    ----------
    value : object
        Candidate link identifier.

    Returns
    -------
    LinkId
        Validated identifier string marked as a link identifier for type
        checkers.
    """
    return LinkId(ensure_nonempty_id(value, field_name="link_id"))


def as_device_id(value: object) -> DeviceId:
    """Validate `value` and return it with the ``DeviceId`` type marker.

    Parameters
    ----------
    value : object
        Candidate device identifier.

    Returns
    -------
    DeviceId
        Validated identifier string marked as a device identifier for type
        checkers.
    """
    return DeviceId(ensure_nonempty_id(value, field_name="device_id"))


def as_session_id(value: object) -> SessionId:
    """Validate `value` and return it with the ``SessionId`` type marker.

    Parameters
    ----------
    value : object
        Candidate session identifier.

    Returns
    -------
    SessionId
        Validated identifier string marked as a session identifier for type
        checkers.
    """
    return SessionId(ensure_nonempty_id(value, field_name="session_id"))


def as_round_id(value: object) -> RoundId:
    """Validate `value` and return it with the ``RoundId`` type marker.

    Parameters
    ----------
    value : object
        Candidate round identifier.

    Returns
    -------
    RoundId
        Validated identifier string marked as a round identifier for type
        checkers.
    """
    return RoundId(ensure_nonempty_id(value, field_name="round_id"))


def build_round_id(session_id: SessionId | str, index: int) -> RoundId:
    """Build a deterministic round identifier from a session and index.

    Parameters
    ----------
    session_id : SessionId or str
        Non-empty session identifier used as the round prefix.
    index : int
        Non-negative round index. ``bool`` values are rejected even though
        ``bool`` is an ``int`` subclass.

    Returns
    -------
    RoundId
        Identifier in ``"{session_id}:round:{index}"`` format.

    Raises
    ------
    TypeError
        If `index` is not an integer, or if `session_id` is not a string.
    ValueError
        If `session_id` is empty or `index` is negative.
    """
    base_session_id = ensure_nonempty_id(session_id, field_name="session_id")
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("index must be int")
    if index < 0:
        raise ValueError("index must be non-negative")
    return RoundId(f"{base_session_id}:round:{index}")


__all__ = [
    "DeviceId",
    "LinkId",
    "NodeId",
    "RoundId",
    "SessionId",
    "as_device_id",
    "as_link_id",
    "as_node_id",
    "as_round_id",
    "as_session_id",
    "build_round_id",
    "ensure_nonempty_id",
]
