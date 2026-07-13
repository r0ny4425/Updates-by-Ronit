"""
Immutable metadata primitives shared across simulation subsystems.

Metadata is represented as ``tuple[tuple[str, object], ...]`` so payload and
record dataclasses can remain frozen and hashable when their values are
hashable. The module validates that shape and converts mapping or iterable
inputs into the tuple form used by the rest of the package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

Meta: TypeAlias = tuple[tuple[str, object], ...]
MetaInput: TypeAlias = Mapping[str, object] | Iterable[tuple[str, object]] | None


def validate_meta(
    meta: Meta,
    *,
    field_name: str = "meta",
    require_hashable: bool = True,
) -> None:
    """Validate a tuple of metadata key/value pairs.

    Parameters
    ----------
    meta : Meta
        Metadata tuple. Each item must be a two-item tuple whose first value is
        a string key.
    field_name : str, default="meta"
        Name used in validation error messages.
    require_hashable : bool, default=True
        When ``True``, each metadata value must be hashable. Set this to
        ``False`` for records that need immutable pair structure but allow
        opaque, unhashable values.

    Raises
    ------
    TypeError
        If `meta` is not a tuple of ``(str, object)`` pairs, or if
        `require_hashable` is true and a value is unhashable.
    """
    if not isinstance(meta, tuple):
        raise TypeError(f"{field_name} must be tuple[tuple[str, object], ...]")
    for item in meta:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{field_name} must contain (key, value) pairs")
        key, value = item
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be str")
        if require_hashable:
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError(f"{field_name} values must be hashable") from exc


def freeze_meta(
    meta: MetaInput,
    *,
    field_name: str = "meta",
    require_hashable: bool = True,
) -> Meta:
    """Convert metadata input into the package tuple representation.

    Parameters
    ----------
    meta : MetaInput
        ``None``, a mapping, or an iterable of ``(key, value)`` pairs. ``None``
        becomes the empty metadata tuple.
    field_name : str, default="meta"
        Name used in validation error messages.
    require_hashable : bool, default=True
        Whether metadata values must be hashable after conversion.

    Returns
    -------
    Meta
        Tuple of ``(key, value)`` pairs in the input iteration order.

    Raises
    ------
    TypeError
        If the converted value does not satisfy :func:`validate_meta`.

    Notes
    -----
    Mapping inputs preserve the mapping's iteration order. The implementation
    does not sort, deduplicate, or normalize keys. Conversion is shallow:
    metadata values are preserved, not recursively frozen.
    """
    if meta is None:
        return ()
    if isinstance(meta, Mapping):
        items = tuple(meta.items())
    else:
        items = tuple(meta)
    frozen_items = tuple((key, value) for key, value in items)
    validate_meta(
        frozen_items,
        field_name=field_name,
        require_hashable=require_hashable,
    )
    return frozen_items


__all__ = [
    "Meta",
    "MetaInput",
    "freeze_meta",
    "validate_meta",
]
