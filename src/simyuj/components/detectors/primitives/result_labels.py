"""Small label-extraction helper shared by detector readout paths."""

from __future__ import annotations


def result_label(result: object | None) -> object | None:
    """
    Return the logical label carried by a qstate or readout result.

    Parameters
    ----------
    result : object or None
        Result object produced by qstate measurement or readout logic.

    Returns
    -------
    object or None
        ``None`` for ``None`` input, otherwise the first available
        ``label``, ``outcome_label``, or ``outcome`` attribute. If none of
        those attributes exist, the original object is returned.

    Notes
    -----
    Detector readout and click resolution use this helper to avoid depending on
    one concrete qstate result class. The helper only inspects attributes; it
    does not mutate the result or qstate store. Attribute precedence is
    ``label``, then ``outcome_label``, then ``outcome``; returning the original
    object is intentional duck typing for already-label-like results.
    """

    if result is None:
        return None

    label = getattr(result, "label", None)
    if label is not None:
        return label

    outcome_label = getattr(result, "outcome_label", None)
    if outcome_label is not None:
        return outcome_label

    outcome = getattr(result, "outcome", None)
    if outcome is not None:
        return outcome

    return result


__all__ = [
    "result_label",
]
