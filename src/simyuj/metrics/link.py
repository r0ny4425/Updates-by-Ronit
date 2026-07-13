"""Per-link metric accessors for topology and routing helpers.

The functions in this module validate caller-provided mappings keyed by link
ID.  They do not compute physical link models; callers decide whether a metric
means cost, delay, loss budget, distance, or another non-negative quantity.
"""

from __future__ import annotations

from collections.abc import Mapping

from simyuj.network.topology import TopologyEdge
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import require_non_negative_real, require_probability


def link_metric(
    link_id: str,
    values: Mapping[str, float],
    *,
    field_name: str = "metric",
    default: float | None = None,
) -> float:
    """Return a finite non-negative metric value for one link ID.

    Parameters
    ----------
    link_id : str
        Link identifier to look up.
    values : Mapping[str, float]
        Mapping from link ID to finite non-negative metric value.
    field_name : str, optional
        Name used in validation and missing-value error messages.
    default : float or None, optional
        Non-negative value returned when ``link_id`` is absent. If ``None``,
        missing values raise ``KeyError``.

    Returns
    -------
    float
        Validated non-negative metric value.

    Raises
    ------
    TypeError
        If ``values`` is not a mapping or a value has an unsupported type.
    ValueError
        If ``link_id`` is empty, ``default`` is negative, or a mapped value is
        negative or non-finite.
    KeyError
        If the link is absent and no default is provided.

    Notes
    -----
    ``default`` is a per-lookup fallback for a missing link ID. It does not fill
    or validate the mapping ahead of time; route helpers apply it independently
    for each missing route link.

    Examples
    --------
    >>> delay_by_link = {"l_ab": 10.0}
    >>> link_metric("l_ab", delay_by_link, field_name="delay")
    10.0
    >>> link_metric("l_missing", delay_by_link, default=2.5)
    2.5
    """

    resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")
    resolved_default = _resolve_optional_non_negative_default(
        default,
        field_name=field_name,
    )

    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping of link_id to metric value")

    if resolved_link_id not in values:
        if resolved_default is None:
            raise KeyError(f"missing {field_name} for link id '{resolved_link_id}'")
        return resolved_default

    return require_non_negative_real(
        values[resolved_link_id],
        field_name=field_name,
    )


def edge_metric(
    edge: TopologyEdge,
    values: Mapping[str, float],
    *,
    field_name: str = "metric",
    default: float | None = None,
) -> float:
    """Return a finite non-negative metric value for one topology edge.

    Parameters
    ----------
    edge : TopologyEdge
        Edge whose ``link_id`` is used for lookup.
    values : Mapping[str, float]
        Mapping from link ID to finite non-negative metric value.
    field_name : str, optional
        Name used in validation and missing-value error messages.
    default : float or None, optional
        Non-negative fallback when the edge link ID is absent.

    Returns
    -------
    float
        Validated metric value for ``edge.link_id``.

    Notes
    -----
    Only ``edge.link_id`` is used for lookup. Edge direction, endpoint IDs, port
    kind, and topology state are not inspected.
    """

    if not isinstance(edge, TopologyEdge):
        raise TypeError("edge must be TopologyEdge")

    return link_metric(
        edge.link_id,
        values,
        field_name=field_name,
        default=default,
    )


def link_success_probability(
    link_id: str,
    probabilities: Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Return a success probability value for one link ID.

    Parameters
    ----------
    link_id : str
        Link identifier to look up.
    probabilities : Mapping[str, float]
        Mapping from link ID to probability in ``[0, 1]``.
    default : float or None, optional
        Probability returned when ``link_id`` is absent. If ``None``, missing
        links raise ``KeyError``.

    Returns
    -------
    float
        Validated probability in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``probabilities`` is not a mapping or a value has an unsupported
        type.
    ValueError
        If identifiers are empty or a probability is outside ``[0, 1]``.
    KeyError
        If the link is absent and no default is provided.

    Notes
    -----
    ``default`` is a per-lookup fallback for a missing link ID. It does not fill
    or validate the mapping ahead of time; route helpers apply it independently
    for each missing route link.

    Examples
    --------
    >>> link_success_probability("l_ab", {"l_ab": 0.75})
    0.75
    """

    resolved_link_id = ensure_nonempty_id(link_id, field_name="link_id")
    resolved_default = _resolve_optional_probability_default(default)

    if not isinstance(probabilities, Mapping):
        raise TypeError(
            "probabilities must be a mapping of link_id to probability value"
        )

    if resolved_link_id not in probabilities:
        if resolved_default is None:
            raise KeyError(
                f"missing success probability for link id '{resolved_link_id}'"
            )
        return resolved_default

    return require_probability(
        probabilities[resolved_link_id],
        field_name="success_probability",
    )


def edge_success_probability(
    edge: TopologyEdge,
    probabilities: Mapping[str, float],
    *,
    default: float | None = None,
) -> float:
    """Return a success probability value for one topology edge.

    Parameters
    ----------
    edge : TopologyEdge
        Edge whose ``link_id`` is used for lookup.
    probabilities : Mapping[str, float]
        Mapping from link ID to probability in ``[0, 1]``.
    default : float or None, optional
        Probability fallback when the edge link ID is absent.

    Returns
    -------
    float
        Validated probability for ``edge.link_id``.

    Notes
    -----
    Only ``edge.link_id`` is used for lookup. Edge direction, endpoint IDs, port
    kind, and topology state are not inspected.
    """

    if not isinstance(edge, TopologyEdge):
        raise TypeError("edge must be TopologyEdge")

    return link_success_probability(
        edge.link_id,
        probabilities,
        default=default,
    )


def _resolve_optional_non_negative_default(
    default: float | None,
    *,
    field_name: str,
) -> float | None:
    if default is None:
        return None

    return require_non_negative_real(default, field_name=f"default_{field_name}")


def _resolve_optional_probability_default(default: float | None) -> float | None:
    if default is None:
        return None

    return require_probability(
        default,
        field_name="default_success_probability",
    )


__all__ = [
    "edge_metric",
    "edge_success_probability",
    "link_metric",
    "link_success_probability",
]
