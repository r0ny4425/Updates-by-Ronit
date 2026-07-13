"""Route-to-memory requirement helpers.

This module connects generic routes to ``ResourceManager`` reservations without
embedding protocol assumptions. Callers supply a requirements provider function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simyuj.network.routing import Route
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.validation import validate_positive_int

from .manager import ResourceManager
from .reservation import Reservation


@dataclass(frozen=True, slots=True)
class NodeMemoryRequirement:
    """Generic memory-slot demand for one node.

    This record does not describe why the memory is needed. Higher-level user
    code decides whether the demand represents endpoint storage, repeater
    storage, purification workspace, buffering, or something else.

    Parameters
    ----------
    node_id : str
        Node that must provide memory slots.
    requirement : int or Mapping[str, int]
        Positive number of memory slots required at the node, or a mapping from
        device ID to positive counts for specific device targeting.
    """

    node_id: str
    requirement: int | Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_id",
            ensure_nonempty_id(self.node_id, field_name="node_id"),
        )
        if isinstance(self.requirement, int):
            validate_positive_int(self.requirement, field_name="requirement")
        elif isinstance(self.requirement, Mapping):
            if not self.requirement:
                raise ValueError("device mapping cannot be empty")
            for device_id, count in self.requirement.items():
                ensure_nonempty_id(device_id, field_name="device_id")
                validate_positive_int(count, field_name="count")
        else:
            raise TypeError("requirement must be int or Mapping[str, int]")


def route_memory_requirements(
    route: Route,
    *,
    node_requirements: Callable[[str, int, int], int | Mapping[str, int]],
) -> tuple[NodeMemoryRequirement, ...]:
    """Build deterministic per-node memory requirements for a route.

    The ``node_requirements`` callable is invoked for every node in the route
    with signature ``(node_id, index_in_route, route_length)``. It should return
    an integer count of generic memory slots required, or a mapping of
    device IDs to counts for targeted reservations. Return 0 or an empty mapping
    if no memory is required for that step.

    Requirements are aggregated per node and returned sorted by node ID. If a route
    repeats a node, its requirements are combined (ints sum with ints, dicts merge
    with dicts; mixed types raise ValueError).

    Parameters
    ----------
    route : Route
        Route whose node sequence is converted to requirements.
    node_requirements : Callable
        Function providing the memory requirements per node visit.

    Returns
    -------
    tuple[NodeMemoryRequirement, ...]
        Non-zero requirements sorted by node ID.
    """
    resolved_route = _require_route(route)
    node_ids = resolved_route.node_ids
    route_len = len(node_ids)

    counts: dict[str, int | dict[str, int]] = {}

    for idx, node_id in enumerate(node_ids):
        req = node_requirements(node_id, idx, route_len)
        if isinstance(req, int) and req == 0:
            continue
        if isinstance(req, Mapping) and not req:
            continue

        resolved_node_id = ensure_nonempty_id(node_id, field_name="node_id")
        existing = counts.get(resolved_node_id)

        if existing is None:
            counts[resolved_node_id] = req if isinstance(req, int) else dict(req)
            continue

        if isinstance(existing, int) and isinstance(req, int):
            counts[resolved_node_id] = existing + req
        elif isinstance(existing, dict) and isinstance(req, Mapping):
            for dev, count in req.items():
                existing[dev] = existing.get(dev, 0) + count
        else:
            raise ValueError(
                "cannot merge integer requirement with device mapping "
                f"for node '{resolved_node_id}'"
            )

    return tuple(
        NodeMemoryRequirement(node_id=node_id, requirement=counts[node_id])
        for node_id in sorted(counts)
    )


def requirements_mapping(
    requirements: tuple[NodeMemoryRequirement, ...],
) -> dict[str, int | Mapping[str, int]]:
    """Convert node memory requirements into a ``node_id -> requirement`` mapping.

    Parameters
    ----------
    requirements : tuple[NodeMemoryRequirement, ...]
        Requirements with unique node IDs.

    Returns
    -------
    dict[str, int | Mapping[str, int]]
        Mapping suitable for ``ResourceManager.reserve_memories``.
    """
    resolved_requirements = _require_requirements(requirements)
    return {
        requirement.node_id: requirement.requirement
        for requirement in resolved_requirements
    }


def reserve_route_memories(
    now: int,
    manager: ResourceManager,
    route: Route,
    *,
    node_requirements: Callable[[str, int, int], int | Mapping[str, int]],
    owner: str,
    reservation_id: str | None = None,
    created_at: int | None = None,
    expires_at: int | None = None,
    metadata: tuple[tuple[str, object], ...] = (),
) -> Reservation:
    """Reserve memory slots for a route using a caller-provided requirement provider.

    Per-node requirements are aggregated and then reserved through
    ``ResourceManager.reserve_memories``.

    Parameters
    ----------
    now : int
        Current timeline tick used to evaluate physical hardware readiness.
    manager : ResourceManager
        Manager used to reserve the selected memory slots.
    route : Route
        Route whose nodes define where memory is required.
    node_requirements : Callable
        Function providing the memory requirements per node visit.
    owner : str
        Reservation owner.
    reservation_id : str or None, optional
        Explicit reservation ID, or ``None`` for generated IDs.
    created_at, expires_at : int or None, optional
        Optional non-negative reservation times.
    metadata : tuple[tuple[str, object], ...], optional
        Metadata attached to the reservation.

    Returns
    -------
    Reservation
        Reservation returned by ``manager.reserve_memories``.

    Raises
    ------
    ValueError
        If the route/count combination produces no memory requirements.
    """
    if not isinstance(manager, ResourceManager):
        raise TypeError("manager must be ResourceManager")

    requirements = route_memory_requirements(
        route,
        node_requirements=node_requirements,
    )

    if not requirements:
        raise ValueError("route memory requirements are empty")

    return manager.reserve_memories(
        now,
        requirements_mapping(requirements),
        owner=owner,
        reservation_id=reservation_id,
        created_at=created_at,
        expires_at=expires_at,
        metadata=metadata,
    )


def _require_route(route: Route) -> Route:
    if not isinstance(route, Route):
        raise TypeError("route must be Route")
    return route


def _require_requirements(
    requirements: tuple[NodeMemoryRequirement, ...],
) -> tuple[NodeMemoryRequirement, ...]:
    if not isinstance(requirements, tuple):
        raise TypeError("requirements must be tuple[NodeMemoryRequirement, ...]")

    seen_node_ids: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, NodeMemoryRequirement):
            raise TypeError(
                "requirements must contain only NodeMemoryRequirement entries"
            )
        if requirement.node_id in seen_node_ids:
            raise ValueError(f"duplicate node requirement '{requirement.node_id}'")
        seen_node_ids.add(requirement.node_id)

    return requirements


__all__ = [
    "NodeMemoryRequirement",
    "requirements_mapping",
    "reserve_route_memories",
    "route_memory_requirements",
]
