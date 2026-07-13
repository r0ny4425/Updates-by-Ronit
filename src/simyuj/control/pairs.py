"""Entangled-pair registry access for control-plane agents.

``PairService`` exposes protocol-neutral pair registration, lookup, and
lifecycle operations from an ``EntangledPairRegistry``. It reads the runtime
timeline only when expiring records relative to the current simulation tick.
"""

from __future__ import annotations

from simyuj.engine.timeline import Timeline
from simyuj.entanglement.pair import EntangledPairRecord, PairState
from simyuj.entanglement.queries import (
    RouteHopPairs,
    available_pairs_for_route_hops,
    route_hops_have_available_pairs,
)
from simyuj.entanglement.registry import EntangledPairRegistry
from simyuj.network.routing import Route
from simyuj.resources.memory import MemoryRef


class PairService:
    """Delegate protocol-neutral entangled-pair lifecycle bookkeeping.

    Parameters
    ----------
    registry : EntangledPairRegistry
        Registry that owns pair records and lifecycle state.
    timeline : Timeline
        Runtime timeline used by ``expire_before_now``.
    """

    def __init__(
        self,
        registry: EntangledPairRegistry,
        *,
        timeline: Timeline,
    ) -> None:
        if not isinstance(registry, EntangledPairRegistry):
            raise TypeError("registry must be EntangledPairRegistry")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        self._registry = registry
        self._timeline = timeline

    def get(self, pair_id: str) -> EntangledPairRecord:
        """Return the pair record for ``pair_id``."""
        return self._registry.get(pair_id)

    def all_pairs(
        self,
        *,
        state: PairState | None = None,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return pair records in deterministic pair-id order."""
        return self._registry.all_pairs(state=state)

    def active_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return available and reserved pairs in deterministic pair-id order."""
        return self._registry.active_pairs()

    def available_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs in deterministic pair-id order."""
        return self._registry.available_pairs()

    def reserved_pairs(self) -> tuple[EntangledPairRecord, ...]:
        """Return reserved pairs in deterministic pair-id order."""
        return self._registry.reserved_pairs()

    def available_for_memory_refs(
        self,
        first: MemoryRef,
        second: MemoryRef,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs connecting two exact memory positions."""
        return self._registry.available_for_memory_refs(first, second)

    def pair_using_memory(
        self,
        memory_ref: MemoryRef,
    ) -> EntangledPairRecord | None:
        """Return the active pair using ``memory_ref``, if any."""
        return self._registry.pair_using_memory(memory_ref)

    def pairs_using_memory(
        self,
        memory_ref: MemoryRef,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return all historical records using ``memory_ref``."""
        return self._registry.pairs_using_memory(memory_ref)

    def register(self, pair: EntangledPairRecord) -> EntangledPairRecord:
        """Register a new entangled pair record.

        Delegates validation and conflict checks to the underlying registry.
        """
        return self._registry.register(pair)

    def available_between(
        self,
        first_node_id: str,
        second_node_id: str,
        *,
        min_fidelity: float | None = None,
        link_id: str | None = None,
    ) -> tuple[EntangledPairRecord, ...]:
        """Return available pairs between two nodes.

        Parameters
        ----------
        first_node_id, second_node_id : str
            Node ids for the pair endpoints.
        min_fidelity : float or None, default=None
            Optional lower fidelity bound delegated to the registry.
        link_id : str or None, default=None
            Optional identifier for the generation link.
        """
        return self._registry.available_between(
            first_node_id,
            second_node_id,
            min_fidelity=min_fidelity,
            link_id=link_id,
        )

    def route_hop_candidates(
        self,
        route: Route,
        *,
        min_fidelity: float | None = None,
    ) -> tuple[RouteHopPairs, ...]:
        """Return available pair candidates for each hop in a route."""
        return available_pairs_for_route_hops(
            self._registry,
            route,
            min_fidelity=min_fidelity,
        )

    def route_hops_ready(
        self,
        route: Route,
        *,
        min_fidelity: float | None = None,
    ) -> bool:
        """Return whether every hop in ``route`` has an available pair."""
        return route_hops_have_available_pairs(
            self._registry,
            route,
            min_fidelity=min_fidelity,
        )

    def reserve(self, pair_id: str) -> EntangledPairRecord:
        """Reserve an available entangled pair."""
        return self._registry.reserve(pair_id)

    def release(self, pair_id: str) -> EntangledPairRecord:
        """Release a reserved entangled pair back to availability."""
        return self._registry.release(pair_id)

    def consume(self, pair_id: str) -> EntangledPairRecord:
        """Mark an entangled pair consumed."""
        return self._registry.consume(pair_id)

    def expire(self, pair_id: str) -> EntangledPairRecord:
        """Mark an entangled pair expired."""
        return self._registry.expire(pair_id)

    def fail(self, pair_id: str) -> EntangledPairRecord:
        """Mark an entangled pair failed."""
        return self._registry.fail(pair_id)

    def expire_before_now(self) -> tuple[EntangledPairRecord, ...]:
        """Expire pairs whose expiration tick is before current timeline time."""
        return self._registry.expire_before(self._timeline.current_time)


__all__ = ["PairService"]
