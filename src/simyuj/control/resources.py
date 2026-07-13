"""Resource-manager access for control-plane agents.

``ResourceService`` delegates protocol-neutral memory-slot bookkeeping to a
``ResourceManager`` while automatically using the owning agent id as the
reservation owner. It does not schedule quantum-memory component events.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from simyuj.components.memories import MemoryAbsorbReport
from simyuj.network.routing import Route
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.resources.manager import ResourceManager
from simyuj.resources.memory import MemoryRef, MemorySlotView
from simyuj.resources.reservation import Reservation
from simyuj.resources.route_requirements import reserve_route_memories


class ResourceService:
    """Delegate protocol-neutral memory resource bookkeeping.

    Parameters
    ----------
    manager : ResourceManager
        Resource manager that owns memory-slot and reservation state.
    owner_agent_id : str
        Non-empty agent identifier used as the owner for new reservations.

    Notes
    -----
    Lifecycle methods update ``ResourceManager`` bookkeeping only. They do not
    update physical ``QuantumMemory`` state or schedule memory component events.
    """

    def __init__(self, manager: ResourceManager, *, owner_agent_id: str) -> None:
        if not isinstance(manager, ResourceManager):
            raise TypeError("manager must be ResourceManager")
        self._manager = manager
        self._owner_agent_id = ensure_nonempty_id(
            owner_agent_id,
            field_name="owner_agent_id",
        )

    def get_slot(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Return the current slot view for ``memory_ref``."""
        return self._manager.get_slot(memory_ref)

    def get_reservation(self, reservation_id: str) -> Reservation:
        """Return a reservation by ID."""
        return self._manager.get_reservation(reservation_id)

    def reservation_for_memory(
        self,
        memory_ref: MemoryRef,
    ) -> Reservation | None:
        """Return the reservation currently holding ``memory_ref``, if any."""
        return self._manager.reservation_for_memory(memory_ref)

    def registered_memories(
        self,
        node_id: str | None = None,
        *,
        device_id: str | None = None,
    ) -> tuple[MemoryRef, ...]:
        """Return registered memory refs filtered by optional node or device."""
        return self._manager.registered_memories(node_id, device_id=device_id)

    def available_memories(
        self,
        now: int,
        node_id: str | None = None,
        *,
        device_id: str | None = None,
        link_id: str | None = None,
    ) -> tuple[MemoryRef, ...]:
        """Return physically free memory references from the resource manager.

        Parameters
        ----------
        now : int
            Current timeline tick used to evaluate physical hardware readiness.
        node_id : str or None, default=None
            Optional node filter.
        device_id : str or None, default=None
            Optional device filter within the selected node.
        link_id : str or None, default=None
            Optional link identifier for filtering slots by metadata.
        """
        return self._manager.available_memories(
            now, node_id, device_id=device_id, link_id=link_id
        )

    def reserve_memories(
        self,
        now: int,
        requirements: Mapping[str, int | Mapping[str, int]],
        *,
        reservation_id: str | None = None,
        created_at: int | None = None,
        expires_at: int | None = None,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> Reservation:
        """Reserve memory slots using this agent as owner.

        Parameters
        ----------
        now : int
            Current timeline tick used to evaluate physical hardware readiness.
        requirements : Mapping[str, int | Mapping[str, int]]
            Number of memory slots required per node id, or per-device counts
            for one node.
        reservation_id : str or None, default=None
            Optional explicit reservation id.
        created_at : int or None, default=None
            Optional creation tick stored by the resource manager.
        expires_at : int or None, default=None
            Optional expiration tick stored by the resource manager.
        metadata : tuple[tuple[str, object], ...], default=()
            Reservation metadata passed through to the resource manager.

        Returns
        -------
        Reservation
            Reservation created by ``ResourceManager.reserve_memories``.
        """
        return self._manager.reserve_memories(
            now,
            requirements,
            owner=self._owner_agent_id,
            reservation_id=reservation_id,
            created_at=created_at,
            expires_at=expires_at,
            metadata=metadata,
        )

    def reserve_for_route(
        self,
        now: int,
        route: Route,
        *,
        node_requirements: Callable[[str, int, int], int | Mapping[str, int]],
        reservation_id: str | None = None,
        created_at: int | None = None,
        expires_at: int | None = None,
        metadata: tuple[tuple[str, object], ...] = (),
    ) -> Reservation:
        """Reserve route endpoint and intermediate memories for this agent.

        Parameters
        ----------
        now : int
            Current timeline tick used to evaluate physical hardware readiness.
        route : Route
            Network route whose endpoints and intermediate nodes need memory.
        node_requirements : Callable
            Function providing the memory requirements per node visit.
        reservation_id : str or None, default=None
            Optional explicit reservation id.
        created_at : int or None, default=None
            Optional creation tick stored by the resource manager.
        expires_at : int or None, default=None
            Optional expiration tick stored by the resource manager.
        metadata : tuple[tuple[str, object], ...], default=()
            Reservation metadata passed through to the route helper.

        Returns
        -------
        Reservation
            Reservation returned by ``reserve_route_memories``.
        """
        return reserve_route_memories(
            now,
            self._manager,
            route,
            node_requirements=node_requirements,
            owner=self._owner_agent_id,
            reservation_id=reservation_id,
            created_at=created_at,
            expires_at=expires_at,
            metadata=metadata,
        )

    def commit(self, reservation_id: str) -> Reservation:
        """Commit a reservation by id."""
        return self._manager.commit_reservation(
            reservation_id,
            owner=self._owner_agent_id,
        )

    def release(self, reservation_id: str) -> Reservation:
        """Release a reservation by id."""
        return self._manager.release_reservation(
            reservation_id,
            owner=self._owner_agent_id,
        )

    def expire_before(self, now: int) -> tuple[Reservation, ...]:
        """Expire reservations that have reached their expiration tick."""
        return self._manager.expire_before(now)

    def mark_occupied(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a memory slot occupied."""
        return self._manager.mark_occupied(memory_ref)

    def mark_absorb_report(
        self,
        report: MemoryAbsorbReport,
        memory_ref: MemoryRef,
    ) -> MemorySlotView:
        """Mark resource state from a successful memory absorb report.

        Rejects failed reports, mismatched positions, and mismatched physical
        memory IDs when the resource slot carries ``memory_id`` metadata.
        """
        if not isinstance(report, MemoryAbsorbReport):
            raise TypeError("report must be MemoryAbsorbReport")
        if not isinstance(memory_ref, MemoryRef):
            raise TypeError("memory_ref must be MemoryRef")

        if not report.success:
            raise ValueError("cannot mark resource from failed absorb report")
        if report.position != memory_ref.position:
            raise ValueError("report position does not match memory_ref")

        slot = self._manager.get_slot(memory_ref)
        for key, value in slot.metadata:
            if key == "memory_id" and value != report.memory_id:
                raise ValueError("report memory_id does not match memory_ref")

        return self.mark_occupied(memory_ref)

    def mark_consumed(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a memory slot consumed."""
        return self._manager.mark_consumed(memory_ref)

    def mark_expired(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a memory slot expired."""
        return self._manager.mark_expired(memory_ref)

    def mark_failed(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a memory slot failed."""
        return self._manager.mark_failed(memory_ref)

    def mark_free(self, memory_ref: MemoryRef) -> MemorySlotView:
        """Mark a memory slot free."""
        return self._manager.mark_free(memory_ref)


__all__ = ["ResourceService"]
