"""Controller-side scheduling helpers for quantum memory components.

``MemoryService`` lets a node-local agent schedule existing ``QuantumMemory``
actions without calling the memory component directly. The service resolves a
node alias to a physical memory component, builds the corresponding request
record, and schedules an event at the current timeline tick.
"""

from __future__ import annotations

from simyuj.components.memories import (
    MEMORY_ABSORB,
    MEMORY_APPLY_OPERATOR,
    MEMORY_DISCARD,
    MEMORY_EMIT,
    MEMORY_MEASURE,
    MEMORY_UPDATE_META,
    MemoryAbsorbRequest,
    MemoryApplyOperatorRequest,
    MemoryDiscardRequest,
    MemoryEmitRequest,
    MemoryMeasureRequest,
    MemoryUpdateMetaRequest,
    QuantumMemory,
)
from simyuj.engine.component import Component
from simyuj.engine.event import Event
from simyuj.engine.timeline import Timeline
from simyuj.network import Node
from simyuj.primitives.ids import ensure_nonempty_id
from simyuj.primitives.meta import Meta
from simyuj.signal import Signal


class MemoryService:
    """Schedule existing ``QuantumMemory`` actions for a node-local agent.

    Parameters
    ----------
    node : Node
        Node whose device aliases are used to resolve memory names.
    timeline : Timeline
        Timeline used for all scheduled memory requests.
    session_id : str
        Non-empty session id copied into request payloads and event metadata.
    source_agent : Component
        Agent component used as the scheduled event source. It must expose a
        non-empty ``agent_id`` attribute.
    priority : int, default=0
        Event priority used for memory requests scheduled by this service.

    Notes
    -----
    Public methods schedule events; they do not call
    ``QuantumMemory.handle_event``. Position, operator, measurement, and signal
    validation remain owned by the memory request records and memory component.
    Requests are scheduled at ``timeline.current_time``. If a service method is
    called while an event is being handled, same-tick work still flows through
    the timeline's normal batching rules rather than running immediately.
    Auto-generated request ids include ``timeline.events_scheduled``; they are
    deterministic for a fixed run but are not durable protocol identifiers.
    Pass ``request_id`` when a stable semantic id is required.
    """

    def __init__(
        self,
        *,
        node: Node,
        timeline: Timeline,
        session_id: str,
        source_agent: Component,
        priority: int = 0,
    ) -> None:
        if not isinstance(node, Node):
            raise TypeError("node must be Node")
        if not isinstance(timeline, Timeline):
            raise TypeError("timeline must be Timeline")
        if not isinstance(source_agent, Component):
            raise TypeError("source_agent must be Component")
        agent_id = getattr(source_agent, "agent_id", None)
        if not isinstance(agent_id, str) or agent_id == "":
            raise ValueError("source_agent must expose non-empty agent_id")
        if type(priority) is not int:
            raise TypeError("priority must be int")

        self._node = node
        self._timeline = timeline
        self._session_id = ensure_nonempty_id(session_id, field_name="session_id")
        self._source_agent = source_agent
        self._source_agent_id = agent_id
        self._priority = priority

    def absorb(
        self,
        memory: str,
        signal: Signal,
        *,
        position: int | None = None,
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a memory absorb request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        signal : Signal
            Signal to absorb.
        position : int or None, default=None
            Optional target memory position. ``None`` lets the memory choose.
        request_id : str or None, default=None
            Optional non-empty request id. When omitted, the service builds one
            from session, agent, memory, action, and scheduled-event count.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryAbsorbRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_ABSORB`` event targeted to the resolved memory.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("absorb", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_ABSORB,
            payload=MemoryAbsorbRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                signal=signal,
                position=position,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def emit(
        self,
        memory: str,
        position: int,
        *,
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a memory emit request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        position : int
            Memory position to emit.
        request_id : str or None, default=None
            Optional non-empty request id.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryEmitRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_EMIT`` event targeted to the resolved memory.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("emit", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_EMIT,
            payload=MemoryEmitRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                position=position,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def measure(
        self,
        memory: str,
        positions: tuple[int, ...],
        *,
        measurement: object = "z",
        collapse: bool = True,
        destructive: bool = True,
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a memory measurement request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        positions : tuple[int, ...]
            Memory positions to measure.
        measurement : object, default="z"
            Measurement basis or object accepted by ``QuantumMemory``.
        collapse : bool, default=True
            Whether the measurement request should collapse measured state.
        destructive : bool, default=True
            Whether measured positions should be cleared by the memory.
        request_id : str or None, default=None
            Optional non-empty request id.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryMeasureRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_MEASURE`` event targeted to the resolved memory.

        Notes
        -----
        The service passes ``measurement`` through unchanged; it does not check
        basis names, operators, or compatibility with the selected positions.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("measure", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_MEASURE,
            payload=MemoryMeasureRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                positions=positions,
                measurement=measurement,
                collapse=collapse,
                destructive=destructive,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def apply(
        self,
        memory: str,
        positions: tuple[int, ...],
        operator: object,
        *,
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule an operator application request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        positions : tuple[int, ...]
            Memory positions the operator should act on.
        operator : object
            Operator object accepted by ``QuantumMemory``.
        request_id : str or None, default=None
            Optional non-empty request id.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryApplyOperatorRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_APPLY_OPERATOR`` event targeted to the resolved
            memory.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("apply", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_APPLY_OPERATOR,
            payload=MemoryApplyOperatorRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                positions=positions,
                operator=operator,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def discard(
        self,
        memory: str,
        position: int,
        *,
        reason: str = "discarded",
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a memory discard request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        position : int
            Memory position to discard.
        reason : str, default="discarded"
            Reason copied into the discard request.
        request_id : str or None, default=None
            Optional non-empty request id.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryDiscardRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_DISCARD`` event targeted to the resolved memory.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("discard", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_DISCARD,
            payload=MemoryDiscardRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                position=position,
                reason=reason,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def update_meta(
        self,
        memory: str,
        position: int,
        *,
        updates: Meta = (),
        remove_keys: tuple[str, ...] = (),
        expected_occupancy_token: int | None = None,
        request_id: str | None = None,
        meta: Meta = (),
    ) -> Event:
        """Schedule a memory-slot metadata update request.

        Parameters
        ----------
        memory : str
            Node-local alias for a ``QuantumMemory`` device.
        position : int
            Memory position whose metadata should change.
        updates : tuple[tuple[str, object], ...], default=()
            Metadata entries to add or replace.
        remove_keys : tuple[str, ...], default=()
            Metadata keys to remove.
        expected_occupancy_token : int or None, default=None
            Optional optimistic token checked by the memory component.
        request_id : str or None, default=None
            Optional non-empty request id.
        meta : tuple[tuple[str, object], ...], default=()
            Request metadata passed to ``MemoryUpdateMetaRequest``.

        Returns
        -------
        Event
            Scheduled ``MEMORY_UPDATE_META`` event targeted to the resolved
            memory.
        """
        qmem = self._memory(memory)
        resolved_request_id = self._request_id("update_meta", qmem, request_id)
        return self._schedule(
            qmem=qmem,
            action=MEMORY_UPDATE_META,
            payload=MemoryUpdateMetaRequest(
                request_id=resolved_request_id,
                memory_id=qmem.memory_id,
                position=position,
                updates=updates,
                remove_keys=remove_keys,
                expected_occupancy_token=expected_occupancy_token,
                session_id=self._session_id,
                meta=meta,
            ),
            request_id=resolved_request_id,
            memory_alias=memory,
        )

    def _memory(self, name: str) -> QuantumMemory:
        device = self._node.get_device(name)
        if not isinstance(device, QuantumMemory):
            raise TypeError(f"device {name!r} is not QuantumMemory")
        return device

    def _request_id(
        self,
        kind: str,
        qmem: QuantumMemory,
        request_id: str | None,
    ) -> str:
        if request_id is not None:
            return ensure_nonempty_id(request_id, field_name="request_id")
        return (
            f"{self._session_id}:{self._source_agent_id}:"
            f"{qmem.memory_id}:{kind}:{self._timeline.events_scheduled}"
        )

    def _schedule(
        self,
        *,
        qmem: QuantumMemory,
        action: str,
        payload: object,
        request_id: str,
        memory_alias: str,
    ) -> Event:
        return self._timeline.schedule(
            Event(
                time=self._timeline.current_time,
                priority=self._priority,
                target_ref=qmem,
                action=action,
                payload_ref=payload,
                source=self._source_agent,
                subsystem_id="control",
                meta={
                    "session_id": self._session_id,
                    "agent_id": self._source_agent_id,
                    "memory_alias": memory_alias,
                    "memory_id": qmem.memory_id,
                    "request_id": request_id,
                },
            )
        )


__all__ = ["MemoryService"]
