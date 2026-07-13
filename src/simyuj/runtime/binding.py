"""Pre-run binding lifecycle for deterministic runtime setup.

Binding gives simulation objects a phase to declare timeline-owned resources
before event execution begins. In particular, objects should request
deterministic RNG streams here rather than during ``handle_event`` after the
timeline has frozen stream creation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from simyuj.engine.timeline import Timeline
    from simyuj.tracing.logger import SimulationLogger

BindingMeta = tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class BindingContext:
    """Immutable context passed to optional pre-run ``bind()`` hooks.

    The binding phase is intentionally explicit and should run before the first
    ``Timeline.run_one_step()`` / ``Timeline.run_until(...)`` call, so
    stochastic entities can declare deterministic RNG streams before freeze.

    Notes
    -----
    The context carries the active timeline, logger, optional entity and
    component identifiers, and caller-supplied metadata.

    ``BindingContext`` does not validate the concrete timeline or logger types
    at construction time; callers and bind hooks are responsible for passing
    compatible objects.

    The metadata tuple container is treated as immutable; values are passed
    through unchanged.
    """

    timeline: "Timeline"
    logger: "SimulationLogger"
    entity_id: str | None = None
    component_id: str | None = None
    meta: BindingMeta = field(default_factory=tuple)


@runtime_checkable
class SupportsBind(Protocol):
    """Protocol for entities that expose a pre-run binding hook."""

    def bind(self, context: BindingContext) -> None:
        """Declare deterministic runtime resources before execution starts."""
        ...


class BindableMixin:
    """Optional no-op mixin for entities that want a ``bind()`` slot."""

    def bind(self, context: BindingContext) -> None:
        """Default bind hook; subclasses may override."""


def bind_if_supported(
    entity: object,
    timeline: "Timeline",
    logger: "SimulationLogger | None" = None,
    *,
    entity_id: str | None = None,
    component_id: str | None = None,
    meta: BindingMeta = (),
) -> bool:
    """Bind ``entity`` when it exposes a callable ``bind(context)`` method.

    Parameters
    ----------
    entity : object
        Candidate object. If it has no ``bind`` attribute, no action is taken.
    timeline : Timeline
        Timeline passed to the binding context.
    logger : SimulationLogger or None, optional
        Logger passed to the binding context. When omitted,
        ``timeline.logger`` is used.
    entity_id : str or None, optional
        Optional higher-level entity identifier for the context.
    component_id : str or None, optional
        Optional component identifier for the context.
    meta : BindingMeta, optional
        Caller-supplied metadata entries. The tuple container is treated as
        immutable; values are passed through unchanged.

    Returns
    -------
    bool
        ``True`` if binding was performed; ``False`` when the entity has no
        bind hook.

    Raises
    ------
    TypeError
        If ``entity`` defines a non-callable ``bind`` attribute.

    Examples
    --------
    >>> from simyuj.engine.timeline import Timeline
    >>> class UsesStream:
    ...     def bind(self, context):
    ...         self.rng = context.timeline.rng("example", "stream")
    >>> entity = UsesStream()
    >>> bind_if_supported(entity, Timeline(master_seed=1))
    True
    """
    bind_method = getattr(entity, "bind", None)
    if bind_method is None:
        return False

    if not callable(bind_method):
        raise TypeError(f"Entity {entity!r} defines a non-callable 'bind' attribute.")

    context = BindingContext(
        timeline=timeline,
        logger=timeline.logger if logger is None else logger,
        entity_id=entity_id,
        component_id=component_id,
        meta=meta,
    )
    bind_method(context)
    return True


def bind_many(
    entities: Iterable[object],
    timeline: "Timeline",
    logger: "SimulationLogger | None" = None,
) -> tuple[object, ...]:
    """Bind all bind-capable entities in iteration order.

    Parameters
    ----------
    entities : Iterable[object]
        Objects to inspect and bind.
    timeline : Timeline
        Timeline passed to each binding context.
    logger : SimulationLogger or None, optional
        Logger passed to each binding context. When omitted,
        ``timeline.logger`` is used once and reused.

    Returns
    -------
    tuple[object, ...]
        Entities that were successfully bound.

    Notes
    -----
    The returned tuple preserves input iteration order and omits objects that
    do not expose a bind hook.

    For per-entity context metadata, call ``bind_if_supported`` directly or use
    a higher-level runtime that supplies component IDs.
    """
    bound_entities: list[object] = []
    resolved_logger = timeline.logger if logger is None else logger
    for entity in entities:
        if bind_if_supported(entity, timeline, resolved_logger):
            bound_entities.append(entity)
    return tuple(bound_entities)


__all__ = [
    "BindableMixin",
    "BindingContext",
    "BindingMeta",
    "SupportsBind",
    "bind_if_supported",
    "bind_many",
]
