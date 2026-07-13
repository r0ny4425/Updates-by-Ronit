from __future__ import annotations

"""Representation-handler registry for qstate payload operations."""

from ..check import normalize_rep
from ..errors import InvalidReprError
from .base import StateHandler


class StateRegistry:
    """Map normalized representation names to state handlers."""

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        """Create an empty handler registry."""
        self._handlers: dict[str, StateHandler] = {}

    def register(self, handler: StateHandler) -> None:
        """Register a handler by its representation name.

        Parameters
        ----------
        handler : StateHandler
            Handler exposing a ``rep`` attribute accepted by ``normalize_rep``.
        """
        rep = normalize_rep(handler.rep)
        self._handlers[rep] = handler

    def get(self, rep: object) -> StateHandler:
        """Return the handler for a representation.

        Parameters
        ----------
        rep : object
            Representation name or alias accepted by ``normalize_rep``.

        Returns
        -------
        StateHandler
            Registered handler.

        Raises
        ------
        InvalidReprError
            If no handler is registered for the normalized representation.
        """
        normalized = normalize_rep(rep)
        try:
            return self._handlers[normalized]
        except KeyError:
            raise InvalidReprError(
                f"no state handler is registered for representation: {normalized}"
            ) from None

    def has(self, rep: object) -> bool:
        """Return whether a representation has a registered handler."""
        return normalize_rep(rep) in self._handlers


__all__ = ["StateRegistry", "normalize_rep"]
