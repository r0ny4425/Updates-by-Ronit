from __future__ import annotations

"""Protocol shared by concrete state-representation handlers."""

from typing import Any, Protocol

from ..measure.basis import MeasurementBasis
from ..measure.result import BellResult, MeasurementResult
from ..ops.unitary import Unitary
from ..space.layout import StateLayout


class StateHandler(Protocol):
    """Structural interface for representation-specific state operations.

    ``QuantumStateManager`` uses handlers to build, tensor, transform, measure,
    and convert payloads without hard-coding the details of each representation.
    Implementations are expected to validate payload type and layout compatibility
    at their public method boundary.
    """

    rep: str

    def make(self, state: object) -> object:
        """Construct or coerce a payload for this representation."""
        ...

    def tensor(self, left: object, right: object) -> object:
        """Return the tensor product of two payloads in left-to-right order."""
        ...

    def apply(
        self,
        payload: object,
        operation: Unitary,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply a unitary operation to target axes of a payload."""
        ...

    def channel(
        self,
        payload: object,
        channel: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply a noise or channel object to target axes of a payload."""
        ...

    def measure(
        self,
        payload: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
        basis: str | MeasurementBasis = "z",
        rng: Any | None = None,
        collapse: bool = True,
    ) -> MeasurementResult:
        """Measure target axes and return a projective measurement result."""
        ...

    def measure_bell(
        self,
        payload: object,
        *,
        layout: StateLayout,
        axes: tuple[int, int],
        rng: Any | None = None,
        collapse: bool = True,
    ) -> BellResult:
        """Measure two target axes in the Bell basis."""
        ...


__all__ = ["StateHandler"]
