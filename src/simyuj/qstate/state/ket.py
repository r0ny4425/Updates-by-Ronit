from __future__ import annotations

"""Dense ket-vector payload and handler implementation."""

from dataclasses import dataclass
from math import log2
from typing import Any

import numpy as np

from ..errors import DimensionError, InvalidLayoutError, InvalidStateError, NoiseError
from ..math.const import ATOL
from ..math.linalg import readonly
from ..math.tensor import kron
from ..measure.basis import MeasurementBasis
from ..measure.result import BellResult, MeasurementResult
from ..ops.unitary import Unitary
from ..space.layout import StateLayout
from .check import _assert_payload_layout_compatible_trusted


def _num_qubits_from_length(length: int) -> int:
    if type(length) is not int or length <= 0:
        raise DimensionError("ket vector length must be positive")
    num_qubits = int(log2(length))
    if 2**num_qubits != length:
        raise DimensionError("ket vector length must be a power of two")
    return num_qubits


@dataclass(frozen=True, slots=True, init=False)
class KetState:
    """Pure qubit state-vector payload.

    Parameters
    ----------
    vector : object
        One-dimensional normalized state vector with length ``2**n``.  The input
        is coerced to ``complex128`` and copied into read-only storage.

    Attributes
    ----------
    vector : ndarray of complex, shape ``(2**num_qubits,)``
        Read-only state vector in computational-basis order.

    Raises
    ------
    DimensionError
        If ``vector`` is not one-dimensional or its length is not a positive
        power of two.
    InvalidStateError
        If the norm is zero, non-finite, or not one within ``ATOL``.
    """

    vector: np.ndarray

    def __init__(self, vector: object) -> None:
        """Validate and store a normalized ket vector."""
        array = np.asarray(vector, dtype=np.complex128)
        if array.ndim != 1:
            raise DimensionError("ket vector must be one-dimensional")
        _num_qubits_from_length(int(array.shape[0]))

        norm = float(np.linalg.norm(array))
        if norm <= 0.0 or not np.isfinite(norm):
            raise InvalidStateError("ket vector norm must be positive and finite")
        if abs(norm - 1.0) > ATOL:
            raise InvalidStateError("ket vector must be normalized")

        object.__setattr__(self, "vector", readonly(array))

    @classmethod
    def _from_trusted(cls, vector: object) -> "KetState":
        """Store an internally produced ket vector without invariant checks."""
        state = object.__new__(cls)
        object.__setattr__(
            state,
            "vector",
            readonly(np.asarray(vector, dtype=np.complex128)),
        )
        return state

    @property
    def num_qubits(self) -> int:
        """Number of qubits implied by the vector length."""
        return _num_qubits_from_length(int(self.vector.shape[0]))


class KetHandler:
    """Representation handler for noiseless pure-state payloads.

    Ket handlers support tensor products, unitary evolution, projective
    measurement, and Bell measurement on qubit layouts.  Noise application is
    rejected because channels require density representation.
    """

    rep = "ket"

    def make(self, state: object) -> object:
        """Coerce ``state`` into a ``KetState`` with :func:`make_ket`."""
        from .make import make_ket

        return make_ket(state)

    def tensor(self, left: object, right: object) -> KetState:
        """Return the state-vector tensor product of two ket payloads.

        Parameters
        ----------
        left, right : object
            ``KetState`` payloads.

        Returns
        -------
        KetState
            Tensor product in left-to-right order.

        Raises
        ------
        TypeError
            If either input is not a ``KetState``.
        """
        if not isinstance(left, KetState):
            raise TypeError("left must be KetState")
        if not isinstance(right, KetState):
            raise TypeError("right must be KetState")
        return KetState._from_trusted(kron(left.vector, right.vector))

    def apply(
        self,
        payload: object,
        operation: Unitary,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply a unitary to selected qubit axes of a ket payload."""
        if not isinstance(payload, KetState):
            raise TypeError("payload must be KetState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..ops.apply import apply_unitary_ket

        return apply_unitary_ket(payload, operation, axes)

    def channel(
        self,
        payload: object,
        channel: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Reject noise application on ket representation."""
        raise NoiseError("noise application requires density representation")

    def channel_sampled(
        self,
        payload: object,
        channel: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
        rng: Any,
    ) -> object:
        """Apply a Kraus channel by sampling one pure ket trajectory branch."""
        if not isinstance(payload, KetState):
            raise TypeError("payload must be KetState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..noise.kraus import _apply_kraus_ket_sampled_checked, check_kraus

        return _apply_kraus_ket_sampled_checked(
            payload,
            check_kraus(channel),
            axes=axes,
            rng=rng,
        )

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
        """Measure selected axes with projective ket measurement."""
        if not isinstance(payload, KetState):
            raise TypeError("payload must be KetState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..measure.projective import _measure_ket_checked

        return _measure_ket_checked(
            payload,
            axes=axes,
            basis=basis,
            rng=rng,
            collapse=collapse,
        )

    def measure_bell(
        self,
        payload: object,
        *,
        layout: StateLayout,
        axes: tuple[int, int],
        rng: Any | None = None,
        collapse: bool = True,
    ) -> BellResult:
        """Measure two selected axes in the Bell basis."""
        if not isinstance(payload, KetState):
            raise TypeError("payload must be KetState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..measure.bell import measure_bell_ket

        return measure_bell_ket(
            payload,
            layout=layout,
            axes=axes,
            rng=rng,
            collapse=collapse,
        )

    @staticmethod
    def _check_layout(payload: KetState, layout: StateLayout) -> None:
        try:
            _assert_payload_layout_compatible_trusted(payload, layout)
        except InvalidLayoutError as exc:
            if "qubit axes" in str(exc):
                raise DimensionError("ket operations support qubit axes only") from exc
            raise DimensionError("layout does not match ket payload") from exc

    @staticmethod
    def _check_qubit_axes(layout: StateLayout, axes: tuple[int, ...]) -> None:
        for axis in axes:
            if layout.dim_at(axis) != 2:
                raise DimensionError("ket operations support qubit axes only")


__all__ = ["KetHandler", "KetState"]
