from __future__ import annotations

"""Dense density-matrix payload and handler implementation."""

from dataclasses import dataclass
from math import log2
from typing import Any

import numpy as np

from ..errors import DimensionError, InvalidLayoutError, InvalidStateError
from ..math.const import ATOL
from ..math.linalg import dagger, is_hermitian, is_psd, readonly, trace
from ..math.tensor import expand_operator, kron
from ..measure.basis import MeasurementBasis
from ..measure.result import BellResult, MeasurementResult
from ..ops.unitary import Unitary
from ..space.layout import StateLayout
from .check import _assert_payload_layout_compatible_trusted


def _num_qubits_from_size(size: int) -> int:
    if type(size) is not int or size <= 0:
        raise DimensionError("density matrix size must be positive")
    num_qubits = int(log2(size))
    if 2**num_qubits != size:
        raise DimensionError("density matrix size must be a power of two")
    return num_qubits


@dataclass(frozen=True, slots=True, init=False)
class DensityState:
    """Dense qubit density-matrix payload.

    Parameters
    ----------
    rho : object
        Density matrix with shape ``(2**n, 2**n)``.  The input is coerced to
        ``complex128`` and copied into read-only storage.

    Attributes
    ----------
    rho : ndarray of complex
        Read-only density matrix.

    Raises
    ------
    DimensionError
        If ``rho`` is not square or its dimension is not a positive power of
        two.
    InvalidStateError
        If ``rho`` is not Hermitian, trace-one, or positive semidefinite within
        package tolerances.
    """

    rho: np.ndarray

    def __init__(self, rho: object) -> None:
        """Validate and store a density matrix."""
        array = np.asarray(rho, dtype=np.complex128)

        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise DimensionError("density matrix must be square")

        _num_qubits_from_size(int(array.shape[0]))

        if not is_hermitian(array):
            raise InvalidStateError("density matrix must be Hermitian")

        tr = trace(array)
        if abs(tr.imag) > ATOL or abs(tr.real - 1.0) > ATOL:
            raise InvalidStateError("density matrix trace must be one")

        if not is_psd(array):
            raise InvalidStateError("density matrix must be positive semidefinite")

        object.__setattr__(self, "rho", readonly(array))

    @classmethod
    def _from_trusted(cls, rho: object) -> "DensityState":
        """Store an internally produced density matrix without invariant checks."""
        state = object.__new__(cls)
        object.__setattr__(
            state,
            "rho",
            readonly(np.asarray(rho, dtype=np.complex128)),
        )
        return state

    @property
    def matrix(self) -> np.ndarray:
        """Alias for ``rho`` used by conversion and compatibility code."""
        return self.rho

    @property
    def num_qubits(self) -> int:
        """Number of qubits implied by the matrix dimension."""
        return _num_qubits_from_size(int(self.rho.shape[0]))


class DensityHandler:
    """Representation handler for density-matrix states.

    Density handlers support tensor products, unitary evolution, Kraus-channel
    noise, projective measurement, and Bell measurement on qubit layouts.
    """

    rep = "density"

    def make(self, state: object) -> object:
        """Coerce ``state`` into a ``DensityState``.

        Existing density payloads are returned unchanged.  Ket payloads and ket
        constructor inputs are converted through :func:`ket_to_density`; matrix-
        shaped array-like inputs are passed directly to ``DensityState``.
        """
        from .ket import KetState

        if isinstance(state, DensityState):
            return state
        if isinstance(state, KetState):
            from .convert import ket_to_density

            return ket_to_density(state)

        array = (
            np.asarray(state, dtype=np.complex128)
            if not isinstance(state, str)
            else None
        )
        if array is not None and array.ndim == 2:
            return DensityState(array)

        from .convert import ket_to_density
        from .make import make_ket

        return ket_to_density(make_ket(state))

    def tensor(self, left: object, right: object) -> DensityState:
        """Return the density-matrix tensor product of two payloads.

        Parameters
        ----------
        left, right : object
            ``DensityState`` payloads.

        Returns
        -------
        DensityState
            Tensor product in left-to-right order.

        Raises
        ------
        TypeError
            If either input is not a ``DensityState``.
        """
        if not isinstance(left, DensityState):
            raise TypeError("left must be DensityState")
        if not isinstance(right, DensityState):
            raise TypeError("right must be DensityState")
        return DensityState._from_trusted(kron(left.rho, right.rho))

    def apply(
        self,
        payload: object,
        operation: Unitary,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> DensityState:
        """Apply a unitary to selected qubit axes of a density payload."""
        if not isinstance(payload, DensityState):
            raise TypeError("payload must be DensityState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        full_op = expand_operator(
            operation.matrix,
            axes=axes,
            num_qubits=payload.num_qubits,
        )
        return DensityState._from_trusted(full_op @ payload.rho @ dagger(full_op))

    def channel(
        self,
        payload: object,
        channel: object,
        *,
        layout: StateLayout,
        axes: tuple[int, ...],
    ) -> object:
        """Apply a Kraus-style noise channel to selected qubit axes."""
        if not isinstance(payload, DensityState):
            raise TypeError("payload must be DensityState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..noise.kraus import _apply_kraus_density_checked, check_kraus

        return _apply_kraus_density_checked(
            payload,
            check_kraus(channel),
            axes=axes,
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
        """Measure selected axes with projective density-matrix measurement."""
        if not isinstance(payload, DensityState):
            raise TypeError("payload must be DensityState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..measure.projective import _measure_density_checked

        return _measure_density_checked(
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
        if not isinstance(payload, DensityState):
            raise TypeError("payload must be DensityState")
        self._check_layout(payload, layout)
        self._check_qubit_axes(layout, axes)

        from ..measure.bell import measure_bell_density

        return measure_bell_density(
            payload,
            layout=layout,
            axes=axes,
            rng=rng,
            collapse=collapse,
        )

    @staticmethod
    def _check_layout(payload: DensityState, layout: StateLayout) -> None:
        try:
            _assert_payload_layout_compatible_trusted(payload, layout)
        except InvalidLayoutError as exc:
            if "qubit axes" in str(exc):
                raise DimensionError(
                    "density operations support qubit axes only"
                ) from exc
            raise DimensionError("layout does not match density payload") from exc

    @staticmethod
    def _check_qubit_axes(layout: StateLayout, axes: tuple[int, ...]) -> None:
        for axis in axes:
            if layout.dim_at(axis) != 2:
                raise DimensionError("density operations support qubit axes only")


__all__ = ["DensityHandler", "DensityState"]
