from __future__ import annotations

"""Duration-dependent noise models that resolve to concrete channels.

This public-facing module keeps elapsed-time policy separate from the concrete
Kraus-channel constructors in sibling modules.  The resolved channels use the
same single-qubit conventions as ``depolarizing`` and ``t1t2_noise_model``.
"""

from dataclasses import dataclass
from math import expm1
from typing import Protocol, TypeAlias

from simyuj.primitives.validation import require_non_negative_real

from .base import NoiseChannel
from .depolarize import depolarizing
from .t1t2 import t1t2_noise_model


class TimeDependentNoiseModel(Protocol):
    """Noise model that resolves using elapsed time.

    Attributes
    ----------
    name : str
        Human-readable model name.
    arity : int
        Number of qubit axes consumed by the resolved channel.
    """

    name: str
    arity: int

    def resolve(self, *, duration_s: float) -> NoiseChannel:
        """Return a concrete channel for an elapsed duration.

        Parameters
        ----------
        duration_s : float
            Elapsed time in seconds.

        Returns
        -------
        NoiseChannel
            Concrete noise channel for the supplied elapsed time.
        """
        ...


NoiseModel: TypeAlias = NoiseChannel | TimeDependentNoiseModel


@dataclass(frozen=True, slots=True)
class DepolarizingNoise:
    """Single-qubit depolarizing noise with exponential mixing probability.

    Parameters
    ----------
    rate_hz : float
        Non-negative exponential decay rate, in inverse seconds, for the
        depolarizing mixing probability.
    name : str, default="depolarizing_noise"
        Human-readable model name.
    arity : int, default=1
        Number of qubit axes consumed by the resolved channel.

    Notes
    -----
    Resolving for duration ``t`` computes
    :math:`p = 1 - e^{-\\lambda t}` and passes that probability to
    ``depolarizing``.  The metadata fields are plain dataclass fields and are
    not normalized by this wrapper.
    """

    rate_hz: float
    name: str = "depolarizing_noise"
    arity: int = 1

    def __post_init__(self) -> None:
        """Validate the depolarizing rate."""
        require_non_negative_real(
            self.rate_hz,
            field_name="rate_hz",
            type_name="int or float",
        )

    def resolve(self, *, duration_s: float) -> NoiseChannel:
        """Return the depolarizing channel for ``duration_s`` seconds.

        Parameters
        ----------
        duration_s : float
            Non-negative finite elapsed time in seconds.

        Returns
        -------
        NoiseChannel
            Single-qubit depolarizing channel for the elapsed duration.

        Raises
        ------
        TypeError
            If ``duration_s`` is not an ``int`` or ``float``, or is a boolean.
        ValueError
            If ``duration_s`` is non-finite or negative.
        """
        duration_s = require_non_negative_real(
            duration_s,
            field_name="duration_s",
            type_name="int or float",
        )
        p = -expm1(-float(self.rate_hz) * duration_s)
        return depolarizing(p)


@dataclass(frozen=True, slots=True)
class T1T2Noise:
    """Single-qubit T1/T2 model resolved using elapsed seconds.

    Parameters
    ----------
    T1 : float
        Relaxation time in seconds.  ``0`` disables amplitude damping.
    T2 : float
        Decoherence time in seconds.  ``0`` disables dephasing.
    name : str, default="t1t2_noise"
        Human-readable model name.
    arity : int, default=1
        Number of qubit axes consumed by the resolved channel.

    Notes
    -----
    Validation of ``T1`` and ``T2`` is delegated to ``t1t2_noise_model`` when
    ``resolve`` is called.  The resolved channel applies the existing T1/T2
    constructor with ``duration=duration_s``.
    """

    T1: float
    T2: float
    name: str = "t1t2_noise"
    arity: int = 1

    def resolve(self, *, duration_s: float) -> NoiseChannel:
        """Return the T1/T2 channel for ``duration_s`` seconds.

        Parameters
        ----------
        duration_s : float
            Non-negative finite elapsed time in seconds.

        Returns
        -------
        NoiseChannel
            Single-qubit T1/T2 channel for the elapsed duration.

        Raises
        ------
        TypeError
            If ``duration_s`` is not an ``int`` or ``float``, or is a boolean.
        ValueError
            If ``duration_s`` is non-finite or negative, or if the stored
            ``T1`` and ``T2`` values fail validation in ``t1t2_noise_model``.
        """
        duration_s = require_non_negative_real(
            duration_s,
            field_name="duration_s",
            type_name="int or float",
        )
        return t1t2_noise_model(
            T1=self.T1,
            T2=self.T2,
            duration=duration_s,
        )


__all__ = [
    "DepolarizingNoise",
    "NoiseModel",
    "T1T2Noise",
    "TimeDependentNoiseModel",
]
