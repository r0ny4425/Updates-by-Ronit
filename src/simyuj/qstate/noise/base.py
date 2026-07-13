from __future__ import annotations

"""Noise-channel records and Kraus-operator validation.

Channels in this module describe mathematical noise operations over qubit axes.
``KrausChannel`` stores read-only dense ``complex128`` operators and checks the
Kraus completeness relation before the channel can be applied.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from simyuj.primitives.validation import require_non_negative_real

from ..errors import DimensionError, NoiseError
from ..math.const import ATOL
from ..math.linalg import dagger, readonly
from ..ops.base import check_arity


@dataclass(frozen=True, slots=True)
class NoiseChannel:
    """Base mathematical state-noise channel descriptor.

    Parameters
    ----------
    name : str
        Human-readable channel name.  The stored value is stripped.
    arity : int
        Positive number of qubit axes consumed by the channel.

    Attributes
    ----------
    name : str
        Non-empty channel name.
    arity : int
        Number of qubit operands.

    Raises
    ------
    TypeError
        If ``name`` is not a string or ``arity`` is not exactly an ``int``.
    ValueError
        If ``name`` is empty after stripping or ``arity`` is not positive.
    """

    name: str
    arity: int

    def __post_init__(self) -> None:
        """Normalize channel metadata after dataclass construction."""
        if not isinstance(self.name, str):
            raise TypeError("name must be str")
        name = self.name.strip()
        if not name:
            raise ValueError("name must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arity", check_arity(self.arity))


@dataclass(frozen=True, slots=True, init=False)
class KrausChannel(NoiseChannel):
    """Dense Kraus channel over qubit axes.

    Parameters
    ----------
    ops : tuple of object
        Non-empty tuple of array-like Kraus operators.  Each operator must have
        shape ``(2**arity, 2**arity)``.
    name : str, default="kraus"
        Human-readable channel name.  The stored value is stripped.
    arity : int, default=1
        Positive number of qubit axes consumed by the channel.

    Attributes
    ----------
    ops : tuple of ndarray
        Read-only ``complex128`` Kraus operators.
    name : str
        Non-empty channel name.
    arity : int
        Number of qubit operands.

    Raises
    ------
    TypeError
        If ``arity`` is not exactly an ``int``, ``ops`` is not a tuple, or
        ``name`` is not a string.
    ValueError
        If ``arity`` is not positive, ``ops`` is empty, or ``name`` is empty.
    DimensionError
        If any operator shape does not match ``arity``.
    NoiseError
        If the operators do not satisfy ``sum(K_i.conj().T @ K_i) = I`` within
        package tolerances.
    """

    ops: tuple[np.ndarray, ...]

    def __init__(
        self,
        ops: tuple[object, ...],
        *,
        name: str = "kraus",
        arity: int = 1,
    ) -> None:
        """Validate and store Kraus operators."""
        arity = check_arity(arity)
        checked_ops = _check_kraus_ops(ops, arity=arity)
        NoiseChannel.__init__(self, name=name, arity=arity)
        object.__setattr__(self, "ops", checked_ops)


def check_noise_channel(channel: object) -> NoiseChannel:
    """Return ``channel`` if it is a ``NoiseChannel``.

    Parameters
    ----------
    channel : object
        Candidate noise channel.

    Returns
    -------
    NoiseChannel
        The input value, unchanged.

    Raises
    ------
    TypeError
        If ``channel`` is not a ``NoiseChannel``.
    """
    if not isinstance(channel, NoiseChannel):
        raise TypeError("channel must be NoiseChannel")
    return channel


def _resolve_noise_model(
    model: object,
    *,
    duration_s: float | None,
    field_name: str,
) -> NoiseChannel:
    """Resolve one concrete or duration-dependent noise model."""
    if isinstance(model, NoiseChannel):
        return model

    resolve = getattr(model, "resolve", None)
    if callable(resolve):
        if duration_s is None:
            name = getattr(model, "name", type(model).__name__)
            raise ValueError(f"{name} requires duration_s")

        return check_noise_channel(resolve(duration_s=duration_s))

    raise TypeError(
        f"{field_name} entries must be NoiseChannel or time-dependent noise model"
    )


def check_noise_models(
    noise_models: object,
    *,
    arity: int | None = None,
    field_name: str = "noise_models",
    duration_s: float | None = None,
) -> tuple[NoiseChannel, ...]:
    """Validate and normalize a sequence of noise channels.

    Parameters
    ----------
    noise_models : object
        Sequence of concrete NoiseChannel objects or time-dependent noise
        models.
    arity : int or None, optional
        If provided, every channel must have this arity.
    field_name : str, optional
        Name used in error messages.
    duration_s : float or None, optional
        Elapsed duration in seconds used to resolve time-dependent models.

    Returns
    -------
    tuple[NoiseChannel, ...]
        Validated immutable tuple of noise channels.
    """
    if isinstance(noise_models, (str, bytes)) or not isinstance(
        noise_models,
        Sequence,
    ):
        raise TypeError(
            f"{field_name} must be a sequence of concrete NoiseChannel "
            "objects or time-dependent noise models"
        )

    resolved_duration_s = None
    if duration_s is not None:
        resolved_duration_s = require_non_negative_real(
            duration_s,
            field_name="duration_s",
            type_name="int or float",
        )

    checked = tuple(
        _resolve_noise_model(
            model,
            duration_s=resolved_duration_s,
            field_name=field_name,
        )
        for model in noise_models
    )

    if arity is not None:
        checked_arity = check_arity(arity)
        for model in checked:
            if model.arity != checked_arity:
                raise NoiseError(
                    f"{field_name} entries must be "
                    f"{checked_arity}-qubit noise channels"
                )

    return checked


def check_kraus_channel(channel: object) -> KrausChannel:
    """Return ``channel`` if it is a ``KrausChannel``.

    Parameters
    ----------
    channel : object
        Candidate Kraus channel.

    Returns
    -------
    KrausChannel
        The input value, unchanged.

    Raises
    ------
    TypeError
        If ``channel`` is not a ``KrausChannel``.
    """
    if not isinstance(channel, KrausChannel):
        raise TypeError("channel must be KrausChannel")
    return channel


def _check_kraus_ops(
    ops: tuple[object, ...],
    *,
    arity: int,
) -> tuple[np.ndarray, ...]:
    """Validate Kraus operator shape and completeness."""
    if not isinstance(ops, tuple):
        raise TypeError("ops must be tuple")
    if not ops:
        raise ValueError("ops must be non-empty")

    dim = 2**arity
    total = np.zeros((dim, dim), dtype=np.complex128)
    checked = []
    for op in ops:
        array = np.asarray(op, dtype=np.complex128)
        if array.shape != (dim, dim):
            raise DimensionError(
                "Kraus operator shape does not match noise channel arity"
            )
        checked.append(readonly(array))
        total = total + dagger(array) @ array

    if not np.allclose(total, np.eye(dim), atol=ATOL, rtol=ATOL):
        raise NoiseError("Kraus operators must satisfy completeness")
    return tuple(checked)


__all__ = [
    "NoiseChannel",
    "KrausChannel",
    "check_noise_channel",
    "check_noise_models",
    "check_kraus_channel",
]
