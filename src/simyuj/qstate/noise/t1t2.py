from __future__ import annotations

"""Single-qubit T1/T2 relaxation and dephasing channel construction."""

import math

import numpy as np

from simyuj.primitives.validation import require_non_negative_real

from ..math import matrix as mat
from .base import KrausChannel
from .damping import amplitude_damping
from .dephase import dephasing

T2_REL_TOL = 1.0e-12
T2_ABS_TOL = 1.0e-18


def T1T2NoiseModel(
    T1: object = 0.0,
    T2: object = 0.0,
    *,
    duration: object = 1.0,
) -> KrausChannel:
    """Return a single-qubit T1/T2 relaxation-decoherence channel.

    Parameters
    ----------
    T1 : object, default=0.0
        Relaxation time.  ``0`` disables amplitude damping.
    T2 : object, default=0.0
        Decoherence time.  ``0`` disables dephasing.
    duration : object, default=1.0
        Elapsed time in the same units as ``T1`` and ``T2``.

    Returns
    -------
    KrausChannel
        Single-qubit Kraus channel produced by :func:`t1t2_noise_model`.

    Raises
    ------
    TypeError
        If ``T1``, ``T2``, or ``duration`` is not an ``int`` or ``float``, or is
        a boolean.
    ValueError
        If a value is non-finite, negative, or if positive ``T1`` and ``T2`` do
        not satisfy :math:`T_2 \\le 2T_1` within tolerance.

    Notes
    -----
    If both time constants are zero, this returns an identity Kraus channel.
    The channel applies amplitude damping first, followed by pure dephasing.
    """
    return t1t2_noise_model(T1=T1, T2=T2, duration=duration)


def t1t2_noise_model(
    T1: object = 0.0,
    T2: object = 0.0,
    *,
    duration: object = 1.0,
) -> KrausChannel:
    """Return a single-qubit T1/T2 relaxation-decoherence channel.

    Parameters
    ----------
    T1 : object, default=0.0
        Relaxation time.  ``0`` means no amplitude damping.
    T2 : object, default=0.0
        Decoherence time.  ``0`` means no dephasing.
    duration : object, default=1.0
        Elapsed time in the same units as ``T1`` and ``T2``.

    Returns
    -------
    KrausChannel
        Single-qubit channel named ``"t1t2_noise_model"``.

    Raises
    ------
    TypeError
        If ``T1``, ``T2``, or ``duration`` is not an ``int`` or ``float``, or is
        a boolean.
    ValueError
        If a value is non-finite, negative, or if positive ``T1`` and ``T2`` do
        not satisfy :math:`T_2 \\le 2T_1` within tolerance.

    Notes
    -----
    For positive ``T1`` and ``duration``, the amplitude-damping probability is
    :math:`1 - e^{-t / T_1}`.  For positive ``T2`` and ``duration``, the
    pure-dephasing phase-flip probability is
    :math:`(1 - e^{-t(1 / T_2 - 1 / (2T_1))}) / 2` when ``T1`` is also
    positive, and :math:`(1 - e^{-t / T_2}) / 2` when amplitude damping is
    disabled.

    The implementation uses ``expm1`` for improved numerical accuracy when
    the exponent is close to zero.  The :math:`T_2 \\le 2T_1` physical
    constraint is enforced when both constants are enabled.  When both computed
    probabilities are zero, an identity channel is returned.
    """
    resolved_t1 = require_non_negative_real(
        T1,
        field_name="T1",
        type_name="int or float",
    )
    resolved_t2 = require_non_negative_real(
        T2,
        field_name="T2",
        type_name="int or float",
    )
    resolved_duration = require_non_negative_real(
        duration,
        field_name="duration",
        type_name="int or float",
    )
    _check_t1_t2_pair(resolved_t1, resolved_t2)

    gamma = _amplitude_damping_probability(
        duration=resolved_duration,
        T1=resolved_t1,
    )
    phase_probability = _phase_flip_probability(
        duration=resolved_duration,
        T1=resolved_t1,
        T2=resolved_t2,
    )

    if gamma == 0.0 and phase_probability == 0.0:
        return _identity_channel()

    if gamma == 0.0:
        return KrausChannel(
            dephasing(phase_probability).ops,
            name="t1t2_noise_model",
            arity=1,
        )

    if phase_probability == 0.0:
        return KrausChannel(
            amplitude_damping(gamma).ops,
            name="t1t2_noise_model",
            arity=1,
        )

    damping_channel = amplitude_damping(gamma)
    dephasing_channel = dephasing(phase_probability)

    return KrausChannel(
        tuple(
            dephasing_op @ damping_op
            for damping_op in damping_channel.ops
            for dephasing_op in dephasing_channel.ops
        ),
        name="t1t2_noise_model",
        arity=1,
    )


def _check_t1_t2_pair(T1: float, T2: float) -> None:
    """Enforce the positive-time :math:`T_2 \\le 2T_1` constraint."""
    if T1 == 0.0 or T2 == 0.0:
        return

    ceiling = 2.0 * T1

    if T2 > ceiling and not math.isclose(
        T2,
        ceiling,
        rel_tol=T2_REL_TOL,
        abs_tol=T2_ABS_TOL,
    ):
        raise ValueError("T2 must be <= 2 * T1")


def _amplitude_damping_probability(*, duration: float, T1: float) -> float:
    """Return the T1 amplitude-damping probability."""
    if T1 == 0.0 or duration == 0.0:
        return 0.0

    return -math.expm1(-duration / T1)


def _phase_flip_probability(*, duration: float, T1: float, T2: float) -> float:
    """Return the pure-dephasing phase-flip probability."""
    if T2 == 0.0 or duration == 0.0:
        return 0.0

    dephasing_rate = 1.0 / T2

    if T1 != 0.0:
        dephasing_rate = dephasing_rate - (1.0 / (2.0 * T1))

    if dephasing_rate <= 0.0:
        return 0.0

    return -math.expm1(-duration * dephasing_rate) / 2.0


def _identity_channel() -> KrausChannel:
    """Return the single-qubit identity channel for the T1/T2 model."""
    return KrausChannel(
        (np.array(mat.I2, dtype=np.complex128),),
        name="t1t2_noise_model",
        arity=1,
    )


__all__ = [
    "T1T2NoiseModel",
    "T2_ABS_TOL",
    "T2_REL_TOL",
    "t1t2_noise_model",
]
