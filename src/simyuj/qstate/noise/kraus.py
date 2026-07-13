from __future__ import annotations

"""Apply dense Kraus channels to density-matrix and sampled ket payloads.

Density application expands local Kraus operators to the full Hilbert space and
uses the exact channel ``sum(K rho K.conj().T)``.  Sampled ket application
applies each local Kraus operator directly to selected tensor axes, draws one
branch with probability ``||K_i |psi>||^2``, and stores the normalized pure
branch ``K_i |psi> / sqrt(p_i)``; repeated sampled trajectories reproduce the
density channel statistically.  This module assumes input payloads have already
validated their representation invariants.
"""

from typing import Any

import numpy as np

from ..errors import NoiseError
from ..math.const import ATOL
from ..math.linalg import dagger, trace
from ..math.tensor import apply_operator_to_axes, expand_operator
from ..measure.sample import sample_probs
from ..state.density import DensityState
from ..state.ket import KetState
from .base import KrausChannel, check_kraus_channel

_EXPANDED_KRAUS_CACHE_MAX_SIZE = 256
_EXPANDED_KRAUS_CACHE: dict[
    tuple[object, ...],
    tuple[tuple[np.ndarray, np.ndarray], ...],
] = {}


def check_kraus(channel: object) -> KrausChannel:
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
    return check_kraus_channel(channel)


def apply_kraus_density(
    state: DensityState,
    channel: object,
    *,
    axes: tuple[int, ...],
) -> DensityState:
    """Apply a Kraus channel to selected axes of a density state.

    Parameters
    ----------
    state : DensityState
        Density payload to update.
    channel : object
        ``KrausChannel`` whose arity must match ``len(axes)``.
    axes : tuple of int
        Target axes in the operand order expected by each local Kraus operator.

    Returns
    -------
    DensityState
        New density payload after applying the channel.

    Raises
    ------
    TypeError
        If ``state`` is not a ``DensityState``, ``channel`` is not a
        ``KrausChannel``, ``axes`` is not a tuple, or delegated axis type checks
        fail.
    ValueError
        If ``len(axes)`` does not match ``channel.arity`` or delegated axis
        checks fail.
    DimensionError
        If expanding a Kraus operator finds incompatible dimensions.
    NoiseError
        If the resulting matrix trace is not one within package tolerances.

    Notes
    -----
    Axis uniqueness, range, and local matrix shape checks are delegated to
    ``expand_operator``.  The trace-preservation check catches drift after the
    expanded-channel application.
    """
    if not isinstance(state, DensityState):
        raise TypeError("state must be DensityState")
    checked = check_kraus_channel(channel)
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != checked.arity:
        raise ValueError("axes count must match noise channel arity")

    return _apply_kraus_density_generic(state, checked, axes=axes)


def apply_kraus_ket_sampled(
    state: KetState,
    channel: object,
    *,
    axes: tuple[int, ...],
    rng: Any,
) -> KetState:
    """Sample one Kraus branch on selected axes of a ket state.

    Parameters
    ----------
    state : KetState
        Pure-state payload to update.
    channel : object
        ``KrausChannel`` whose arity must match ``len(axes)``.
    axes : tuple of int
        Target axes in the operand order expected by each local Kraus operator.
    rng : object
        Explicit random stream used to sample non-deterministic branch
        probabilities.

    Returns
    -------
    KetState
        Normalized pure post-branch ket.

    Notes
    -----
    This is a Monte Carlo trajectory realization of a Kraus channel. A single
    call keeps one pure branch; ensemble averages over repeated runs recover
    exact density-channel statistics.
    """
    if not isinstance(state, KetState):
        raise TypeError("state must be KetState")
    checked = check_kraus_channel(channel)
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != checked.arity:
        raise ValueError("axes count must match noise channel arity")

    return _apply_kraus_ket_sampled_checked(state, checked, axes=axes, rng=rng)


def _apply_kraus_density_checked(
    state: DensityState,
    channel: KrausChannel,
    *,
    axes: tuple[int, ...],
) -> DensityState:
    """Apply a validated Kraus channel to a density payload."""
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != channel.arity:
        raise ValueError("axes count must match noise channel arity")

    if state.num_qubits == 1 and channel.arity == 1 and axes == (0,):
        rho = np.zeros_like(state.rho)
        for op in channel.ops:
            rho = rho + op @ state.rho @ dagger(op)
        return _trace_checked_density(rho)

    return _apply_kraus_density_generic(state, channel, axes=axes)


def _apply_kraus_ket_sampled_checked(
    state: KetState,
    channel: KrausChannel,
    *,
    axes: tuple[int, ...],
    rng: Any,
) -> KetState:
    """Apply a validated Kraus channel by sampling one ket branch."""
    if not isinstance(axes, tuple):
        raise TypeError("axes must be tuple")
    if len(axes) != channel.arity:
        raise ValueError("axes count must match noise channel arity")

    branch_vectors = []
    probabilities = []
    for op in channel.ops:
        if state.num_qubits == 1 and channel.arity == 1 and axes == (0,):
            branch = op @ state.vector
        else:
            branch = apply_operator_to_axes(
                state.vector,
                op,
                axes=axes,
                num_qubits=state.num_qubits,
            )
        branch_vectors.append(branch)
        probabilities.append(float(np.vdot(branch, branch).real))

    branch_index = sample_probs(probabilities, rng=rng)
    probability = probabilities[branch_index]
    if probability <= ATOL:
        raise NoiseError("sampled Kraus branch has zero probability")
    return KetState._from_trusted(branch_vectors[branch_index] / np.sqrt(probability))


def _apply_kraus_density_generic(
    state: DensityState,
    channel: KrausChannel,
    *,
    axes: tuple[int, ...],
) -> DensityState:
    """Apply a Kraus channel through full-operator expansion."""
    rho = np.zeros_like(state.rho)
    for full_op, full_op_dagger in _expanded_kraus_ops(
        channel,
        axes=axes,
        num_qubits=state.num_qubits,
    ):
        rho = rho + full_op @ state.rho @ full_op_dagger
    return _trace_checked_density(rho)


def _expanded_kraus_ops(
    channel: KrausChannel,
    *,
    axes: tuple[int, ...],
    num_qubits: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return cached full-space Kraus operators and adjoints."""
    key = (
        id(expand_operator),
        axes,
        num_qubits,
        channel.arity,
        tuple(_array_cache_key(op) for op in channel.ops),
    )
    cached = _EXPANDED_KRAUS_CACHE.get(key)
    if cached is not None:
        return cached

    expanded = []
    for op in channel.ops:
        full_op = expand_operator(op, axes=axes, num_qubits=num_qubits)
        full_op_dagger = dagger(full_op)
        full_op.setflags(write=False)
        full_op_dagger.setflags(write=False)
        expanded.append((full_op, full_op_dagger))

    cached = tuple(expanded)
    if len(_EXPANDED_KRAUS_CACHE) >= _EXPANDED_KRAUS_CACHE_MAX_SIZE:
        _EXPANDED_KRAUS_CACHE.clear()
    _EXPANDED_KRAUS_CACHE[key] = cached
    return cached


def _array_cache_key(array: np.ndarray) -> tuple[tuple[int, ...], str, bytes]:
    """Return a stable content key for small dense operators."""
    checked = np.asarray(array, dtype=np.complex128)
    return checked.shape, checked.dtype.str, checked.tobytes()


def _trace_checked_density(rho: np.ndarray) -> DensityState:
    """Return trusted density state after trace-preservation check."""
    tr = trace(rho)
    if abs(tr.imag) > ATOL or abs(tr.real - 1.0) > ATOL:
        raise NoiseError("Kraus channel did not preserve trace")
    return DensityState._from_trusted(rho)


__all__ = ["apply_kraus_density", "apply_kraus_ket_sampled", "check_kraus"]
