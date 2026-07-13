from __future__ import annotations

"""Noise-channel constructors and Kraus application helpers for qstate.

The public noise surface is density-matrix oriented.  Constructors return
``KrausChannel`` records whose operators act on qubit axes in the target order
used by the caller.  Applying those channels is handled by
``apply_kraus_density``, sampled ket trajectory helpers, or by
``QuantumStateManager.apply_noise`` and ``apply_noise_models``.
"""

from .base import (
    KrausChannel,
    NoiseChannel,
    check_kraus_channel,
    check_noise_channel,
    check_noise_models,
)
from .damping import amplitude_damping, generalized_amplitude_damping
from .dephase import common_mode_dephasing, dephasing, phase_damping
from .depolarize import depolarizing, two_qubit_depolarizing
from .kraus import apply_kraus_density, apply_kraus_ket_sampled, check_kraus
from .noisy_gates import imperfect_cnot, imperfect_cz
from .pauli import (
    bit_flip,
    bit_phase_flip,
    pauli_channel,
    phase_flip,
    two_qubit_pauli_channel,
)
from .t1t2 import T1T2NoiseModel, t1t2_noise_model
from .time import DepolarizingNoise, NoiseModel, T1T2Noise, TimeDependentNoiseModel

# Public noise-channel surface for ``simyuj.qstate.noise``.
__all__ = [
    "DepolarizingNoise",
    "NoiseChannel",
    "NoiseModel",
    "KrausChannel",
    "T1T2NoiseModel",
    "T1T2Noise",
    "TimeDependentNoiseModel",
    "amplitude_damping",
    "apply_kraus_density",
    "apply_kraus_ket_sampled",
    "bit_flip",
    "bit_phase_flip",
    "common_mode_dephasing",
    "check_noise_channel",
    "check_noise_models",
    "check_kraus",
    "check_kraus_channel",
    "dephasing",
    "depolarizing",
    "generalized_amplitude_damping",
    "imperfect_cnot",
    "imperfect_cz",
    "pauli_channel",
    "phase_damping",
    "phase_flip",
    "t1t2_noise_model",
    "two_qubit_depolarizing",
    "two_qubit_pauli_channel",
]
