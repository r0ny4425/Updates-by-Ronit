"""Configuration dataclasses for the DPS-QKD example.

Physical units throughout, converted to simulation ticks at construction by the
components themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

# --------------------------------------------------------------------------
# The encoding alphabet, defined exactly once.
# --------------------------------------------------------------------------
#
# `CoherentPulsePreparationReport.encoding_phase_index` is a position in *this*
# tuple, and the differential-bit decoder reads that index rather than a phase
# in radians. Nothing checks that the two agree.
#
# So: index 0 is phase 0, index 1 is phase pi, and both the source
# configuration below and `helpers.dps_differential_bit` are defined against
# this one constant. Write `(pi, 0.0)` here instead and every differential bit
# inverts -- no exception is raised, and the run still produces a plausible key
# that is wrong. Do not inline the alphabet anywhere else, and do not let a
# caller pass a different one to the source without also passing it to the
# decoder.
DPS_ENCODING_PHASES: tuple[float, ...] = (0.0, pi)


@dataclass(frozen=True)
class DPSAliceSourceConfig:
    """Settings for Alice's weak coherent pulse source."""

    device_id: str = "alice_laser"

    # Pulse repetition rate. 1 GHz is the usual DPS operating point.
    clock_hz: float = 1e9

    # Number of pulse slots to emit.
    num_slots: int = 2_000

    # Mean photon number per pulse. There is no default that is physically
    # neutral, so this one is a choice: 0.2 is a common DPS signal intensity.
    mean_photon_number: float = 0.2

    wavelength_nm: float = 1550.0

    # Field-envelope standard deviation of each pulse. Not converted to ticks:
    # a continuous width would be quantized to integer picoseconds and any
    # overlap computed from it downstream would be quantized with it.
    temporal_mode_sigma_s: float = 30e-12

    # Carrier phase. Held fixed for the whole train, which is what makes the
    # differential phase carry the encoding alone -- Theta_n - Theta_{n-1} is
    # exactly zero. This also means infinite laser coherence length; finite
    # linewidth is not modelled. See the run report.
    carrier_phase_rad: float = 0.0
