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


@dataclass(frozen=True)
class DPSReceiverConfig:
    """Settings for Bob's delay interferometer.

    There is deliberately **no** ``delay_ticks`` field. The long-arm delay must
    equal the pulse period, and the pulse period is already fixed by
    ``DPSAliceSourceConfig.clock_hz``; a second copy here could drift from it,
    and the interferometer does not validate one against the other -- it cannot,
    since it never sees the clock. ``trial.py`` derives tau from the same
    ``clock_hz`` through ``dps_slot_period_ticks``, which is the same
    define-the-constant-once discipline as ``DPS_ENCODING_PHASES`` above.
    """

    device_id: str = "bob_interferometer"


@dataclass(frozen=True)
class DPSDetectorConfig:
    """Settings for the two detectors inside Bob's interferometer.

    One per output port: index 0 reads ``out_0``, index 1 reads ``out_1``. They
    live inside the interferometer rather than downstream because a DPS receiver
    is one physical unit, and because BS2 produces both amplitudes in one call --
    so one slot decision follows with no arrival buffering.

    **Detector efficiency is not the probability of a click.** For a coherent
    pulse the click probability is ``1 - exp(-eta * mu)``, so at the default
    ``eta = 0.6`` and ``mu = 0.2`` a bright port fires on about 11% of slots and
    a *perfect* detector would still only reach 18%. Most pulses contain no
    photon at all. Raising ``efficiency`` towards 1.0 does not approach
    certainty; raising ``mu`` does, at the cost of the multi-photon security
    that decoy states exist to restore.
    """

    # A good InGaAs APD or a modest SNSPD at 1550 nm.
    efficiency: float = 0.6

    # Dark counts per second per detector. With the 500 ps window below this is
    # 5e-8 events per slot per detector, which is why a default run measures a
    # QBER of exactly zero rather than a small one. `demo.py` exposes the knob
    # so the effect can be made visible.
    dark_count_rate_hz: float = 100.0

    # 10 ns, not BB84's 50 ns. BB84 runs a 100 MHz source, where 50 ns is five
    # slots; DPS runs at 1 GHz, where it would be fifty and would swallow most
    # of the train. 10 ns is a fast SNSPD and still costs real slots at this
    # clock -- the summary reports how many, because it is physics, not a knob
    # chosen to make a number look good.
    dead_time_s: float = 10e-9

    jitter_stddev_s: float = 50e-12

    p_afterpulse: float = 0.001
    afterpulse_decay_s: float = 100e-9

    # Half a slot at 1 GHz, so one slot's window cannot reach into the next.
    detection_window_s: float = 500e-12

    # What a slot reports when both ports fire. "fail" discards it: both
    # detectors seeing light means the interferometer said nothing, and a
    # discarded slot is honest where a guess is not. "random" is the BB84
    # choice and would instead assign a bit that is wrong half the time,
    # feeding a real error rate into the QBER. The rate itself is physics and
    # belongs to the two ports; only this response is protocol.
    double_click_policy: str = "fail"


@dataclass(frozen=True)
class DPSChannelConfig:
    """Settings for the fibre between Alice and Bob.

    **Lossless and phase-noise free by default**, so a default run isolates the
    wiring: the only effect of the channel is a uniform delay. Every pulse is
    shifted by the same number of ticks, so the spacing at BS2 is unchanged,
    the temporal overlap stays 1, and the bright-port pattern is identical to a
    run with no channel at all. Only the arrival ticks move.

    That is a deliberately unphysical default for a 10 km fibre, and it is the
    right one here: an example whose default run already loses light cannot tell
    a wiring mistake from an attenuation. ``demo.py`` exposes both knobs so the
    physics can be switched on explicitly.
    """

    channel_id: str = "alice_to_bob"

    # 10 km of standard fibre. At the repository's default propagation speed of
    # 2e8 m/s this is a 50 us delay, which is 5e7 ticks -- far longer than the
    # whole pulse train, and harmless: the train arrives intact, just later.
    length_m: float = 10_000.0

    # Real standard fibre at 1550 nm is about 0.2 dB/km. Zero here on purpose;
    # see the class docstring. On the coherent path this is a *power*
    # transmission applied deterministically -- alpha -> sqrt(eta) * alpha --
    # with no Bernoulli trial and no RNG consumed.
    attenuation_db_per_km: float = 0.0

    # Per-pulse optical phase noise. Zero here on purpose. Non-zero destroys
    # differential-phase encoding: the differential phase picks up
    # theta_n - theta_{n-1}, which is independent noise, so the bit is lost.
    phase_noise_stddev_rad: float = 0.0
