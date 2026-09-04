"""Reporting helpers for the DPS-QKD example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUMMARY_KEYS = (
    "configured_slots",
    "pulses_emitted",
    "pulses_delivered",
    "preparation_reports",
    "qstate_records",
    "differential_bits",
    "interference_slots",
    # --- what the detectors decided ---------------------------------------
    # These five partition every slot and sum to interference_slots.
    "edge_slots_dropped",
    "slots_with_click",
    "slots_no_click",
    "slots_double_click",
    "sifted_bits",
    "sifted_errors",
    "qber",
    "clicks_per_pulse",
    "raw_clicks",
    "detector_efficiency",
    "dark_count_rate_hz",
    # --- the ideal readout, kept as the reference the detector is judged
    # against: same optics, no detector, no statistics.
    "optical_differential_bits",
    "optical_bits_match_alice",
    "interferometer_mu_in",
    "interferometer_mu_out",
    "temporal_overlap_min",
    "held_arms_at_end",
    "channel_delay_ticks",
    "channel_power_transmission",
    "channel_received",
    "channel_delivered",
    "channel_lost",
    "channel_phase_noise_rad",
    "mean_photon_number_min",
    "mean_photon_number_max",
    "encoding_phase_histogram",
    "carrier_phase_distinct_values",
    "master_seed",
)

# Per-pulse records are useful in Python and unhelpful in a summary or a JSON
# file with 2000 slots in it.
_RECORD_KEYS = (
    "encoding_phase_indices",
    "alice_differential_bits",
    "bob_optical_differential_bits",
    "arrival_ticks",
    "reports",
    "interference_reports",
    "detection_reports",
    "detection_slots_detail",
)

UNMODELLED_PHYSICS = (
    "Finite laser linewidth. FixedCarrierPhase means infinite coherence "
    "length; a Wiener drift on Theta is the honest model and does not exist.",
    "Modulator insertion loss, finite extinction ratio, and intensity "
    "modulator dynamics. Preparation is ideal.",
    "Chirp, dispersion, and pulse broadening. temporal_mode_sigma_s is fixed "
    "per source and nothing widens it.",
    "Channel realism. The fibre is a delay plus an optional uniform "
    "attenuation and an optional per-pulse phase noise, and both are off by "
    "default. Chromatic dispersion, polarization drift, correlated phase "
    "drift and any wavelength dependence are absent.",
    "Photon arrival time within the pulse envelope. A click is timed at the "
    "detection window's start plus jitter, never at a position sampled inside "
    "the Gaussian the pulse occupies. Sub-slot timing resolution is therefore "
    "not modelled; see CAPABILITY_MAP.md section 5 for the asymmetry in "
    "SinglePhotonDetector that has to be fixed before it can be.",
    "Polarization-resolved detection. A polarizing beamsplitter splits a weak "
    "coherent pulse rather than routing it, and polarization_weights has not "
    "shipped. This receiver reads phase, not polarization, so decoy-state "
    "BB84 is still out of reach.",
    "Photon-number resolution. Each port contributes at most one Bernoulli "
    "trial per slot, so P(n >= 2 | click) is not modelled and a double click "
    "here means two ports fired, never two photons in one port.",
    "Interferometer non-idealities. Its optics are ideal by specification: "
    "no insertion loss, no arm imbalance, no splitting-ratio error, and no "
    "thermal or mechanical drift of the arm lengths. Its two detectors are "
    "not ideal -- efficiency, dark counts, dead time, jitter and afterpulsing "
    "are all modelled.",
    "The protocol layer. There is no classical channel here and no agents: "
    "Alice's bits are read from her own preparation reports in-process, not "
    "learned from a message. Sifting over a public channel, error correction "
    "and privacy amplification are step 6.",
)


def summarize_trial(trial: dict[str, Any]) -> str:
    """Return a human-readable counter summary followed by the modelling gaps."""
    lines = [f"{key}: {trial.get(key)}" for key in SUMMARY_KEYS]
    lines.append("")
    lines.append("not modelled:")
    lines.extend(f"  - {note}" for note in UNMODELLED_PHYSICS)
    return "\n".join(lines)


def write_trial_report(trial: dict[str, Any], path: str | Path) -> Path:
    """Write the trial counters to ``path`` as JSON, omitting per-pulse records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in trial.items() if key not in _RECORD_KEYS}
    payload["not_modelled"] = list(UNMODELLED_PHYSICS)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SUMMARY_KEYS",
    "UNMODELLED_PHYSICS",
    "summarize_trial",
    "write_trial_report",
]
