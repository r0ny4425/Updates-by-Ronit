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
    "arrival_ticks",
    "reports",
)

UNMODELLED_PHYSICS = (
    "Finite laser linewidth. FixedCarrierPhase means infinite coherence "
    "length; a Wiener drift on Theta is the honest model and does not exist.",
    "Modulator insertion loss, finite extinction ratio, and intensity "
    "modulator dynamics. Preparation is ideal.",
    "Chirp, dispersion, and pulse broadening. temporal_mode_sigma_s is fixed "
    "per source and nothing widens it.",
    "Transport, interference, and detection. There is no amplitude path "
    "through QuantumChannel, no interferometer, and no optical detector yet, "
    "so this trial ends at a tap rather than at a receiver.",
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
