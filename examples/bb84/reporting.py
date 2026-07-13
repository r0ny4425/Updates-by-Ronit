"""Reporting helpers for the single-photon BB84 example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUMMARY_KEYS = (
    "prepared_photons",
    "bob_detections",
    "sifted_bits",
    "estimated_qber",
    "qber_accepted",
    "cascade_complete",
    "reconciled_bits_equal",
    "verification_accepted",
    "privacy_complete",
    "final_key_length",
    "final_keys_equal",
    "protocol_complete",
    "alice_abort",
    "bob_abort",
)


def write_trial_report(trial: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trial, indent=2), encoding="utf-8")
    return path


def summarize_trial(trial: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {trial.get(key)}" for key in SUMMARY_KEYS)


__all__ = [
    "SUMMARY_KEYS",
    "summarize_trial",
    "write_trial_report",
]
