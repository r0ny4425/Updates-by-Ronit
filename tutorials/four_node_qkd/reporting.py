"""Compact console and JSON reporting for the four-node tutorial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_four_node_trial(result: dict[str, Any]) -> str:
    """Return a short human-readable summary without exposing key material."""
    bb84 = result["bb84"]
    e91 = result["e91"]
    lines = [
        "Four-node concurrent BB84 + E91 trial",
        f"  nodes: {', '.join(result['topology']['nodes'])}",
        (
            "  concurrent quantum frames: "
            f"{result['concurrency']['quantum_frames_overlapped']}"
        ),
        (
            "  BB84: "
            f"sifted={bb84['sifted_bits']}, "
            f"QBER={bb84['estimated_qber']}, "
            f"final={bb84['final_key_length']} bits, "
            f"equal={bb84['final_keys_equal']}"
        ),
        (
            "  E91: "
            f"coincidences={e91['coincident_successful_detections']}, "
            f"|S|={abs(e91['observed_s']) if e91['observed_s'] is not None else None}, "
            f"QBER={e91['estimated_qber']}, "
            f"final={e91['final_key_length']} bits, "
            f"equal={e91['final_keys_equal']}"
        ),
        f"  both protocols complete: {result['protocols_complete']}",
    ]
    if bb84["a_abort"] or bb84["b_abort"]:
        lines.append(f"  BB84 aborts: A={bb84['a_abort']!r}, B={bb84['b_abort']!r}")
    if e91["b_abort"] or e91["d_abort"]:
        lines.append(f"  E91 aborts: B={e91['b_abort']!r}, D={e91['d_abort']!r}")
    return "\n".join(lines)


def write_four_node_trial_report(
    result: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write the JSON-serializable summary and return its resolved path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return target.resolve()


__all__ = ["summarize_four_node_trial", "write_four_node_trial_report"]
