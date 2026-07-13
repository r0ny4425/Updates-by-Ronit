"""Runnable single-photon BB84 example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.bb84"

from .reporting import write_trial_report
from .trial import run_bb84_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-photon BB84 example.",
        epilog=(
            "The CLI exposes common knobs. For full control, import "
            "examples.bb84.run_bb84_trial in Python and pass "
            "source_overrides, quantum_channel_overrides, "
            "detector_overrides, or classical_channel_overrides. "
            "See examples/bb84/configs.py for available fields."
        ),
    )
    parser.add_argument("--distance-km", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-slots", type=int, default=30_000)
    parser.add_argument("--emission-probability", type=float, default=None)
    parser.add_argument("--depolarizing-probability", type=float, default=None)
    parser.add_argument("--detector-efficiency", type=float, default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="optional JSONL event log path",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trial = run_bb84_trial(
        distance_km=args.distance_km,
        master_seed=args.seed,
        num_slots=args.num_slots,
        emission_probability=args.emission_probability,
        depolarizing_probability=args.depolarizing_probability,
        detector_efficiency=args.detector_efficiency,
        log_file=args.log_file,
    )

    if args.report_file is not None:
        write_trial_report(trial, args.report_file)

    print(json.dumps(trial, indent=2))
    if args.report_file is not None:
        print("Saved report to:", args.report_file)


if __name__ == "__main__":
    main()
