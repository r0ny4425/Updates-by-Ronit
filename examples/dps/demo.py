"""Runnable DPS-QKD transmitter example.

Stage 1 of the DPS build: Alice's weak coherent pulse source, running on a real
timeline and producing real preparation reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.dps"

from .helpers import dps_differential_bit
from .reporting import summarize_trial, write_trial_report
from .trial import run_dps_transmitter_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DPS-QKD transmitter example.",
        epilog=(
            "The CLI exposes common knobs. For full control, import "
            "examples.dps.run_dps_transmitter_trial in Python and pass "
            "source_overrides. See examples/dps/configs.py for the defaults."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--mean-photon-number", type=float, default=None)
    parser.add_argument(
        "--randomize-carrier-phase",
        action="store_true",
        help=(
            "draw an independent carrier phase per pulse. This destroys the "
            "differential-phase encoding; it is here so that failure is "
            "reproducible."
        ),
    )
    parser.add_argument(
        "--show-pulses",
        type=int,
        default=8,
        help="how many per-pulse preparations to print (0 for none)",
    )
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


def _print_pulse_table(trial: dict, limit: int) -> None:
    reports = trial["reports"][:limit]
    if not reports:
        return

    print()
    print(
        f"{'pulse':>5}  {'slot':>8}  {'mu':>6}  {'Theta':>8}  "
        f"{'phi_enc':>8}  {'idx':>3}  {'|alpha|^2':>10}  {'bit':>3}"
    )
    for position, report in enumerate(reports):
        state = report.coherent_state
        if position == 0:
            bit = "-"  # no predecessor to pair with
        else:
            bit = str(
                dps_differential_bit(
                    trial["reports"][position - 1].encoding_phase_index,
                    report.encoding_phase_index,
                )
            )
        print(
            f"{report.pulse_index:>5}  "
            f"{report.emission_slot_tick:>8}  "
            f"{report.mean_photon_number:>6.3f}  "
            f"{report.carrier_phase_rad:>8.4f}  "
            f"{report.encoding_phase_rad:>8.4f}  "
            f"{report.encoding_phase_index:>3}  "
            f"{state.mean_photon_number:>10.6f}  "
            f"{bit:>3}"
        )


def main() -> None:
    args = parse_args()
    trial = run_dps_transmitter_trial(
        master_seed=args.seed,
        num_slots=args.num_slots,
        mean_photon_number=args.mean_photon_number,
        randomize_carrier_phase=args.randomize_carrier_phase,
        log_file=args.log_file,
    )

    print(summarize_trial(trial))

    if args.show_pulses:
        _print_pulse_table(trial, args.show_pulses)

    if args.report_file is not None:
        path = write_trial_report(trial, args.report_file)
        print()
        print(f"report written to {path}")


if __name__ == "__main__":
    main()
