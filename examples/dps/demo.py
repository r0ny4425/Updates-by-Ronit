"""Runnable DPS-QKD example: transmitter, fibre, and detecting receiver.

Alice's weak coherent pulse source, a quantum channel, and Bob's delay
interferometer with a detector on each output arm, running on a real timeline.
Alice's differential bits come from her preparation reports; Bob's come from
which port actually clicked.

Four flags turn imperfections on, and each degrades the key in its own way:

- ``--channel-attenuation-db-per-km`` costs clicks and **not** QBER. Both
  interferometer arms are split from the same attenuated pulse, so loss scales
  them together and never moves the light to the wrong port.
- ``--channel-phase-noise-rad`` costs QBER and **not** clicks. A phase shift
  redistributes light between the two ports without destroying any, so the same
  number of slots click and more of them click on the wrong side.
- ``--dark-count-rate-hz`` costs QBER and *adds* clicks: a dark count fires a
  port the light did not.
- ``--detector-efficiency`` costs clicks alone, like loss.
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
        "--channel-attenuation-db-per-km",
        type=float,
        default=None,
        help=(
            "fibre attenuation (default 0.0, i.e. lossless). Real standard "
            "fibre at 1550 nm is about 0.2. On a coherent pulse this is "
            "deterministic -- the amplitude is scaled, never discarded -- so "
            "channel_lost stays 0 and the run still replays at any seed. Both "
            "interferometer arms scale together, so the bits survive."
        ),
    )
    parser.add_argument(
        "--channel-phase-noise-rad",
        type=float,
        default=None,
        help=(
            "per-pulse optical phase noise (default 0.0). Non-zero destroys "
            "the differential-phase encoding, the same way "
            "--randomize-carrier-phase does but from the fibre rather than the "
            "laser. Try 1.0 to see the bit agreement break."
        ),
    )
    parser.add_argument(
        "--detector-efficiency",
        type=float,
        default=None,
        help=(
            "quantum efficiency of both detectors (default 0.6). NOT the "
            "probability of a click: for a coherent pulse that is "
            "1 - exp(-eta*mu), so even 1.0 leaves a bright port silent on most "
            "slots at mu=0.2, because most pulses carry no photon at all. "
            "Lowering it costs clicks and leaves the QBER alone."
        ),
    )
    parser.add_argument(
        "--dark-count-rate-hz",
        type=float,
        default=None,
        help=(
            "dark counts per second per detector (default 100.0, which is "
            "5e-8 per slot and invisible). A dark count fires a port the light "
            "did not, so unlike loss it produces a *wrong* bit rather than a "
            "missing one. Try 1e8 to put a visible floor under the QBER."
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


def _print_key_summary(trial: dict) -> None:
    """Print the slot ledger and the two numbers a QKD run is judged on."""
    slots = trial["interference_slots"]

    def share(count: int) -> str:
        return f"{count:>6}  ({count / slots:6.2%})" if slots else f"{count:>6}"

    qber = trial["qber"]
    print()
    print("receiver")
    print(f"  pulses sent          {trial['pulses_emitted']:>6}")
    print(f"  interference slots   {slots:>6}")
    print(f"  edge slots dropped   {share(trial['edge_slots_dropped'])}")
    print(f"  slots with a click   {share(trial['slots_with_click'])}")
    print(f"  slots with no click  {share(trial['slots_no_click'])}")
    print(f"  double-click slots   {share(trial['slots_double_click'])}")
    print()
    print(f"  sifted bits          {trial['sifted_bits']:>6}")
    print(f"  sifted errors        {trial['sifted_errors']:>6}")
    print(
        "  QBER                 "
        + ("     -" if qber is None else f"{qber:6.2%}")
        + "   (Alice's bit vs Bob's, over the sifted set)"
    )
    print(
        f"  clicks per pulse     {trial['clicks_per_pulse']:>6.4f}"
        "   (end to end: source -> fibre -> optics -> detector)"
    )


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
        channel_attenuation_db_per_km=args.channel_attenuation_db_per_km,
        channel_phase_noise_rad=args.channel_phase_noise_rad,
        detector_efficiency=args.detector_efficiency,
        dark_count_rate_hz=args.dark_count_rate_hz,
        log_file=args.log_file,
    )

    print(summarize_trial(trial))
    _print_key_summary(trial)

    if args.show_pulses:
        _print_pulse_table(trial, args.show_pulses)

    if args.report_file is not None:
        path = write_trial_report(trial, args.report_file)
        print()
        print(f"report written to {path}")


if __name__ == "__main__":
    main()
