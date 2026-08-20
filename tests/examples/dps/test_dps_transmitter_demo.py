"""Tests for the DPS transmitter example.

Running ``demo.py`` is most of its own test: it catches import errors, wiring
errors, and crashes. These five cover what watching stdout does not --
the trial's own slot arithmetic, the decoder, JSON serializability, the CLI
under CI, and the logging invariant. Behaviour of the *source* is tested at the
component level in ``tests/coherent_optics/``, not again through this door.
"""

from __future__ import annotations

import json
from math import pi

import pytest

from examples.dps import demo as demo_module
from examples.dps.configs import DPS_ENCODING_PHASES
from examples.dps.helpers import dps_differential_bit, dps_differential_bits
from examples.dps.reporting import UNMODELLED_PHYSICS, write_trial_report
from examples.dps.trial import run_dps_transmitter_trial

SLOTS = 40


def test_trial_emits_one_pulse_per_slot_and_decodes_every_adjacent_pair() -> None:
    # The slot count is the trial's own claim, not the source's: the stop tick
    # is exclusive and both the nominal slot and the delayed emission must fall
    # before it, so an off-by-one in dps_source_duration_s yields 39 or 41
    # pulses and prints without complaint.
    trial = run_dps_transmitter_trial(num_slots=SLOTS)

    assert trial["configured_slots"] == SLOTS
    assert trial["pulses_emitted"] == SLOTS
    assert trial["pulses_delivered"] == SLOTS
    assert trial["preparation_reports"] == SLOTS
    # A coherent source touches timeline.qstate zero times.
    assert trial["qstate_records"] == 0
    assert trial["differential_bits"] == SLOTS - 1

    period = trial["slot_period_ticks"]
    assert trial["arrival_ticks"] == tuple(index * period for index in range(SLOTS))

    # The report -> decoder join: indices are lifted from the reports, and the
    # bits are derived from those same indices.
    reported = [report.encoding_phase_index for report in trial["reports"]]
    assert trial["encoding_phase_indices"] == tuple(reported)
    assert trial["alice_differential_bits"] == dps_differential_bits(reported)


def test_dps_differential_bit_decodes_against_the_example_alphabet() -> None:
    # The alphabet ordering the decoder assumes. Reversing DPS_ENCODING_PHASES
    # inverts every bit with nothing raising, which is why it is pinned here as
    # well as commented in configs.py and warned about on RandomPhaseChoice.
    assert DPS_ENCODING_PHASES == (0.0, pi)

    assert dps_differential_bit(0, 0) == 0
    assert dps_differential_bit(1, 1) == 0
    assert dps_differential_bit(0, 1) == 1
    assert dps_differential_bit(1, 0) == 1

    with pytest.raises(ValueError, match="must be 0 or 1"):
        dps_differential_bit(0, 2)


def test_report_file_omits_per_pulse_records_and_keeps_the_gap_list(tmp_path) -> None:
    # A CoherentState left in the payload would raise TypeError from
    # json.dumps. Running the demo without --report-file never reaches this.
    trial = run_dps_transmitter_trial(num_slots=SLOTS)
    path = write_trial_report(trial, tmp_path / "dps.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pulses_emitted"] == SLOTS
    assert payload["qstate_records"] == 0
    assert payload["not_modelled"] == list(UNMODELLED_PHYSICS)
    for record_key in ("reports", "arrival_ticks", "encoding_phase_indices"):
        assert record_key not in payload


def test_demo_main_runs_every_cli_path(monkeypatch, capsys, tmp_path) -> None:
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        demo_module.sys,
        "argv",
        [
            "demo",
            "--num-slots",
            "12",
            "--show-pulses",
            "4",
            # Wired to PerPulseRandomCarrierPhase, which is the DPS-destroying
            # policy: this checks the flag reaches the right selector.
            "--randomize-carrier-phase",
            "--report-file",
            str(report_path),
        ],
    )

    demo_module.main()

    out = capsys.readouterr().out
    assert "pulses_emitted: 12" in out
    assert "qstate_records: 0" in out
    assert "carrier_phase_distinct_values: 12" in out
    assert "not modelled:" in out
    assert "phi_enc" in out
    assert report_path.exists()


def test_log_file_is_written_without_changing_the_run(tmp_path) -> None:
    # Logging is observational: the traced run must produce the same record.
    untraced = run_dps_transmitter_trial(num_slots=SLOTS)
    log_path = tmp_path / "dps.jsonl"
    traced = run_dps_transmitter_trial(num_slots=SLOTS, log_file=log_path)

    assert log_path.exists()
    assert log_path.stat().st_size > 0
    assert traced["encoding_phase_indices"] == untraced["encoding_phase_indices"]
    assert traced["arrival_ticks"] == untraced["arrival_ticks"]
