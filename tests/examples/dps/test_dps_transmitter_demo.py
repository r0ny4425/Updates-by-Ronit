"""Tests for the DPS transmitter example."""

from __future__ import annotations

import json
from math import pi

import pytest

from examples.dps import demo as demo_module
from examples.dps.configs import DPS_ENCODING_PHASES
from examples.dps.helpers import (
    dps_differential_bit,
    dps_differential_bits,
    dps_source_duration_s,
)
from examples.dps.reporting import UNMODELLED_PHYSICS, write_trial_report
from examples.dps.trial import run_dps_transmitter_trial

SLOTS = 40


def test_trial_emits_one_pulse_per_slot_and_no_quantum_state() -> None:
    trial = run_dps_transmitter_trial(num_slots=SLOTS)

    assert trial["configured_slots"] == SLOTS
    assert trial["pulses_emitted"] == SLOTS
    assert trial["pulses_delivered"] == SLOTS
    assert trial["preparation_reports"] == SLOTS
    # A coherent source touches timeline.qstate zero times.
    assert trial["qstate_records"] == 0
    assert trial["differential_bits"] == SLOTS - 1


def test_pulses_land_on_the_slot_clock() -> None:
    trial = run_dps_transmitter_trial(num_slots=SLOTS)
    period = trial["slot_period_ticks"]

    assert trial["arrival_ticks"] == tuple(index * period for index in range(SLOTS))


def test_intensity_is_constant_and_carrier_phase_is_held() -> None:
    trial = run_dps_transmitter_trial(num_slots=SLOTS, mean_photon_number=0.35)

    assert trial["mean_photon_number_min"] == pytest.approx(0.35)
    assert trial["mean_photon_number_max"] == pytest.approx(0.35)
    assert trial["carrier_phase_distinct_values"] == 1
    assert trial["carrier_phase_randomized"] is False


def test_randomized_carrier_phase_gives_a_distinct_phase_per_pulse() -> None:
    trial = run_dps_transmitter_trial(
        num_slots=SLOTS,
        randomize_carrier_phase=True,
    )

    assert trial["carrier_phase_distinct_values"] == SLOTS
    assert trial["carrier_phase_randomized"] is True


def test_encoding_phases_cover_the_alphabet() -> None:
    trial = run_dps_transmitter_trial(num_slots=200)

    histogram = trial["encoding_phase_histogram"]
    assert len(histogram) == len(DPS_ENCODING_PHASES)
    assert sum(histogram) == 200
    assert all(count > 0 for count in histogram)


def test_trial_replays_from_the_seed_and_differs_across_seeds() -> None:
    def indices(seed: int):
        return run_dps_transmitter_trial(num_slots=SLOTS, master_seed=seed)[
            "encoding_phase_indices"
        ]

    assert indices(2026) == indices(2026)
    assert indices(2026) != indices(2027)


def test_differential_bits_match_the_reported_indices() -> None:
    trial = run_dps_transmitter_trial(num_slots=SLOTS)

    reported = [report.encoding_phase_index for report in trial["reports"]]
    assert trial["encoding_phase_indices"] == tuple(reported)
    assert trial["alice_differential_bits"] == dps_differential_bits(reported)


def test_differential_bit_is_the_alphabet_index_xor() -> None:
    assert dps_differential_bit(0, 0) == 0
    assert dps_differential_bit(1, 1) == 0
    assert dps_differential_bit(0, 1) == 1
    assert dps_differential_bit(1, 0) == 1


def test_differential_bit_rejects_an_index_outside_the_alphabet() -> None:
    with pytest.raises(ValueError, match="must be 0 or 1"):
        dps_differential_bit(0, 2)


def test_encoding_alphabet_orders_zero_before_pi() -> None:
    # The decoder is written against this ordering. Reversing it inverts every
    # bit with nothing raising, which is why it is pinned here as well as
    # commented where it is defined.
    assert DPS_ENCODING_PHASES == (0.0, pi)
    assert dps_differential_bit(0, 1) == 1


def test_source_duration_yields_exactly_the_requested_slot_count() -> None:
    assert dps_source_duration_s(clock_hz=1e9, num_slots=4) == pytest.approx(4e-9)

    with pytest.raises(ValueError, match="at least 1"):
        dps_source_duration_s(clock_hz=1e9, num_slots=0)


def test_report_file_omits_per_pulse_records_and_keeps_the_gap_list(tmp_path) -> None:
    trial = run_dps_transmitter_trial(num_slots=SLOTS)
    path = write_trial_report(trial, tmp_path / "dps.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pulses_emitted"] == SLOTS
    assert payload["qstate_records"] == 0
    assert payload["not_modelled"] == list(UNMODELLED_PHYSICS)
    for record_key in ("reports", "arrival_ticks", "encoding_phase_indices"):
        assert record_key not in payload


def test_demo_main_runs_and_prints_a_summary(monkeypatch, capsys, tmp_path) -> None:
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
            "--report-file",
            str(report_path),
        ],
    )

    demo_module.main()

    out = capsys.readouterr().out
    assert "pulses_emitted: 12" in out
    assert "qstate_records: 0" in out
    assert "not modelled:" in out
    # Header plus four rows, the first of which has no predecessor to pair with.
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
