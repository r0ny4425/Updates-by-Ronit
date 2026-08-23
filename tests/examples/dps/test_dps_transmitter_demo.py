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

    # Arrival is emission plus the channel delay, uniformly: the fibre shifts
    # the whole train and preserves its spacing, which is why the interference
    # pattern below is unchanged by inserting it.
    period = trial["slot_period_ticks"]
    delay = trial["channel_delay_ticks"]
    assert delay > 0
    assert trial["arrival_ticks"] == tuple(
        index * period + delay for index in range(SLOTS)
    )

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


def test_the_receiver_optics_reproduce_alices_bits_and_conserve_energy() -> None:
    # The claim the interferometer was wired in to make. Alice's bits come from
    # encoding_phase_index on the control plane; Bob's come from which output
    # port is bright. Two routes, no shared step, and they must agree.
    trial = run_dps_transmitter_trial(num_slots=SLOTS)

    assert trial["bob_optical_differential_bits"] == trial["alice_differential_bits"]
    assert trial["optical_bits_match_alice"] is True
    assert trial["optical_differential_bits"] == SLOTS - 1

    # N pulses give N+1 output slots: the first pulse's short arm and the last
    # pulse's long arm each meet vacuum and carry no bit. Getting this wrong is
    # how a decoder ends up one bit out of step with the key.
    assert trial["interference_slots"] == SLOTS + 1
    assert len(trial["interference_reports"]) == SLOTS + 1

    # Ideal device: nothing held at the end, nothing lost, and tau matches the
    # slot period so every pair overlaps fully.
    assert trial["held_arms_at_end"] == 0
    assert trial["interferometer_mu_out"] == pytest.approx(
        trial["interferometer_mu_in"],
        abs=1e-12,
    )
    # Energy ledger across the whole train: 1/2 mu + (N-1) mu + 1/2 mu.
    assert trial["interferometer_mu_in"] == pytest.approx(
        SLOTS * trial["mean_photon_number_max"],
        abs=1e-12,
    )
    assert trial["temporal_overlap_min"] == pytest.approx(1.0, abs=1e-12)


def test_the_lossless_channel_only_delays_the_train() -> None:
    # The channel's coherent path has no other end-to-end exercise until the
    # detectors arrive, so this is what would catch a wiring mistake. Configured
    # lossless, its whole effect must be a uniform shift.
    trial = run_dps_transmitter_trial(num_slots=SLOTS)

    assert trial["channel_power_transmission"] == 1.0
    assert trial["channel_received"] == SLOTS
    assert trial["channel_delivered"] == SLOTS
    assert trial["channel_lost"] == 0

    # Spacing preserved, so every pair still overlaps fully and the bits are
    # exactly what a run with no channel would give.
    assert trial["temporal_overlap_min"] == pytest.approx(1.0, abs=1e-12)
    assert trial["optical_bits_match_alice"] is True
    assert trial["interferometer_mu_in"] == pytest.approx(
        SLOTS * trial["mean_photon_number_max"],
        abs=1e-12,
    )


def test_attenuation_scales_both_arms_equally_so_the_bits_survive() -> None:
    # An amplitude is never discarded on this path: eta is a power transmission
    # applied deterministically, so lost_count stays 0 however dark the fibre
    # is. Reading lost_count == 0 as "lossless" is the trap.
    lossless = run_dps_transmitter_trial(num_slots=SLOTS)
    lossy = run_dps_transmitter_trial(
        num_slots=SLOTS,
        channel_attenuation_db_per_km=0.2,
    )

    eta = lossy["channel_power_transmission"]
    assert eta == pytest.approx(10 ** (-0.2), abs=1e-12)
    assert lossy["interferometer_mu_in"] == pytest.approx(
        eta * lossless["interferometer_mu_in"],
        abs=1e-12,
    )
    assert lossy["channel_lost"] == 0
    assert lossy["channel_delivered"] == lossy["channel_received"] == SLOTS

    # The point: both interferometer arms are split from the same attenuated
    # pulse, so the interference term and the intensities scale together and
    # which port is bright never changes. Loss costs signal, not key.
    assert lossy["optical_bits_match_alice"] is True

    # Deterministic attenuation consumes no RNG, so a lossy run replays
    # identically at any seed -- the channel's own claim, checked here because
    # this is the only place it runs end to end.
    other_seed = run_dps_transmitter_trial(
        num_slots=SLOTS,
        master_seed=99_991,
        channel_attenuation_db_per_km=0.2,
    )
    assert other_seed["interferometer_mu_in"] == lossy["interferometer_mu_in"]


def test_phase_noise_degrades_the_bit_agreement() -> None:
    # The first place in this build where a physical imperfection becomes a key
    # error rather than a number. The differential phase picks up
    # theta_n - theta_{n-1} from two independent draws, which is exactly what
    # randomize_carrier_phase does at the source -- here it comes from the
    # fibre instead.
    def agreement(sigma_rad: float, seed: int) -> float:
        trial = run_dps_transmitter_trial(
            num_slots=SLOTS,
            master_seed=seed,
            channel_phase_noise_rad=sigma_rad,
        )
        alice = trial["alice_differential_bits"]
        bob = trial["bob_optical_differential_bits"]
        assert len(alice) == len(bob) == SLOTS - 1
        return sum(a == b for a, b in zip(alice, bob)) / len(alice)

    seeds = range(1, 21)

    # At 1 rad the encoding is measurably broken on every seed.
    heavy = [agreement(1.0, seed) for seed in seeds]
    assert max(heavy) < 1.0
    assert sum(heavy) / len(heavy) < 0.95

    # At 0.2 rad it is not broken at all, on any seed, and that is a real
    # property rather than a weak threshold: Bob's bit here is `mu_0 > mu_1`,
    # a noiseless threshold at the midpoint, so a differential phase error only
    # flips a bit once it exceeds pi/2. Visibility degrades continuously long
    # before the bit does. A detector would smear this -- there isn't one yet,
    # which is precisely what makes the distinction worth pinning.
    light = [agreement(0.2, seed) for seed in seeds]
    assert min(light) == 1.0
