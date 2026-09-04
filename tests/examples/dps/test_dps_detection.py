"""Detection in the DPS example: the slot ledger, the decoder, and the QBER.

``test_dps_transmitter_demo.py`` covers the transmitter and the receiver
*optics*, at exact amplitudes and exact ticks. It stays the ground truth: the
ideal readout it checks -- ``mu_0 > mu_1``, no detector, no statistics -- is what
the detected bits are compared against here.

This file covers only what the two detectors add. Every claim about a rate is
swept over eight seeds and two thousand slots, because one run of a Bernoulli
trial asserts nothing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from math import exp

import pytest

from examples.dps import demo as demo_module
from examples.dps.configs import DPSAliceSourceConfig, DPSDetectorConfig
from examples.dps.helpers import dps_detected_bit, dps_differential_bit
from examples.dps.reporting import SUMMARY_KEYS, write_trial_report
from examples.dps.trial import run_dps_transmitter_trial
from simyuj.components.detectors import FLAG_DOUBLE_CLICK, FLAG_NO_CLICK
from simyuj.components.interferometers import PORT_OUT_0, PORT_OUT_1

SLOTS = 2000
SEEDS = tuple(range(1, 9))


@lru_cache(maxsize=None)
def _run(seed: int, **kwargs) -> dict:
    """One trial, cached: several tests share the same baseline sweep."""
    return run_dps_transmitter_trial(num_slots=SLOTS, master_seed=seed, **kwargs)


def _sweep(**kwargs) -> dict:
    """Aggregate a whole seed sweep into the rates a claim is about."""
    sifted = errors = clicks = pulses = raw = 0
    ideal_perfect = True
    for seed in SEEDS:
        trial = _run(seed, **kwargs)
        sifted += trial["sifted_bits"]
        errors += trial["sifted_errors"]
        clicks += trial["slots_with_click"] + trial["slots_double_click"]
        pulses += trial["pulses_emitted"]
        raw += trial["raw_clicks"]
        ideal_perfect &= trial["optical_bits_match_alice"]
    return {
        "sifted": sifted,
        "errors": errors,
        "qber": errors / sifted if sifted else None,
        "click_rate": clicks / pulses,
        "clicks_per_pulse": raw / pulses,
        "ideal_readout_perfect": ideal_perfect,
    }


# --------------------------------------------------------------------------
# the slot ledger
# --------------------------------------------------------------------------


def test_every_slot_lands_in_exactly_one_bucket() -> None:
    # T1. The four counters partition the run. If they stop summing, a slot is
    # being counted twice or silently swallowed, and every rate below is
    # computed against a denominator that no longer means anything.
    trial = _run(SEEDS[0])

    total = (
        trial["edge_slots_dropped"]
        + trial["slots_with_click"]
        + trial["slots_no_click"]
        + trial["slots_double_click"]
    )
    assert total == trial["interference_slots"] == trial["detection_slots"]

    # N pulses give N+1 combinations, of which exactly two meet vacuum.
    assert trial["interference_slots"] == SLOTS + 1
    assert trial["edge_slots_dropped"] == 2


def test_the_dropped_slots_are_the_vacuum_ones_and_the_rest_stay_aligned() -> None:
    # T2. Dropping by position rather than by pulse index would still give the
    # right *count* and put every later bit one step out of phase with Alice --
    # a failure that produces a plausible key and a QBER near 50%.
    trial = _run(SEEDS[0])
    slots = trial["detection_slots_detail"]

    edges = [slot for slot in slots if slot.kind == "edge"]
    assert len(edges) == 2
    # First combination: the opening pulse's short arm met vacuum, so there is
    # no predecessor. Last: the flushed long arm met vacuum, no successor.
    assert edges[0].long_pulse_index is None
    assert edges[0].short_pulse_index == 1
    assert edges[1].short_pulse_index is None
    assert edges[1].long_pulse_index == SLOTS
    assert all(slot.alice_bit is None and slot.bob_bit is None for slot in edges)

    # Every surviving slot names two adjacent pulses, previous first, and its
    # Alice bit is joined from those two indices rather than from its position.
    indices = trial["encoding_phase_indices"]
    for slot in slots:
        if slot.kind == "edge":
            continue
        assert slot.short_pulse_index == slot.long_pulse_index + 1
        expected = dps_differential_bit(
            indices[slot.long_pulse_index - 1],
            indices[slot.short_pulse_index - 1],
        )
        if slot.bob_bit is not None:
            assert slot.alice_bit == expected


# --------------------------------------------------------------------------
# the number that matters
# --------------------------------------------------------------------------


def test_a_default_run_measures_a_qber_of_exactly_zero() -> None:
    # T3. Exactly zero, not merely small, and that is a statement about the
    # wiring rather than a claim about physics: the channel is lossless and
    # noiseless, so the only thing that could put a floor under this is dark
    # counts, and at 100 Hz against a 500 ps window there are effectively none.
    # A non-zero result here means a decoder or an alignment bug, not noise.
    result = _sweep()

    assert result["errors"] == 0
    assert result["qber"] == 0.0
    assert result["sifted"] > 1000


def test_bobs_bit_is_the_port_that_fired() -> None:
    # T4. The decoder's whole contract, checked against the real reports rather
    # than against itself, and against the port constants rather than against
    # string literals copied into the helper.
    trial = _run(SEEDS[0])
    reports = {
        dict(report.meta)["interference_index"]: report
        for report in trial["detection_reports"]
    }

    checked = 0
    for slot in trial["detection_slots_detail"]:
        if slot.bob_bit is None:
            continue
        report = reports[slot.interference_index]
        assert slot.bob_bit == dps_detected_bit(report.outcome)
        assert report.outcome in (PORT_OUT_0, PORT_OUT_1)
        checked += 1
    assert checked == trial["sifted_bits"] > 0

    # The convention itself, pinned against the constants. Equal phases
    # interfere constructively at port 1 and give bit 0; a pi step moves the
    # light to port 0 and gives bit 1. Swapping these inverts every key.
    assert dps_detected_bit(PORT_OUT_0) == 1
    assert dps_detected_bit(PORT_OUT_1) == 0


def test_a_slot_without_a_bit_is_refused_rather_than_defaulted() -> None:
    # T5. A no-click slot and a discarded double click must be counted, never
    # mapped to a bit. Returning 0 for None would quietly halve the QBER by
    # padding the sifted set with coin flips that happen to be right half the
    # time.
    for absent in (None, "", "out_2", 0, 1):
        with pytest.raises(ValueError, match="port label"):
            dps_detected_bit(absent)


# --------------------------------------------------------------------------
# one assertion per physical claim
# --------------------------------------------------------------------------


def test_loss_costs_clicks_and_not_qber() -> None:
    # T6. Both interferometer arms are split from the same attenuated pulse, so
    # eta scales the interference term and the intensities together and never
    # moves light to the wrong port. Loss costs signal, not key -- which is why
    # a QKD link can tolerate a great deal of it.
    rates = []
    for db_per_km in (0.0, 0.2, 1.0):
        result = _sweep(channel_attenuation_db_per_km=db_per_km)
        rates.append(result["click_rate"])
        # The claim: at every level, not merely on average.
        assert result["errors"] == 0
        assert result["qber"] == 0.0

    # Strictly monotone, and by a wide margin rather than within noise.
    assert rates[0] > rates[1] > rates[2]
    assert rates[2] < rates[0] / 5


def test_phase_noise_costs_qber_and_not_clicks() -> None:
    # T7. A phase shift redistributes light *between* the two ports without
    # destroying any, so the same slots click and more of them click on the
    # wrong side. That is the opposite signature to loss, and the pair is what
    # makes the two distinguishable in a run summary.
    results = {
        sigma: _sweep(channel_phase_noise_rad=sigma) for sigma in (0.0, 0.3, 0.6, 1.0)
    }

    qbers = [results[sigma]["qber"] for sigma in (0.0, 0.3, 0.6, 1.0)]
    assert qbers == sorted(qbers)
    assert qbers[0] == 0.0
    assert qbers[-1] > 0.25

    # Click rate is flat: no light is lost, only moved.
    baseline = results[0.0]["click_rate"]
    for sigma in (0.3, 0.6, 1.0):
        assert results[sigma]["click_rate"] == pytest.approx(baseline, rel=0.1)


def test_dark_counts_cost_qber_and_add_clicks() -> None:
    # T8. A dark count fires a port the light did not, so it both adds a click
    # and gets it wrong about half the time. Third distinct signature: loss
    # takes clicks away, phase noise leaves them alone, dark counts add them.
    quiet = _sweep()
    noisy = _sweep(dark_count_rate_hz=1e8)

    assert noisy["click_rate"] > quiet["click_rate"]
    assert noisy["qber"] > 0.1
    assert quiet["qber"] == 0.0


def test_a_weaker_detector_costs_clicks_alone() -> None:
    # T9. Detector efficiency enters the same exponent the channel's eta does,
    # so it behaves like loss: fewer clicks, no errors.
    weak = _sweep(detector_efficiency=0.1)

    assert weak["click_rate"] < _sweep()["click_rate"] / 3
    assert weak["qber"] == 0.0


def test_a_double_click_is_counted_apart_from_a_silence() -> None:
    # T10. Under the default "fail" policy both are unsuccessful reports, and
    # collapsing them would hide the one that means the interferometer said
    # something contradictory rather than nothing at all. Driven with a heavy
    # dark rate, which is the only thing that makes both ports fire often.
    trial = _run(SEEDS[0], dark_count_rate_hz=2e9)

    assert trial["slots_double_click"] > 0
    assert trial["slots_no_click"] > 0

    kinds = {
        dict(report.meta)["interference_index"]: report
        for report in trial["detection_reports"]
    }
    for slot in trial["detection_slots_detail"]:
        report = kinds[slot.interference_index]
        if slot.kind == "double_click":
            assert FLAG_DOUBLE_CLICK in report.flags
            # "fail" discards it, so it contributes no bit either way.
            assert slot.bob_bit is None
            assert slot.alice_bit is None
        elif slot.kind == "no_click":
            assert report.flags == (FLAG_NO_CLICK,)
            assert slot.bob_bit is None


# --------------------------------------------------------------------------
# the detector against the optics
# --------------------------------------------------------------------------


def test_the_detector_diverges_from_the_ideal_readout_under_phase_noise() -> None:
    # T11. The reason both readouts are kept. The ideal one thresholds at the
    # midpoint -- `mu_0 > mu_1` -- so a differential phase error only flips a
    # bit once it exceeds pi/2, and at 0.2 rad it is still perfect on every
    # seed. A detector instead samples the *ratio*: port 1 stays brighter but
    # port 0 is no longer dark, so it sometimes fires first.
    #
    # The measured figure is load-bearing documentation, so it is pinned rather
    # than described. p = 0.01829 over n = 1148 sifted bits gives a binomial
    # standard deviation of 0.00396; the tolerance below is three of those.
    clean = _sweep(channel_phase_noise_rad=0.0)
    assert clean["ideal_readout_perfect"] is True
    assert clean["qber"] == 0.0

    smeared = _sweep(channel_phase_noise_rad=0.2)
    assert smeared["ideal_readout_perfect"] is True
    assert smeared["qber"] == pytest.approx(0.0183, abs=0.012)
    assert smeared["qber"] > 0.0


def test_clicks_per_pulse_matches_the_dead_time_model() -> None:
    # T12. Derived rather than recorded. Without dead time a bright port clicks
    # at 1 - exp(-eta*mu) = 11.3%, but the measured rate is 7.2% and the gap
    # has to be accounted for or the number is just an observation.
    #
    # The bright port alternates with the bit, so each detector sees light on
    # about half the slots: p/2 per slot. A 10 ns dead time is 10 slots at
    # 1 GHz, so in steady state each channel clicks at (p/2) / (1 + (p/2)*D),
    # and there are two of them. Dead time costs about a third of the raw
    # clicks, and that is physics rather than a tuned constant.
    trial = _run(SEEDS[0])
    assert trial["clicks_per_pulse"] == pytest.approx(
        trial["raw_clicks"] / trial["pulses_emitted"]
    )

    source = DPSAliceSourceConfig()
    detector = DPSDetectorConfig()
    slot_period_s = 1.0 / source.clock_hz
    dead_time_slots = detector.dead_time_s / slot_period_s

    p_bright = 1.0 - exp(-detector.efficiency * source.mean_photon_number)
    per_detector = (p_bright / 2) / (1 + (p_bright / 2) * dead_time_slots)
    predicted = 2 * per_detector

    measured = _sweep()["clicks_per_pulse"]
    assert measured == pytest.approx(predicted, rel=0.03)
    # And the dead time really is what accounts for the gap.
    assert measured < p_bright / 1.4


# --------------------------------------------------------------------------
# the demo and the report
# --------------------------------------------------------------------------


def test_the_demo_prints_the_receiver_ledger_and_honours_every_new_flag(
    monkeypatch,
    capsys,
) -> None:
    # T13. The counters exist to be read, so this checks they reach stdout, and
    # that the four new flags reach the trial rather than being parsed and
    # dropped.
    monkeypatch.setattr(
        demo_module.sys,
        "argv",
        [
            "demo",
            "--num-slots",
            "400",
            "--show-pulses",
            "0",
            "--channel-attenuation-db-per-km",
            "0.2",
            "--channel-phase-noise-rad",
            "0.5",
            "--detector-efficiency",
            "0.9",
            "--dark-count-rate-hz",
            "1e7",
        ],
    )

    demo_module.main()
    out = capsys.readouterr().out

    for line in (
        "pulses sent",
        "interference slots",
        "edge slots dropped",
        "slots with a click",
        "slots with no click",
        "double-click slots",
        "sifted bits",
        "QBER",
        "clicks per pulse",
    ):
        assert line in out

    # The flags landed: the summary echoes the two detector values, and a
    # lossy phase-noisy run cannot report a zero QBER.
    assert "detector_efficiency: 0.9" in out
    assert "dark_count_rate_hz: 10000000.0" in out
    assert "QBER                 0.00%" not in out


def test_the_json_report_keeps_every_counter_and_drops_only_bulk_records(
    tmp_path,
) -> None:
    # T14. The exclusion list is the only thing standing between this file and
    # a TypeError, so it is easy to over-prune it and quietly lose a counter.
    #
    # The rule asserted here: everything in SUMMARY_KEYS survives, and what is
    # dropped is dropped for one of two stated reasons.
    trial = run_dps_transmitter_trial(num_slots=200)
    payload = json.loads(
        write_trial_report(trial, tmp_path / "dps.json").read_text(encoding="utf-8")
    )

    # Nothing a reader would look for is missing.
    for key in SUMMARY_KEYS:
        assert key in payload, key
    assert payload["qber"] is not None
    assert payload["sifted_bits"] > 0

    # Dropped because they are objects json cannot encode -- they carry
    # CoherentState, RawClick tuples and nested dataclasses.
    for key in (
        "reports",
        "interference_reports",
        "detection_reports",
        "detection_slots_detail",
    ):
        assert key in trial
        assert key not in payload

    # Dropped because they are per-pulse bulk: one entry per slot, and every
    # aggregate a report needs is already a counter above. These are plain
    # tuples of ints and would serialize fine; the reason is size, not type.
    for key in (
        "encoding_phase_indices",
        "alice_differential_bits",
        "bob_optical_differential_bits",
        "arrival_ticks",
    ):
        assert key in trial
        assert key not in payload
        assert all(isinstance(value, int) for value in trial[key])
