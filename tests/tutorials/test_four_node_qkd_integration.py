from __future__ import annotations

import json
from dataclasses import replace

import pytest

from simyuj.components import ClassicalChannel, QuantumChannel, SinglePhotonSource
from simyuj.components.detectors import DetectorArray
from simyuj.components.sources import EntangledPairSource
from tutorials.four_node_qkd import (
    FourNodeQKDConfig,
    run_four_node_qkd_trial,
    write_four_node_trial_report,
)
from tutorials.four_node_qkd.trial import build_four_node_qkd_trial


@pytest.fixture(scope="module")
def default_result() -> dict:
    return run_four_node_qkd_trial()


def test_builder_uses_exact_topology_and_real_components() -> None:
    artifacts = build_four_node_qkd_trial()

    assert tuple(artifacts.network.nodes) == ("A", "B", "C", "D")
    assert set(artifacts.network.quantum_links) == {
        "A_to_B_BB84",
        "C_to_B_E91",
        "C_to_D_E91",
    }
    assert all(
        isinstance(channel, QuantumChannel)
        for channel in artifacts.quantum_channels.values()
    )
    assert all(
        isinstance(channel, ClassicalChannel)
        for channel in artifacts.classical_channels.values()
    )
    assert isinstance(artifacts.bb84_source, SinglePhotonSource)
    assert isinstance(artifacts.e91_source, EntangledPairSource)
    assert isinstance(artifacts.b_bb84_detector, DetectorArray)
    assert isinstance(artifacts.b_e91_detector, DetectorArray)
    assert artifacts.b_bb84_detector is not artifacts.b_e91_detector


def test_default_concurrent_trial_finishes_both_keys(default_result: dict) -> None:
    result = default_result

    assert result["concurrency"]["quantum_frames_overlapped"]
    assert result["concurrency"]["bb84_postprocessing_started_before_e91_frame_end"]
    assert result["bb84"]["protocol_complete"]
    assert result["bb84"]["final_key_length"] > 0
    assert result["bb84"]["final_keys_equal"]
    assert result["bb84"]["reconciled_bits_equal"]
    assert (
        result["bb84"]["cascade_leaked_bits"]
        == result["bb84"]["cascade_parity_requests"]
    )
    assert result["e91"]["protocol_complete"]
    assert result["e91"]["final_key_length"] > 0
    assert result["e91"]["final_keys_equal"]
    assert result["e91"]["reconciled_bits_equal"]
    assert (
        result["e91"]["cascade_leaked_bits"] == result["e91"]["cascade_parity_requests"]
    )
    assert result["e91"]["bell_accepted"]
    assert abs(result["e91"]["observed_s"]) > 2
    assert result["protocols_complete"]


def test_b_detector_reports_remain_protocol_isolated(default_result: dict) -> None:
    assert default_result["report_isolation"] == {
        "b_bb84_reports_only_from_bb84_detector": True,
        "b_e91_reports_only_from_e91_detector": True,
    }


def test_json_report_omits_key_material(default_result: dict, tmp_path) -> None:
    report_path = write_four_node_trial_report(
        default_result,
        tmp_path / "four_node_report.json",
    )
    saved = json.loads(report_path.read_text())

    assert saved == default_result

    def all_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from all_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from all_keys(nested)

    assert "final_key" not in set(all_keys(saved))


def test_noisy_e91_aborts_without_stopping_bb84() -> None:
    base = FourNodeQKDConfig(master_seed=2031)
    config = replace(
        base,
        e91_source=replace(base.e91_source, num_slots=5_000),
        e91_c_to_b=replace(base.e91_c_to_b, depolarizing_probability=0.45),
        e91_c_to_d=replace(base.e91_c_to_d, depolarizing_probability=0.45),
    )

    result = run_four_node_qkd_trial(config)

    assert result["bb84"]["protocol_complete"]
    assert not result["e91"]["protocol_complete"]
    assert result["e91"]["d_abort"] is not None
    assert not result["protocols_complete"]


@pytest.mark.parametrize(
    ("e91_slots", "expected_abort"),
    [
        (200, "insufficient E91 CHSH samples"),
        (5_000, "E91 final key too short"),
    ],
)
def test_e91_frame_size_abort_paths(e91_slots: int, expected_abort: str) -> None:
    base = FourNodeQKDConfig(master_seed=2026)
    config = replace(
        base,
        bb84_source=replace(base.bb84_source, num_slots=200),
        e91_source=replace(base.e91_source, num_slots=e91_slots),
    )

    result = run_four_node_qkd_trial(config)

    assert result["e91"]["d_abort"] == expected_abort
    assert not result["e91"]["protocol_complete"]
