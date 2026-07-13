from __future__ import annotations

import random
from typing import cast

import pytest

from simyuj.components.detectors.primitives.measurement import MeasurementCall
from simyuj.components.detectors.primitives.readout import (
    BasisOutcomeMapReadout,
    DetectorExposure,
    FixedReadout,
    OutcomeMapReadout,
    ReadoutContext,
    ReadoutLayout,
    normalize_readout_exposures,
    readout_from_spec,
    run_qubit_readout,
)
from simyuj.qstate import MeasurementResult, QuantumStateManager, SubsystemId
from simyuj.signal import EncodingScheme, Signal, SignalKind


class Result:
    def __init__(self, label: object) -> None:
        self.label = label


class FlipReadoutModel:
    def report_outcome(
        self,
        *,
        true_outcome: object | None,
        qstate_result: object | None,
        measurement_call: MeasurementCall,
        context: object,
        rng: object | None,
    ) -> object | None:
        del qstate_result, measurement_call, context, rng
        return "1" if true_outcome == "0" else "0"


def _signal() -> Signal:
    return Signal(
        id="sig-0",
        signal_kind=SignalKind.PHOTON,
        encoding_scheme=EncodingScheme.POLARIZATION,
        emission_time=0,
        origin="src",
    )


def _context(
    *,
    detector_ids: tuple[str, ...] = ("d0", "d1"),
    qstate_result: object | None = Result("0"),
) -> ReadoutContext:
    return ReadoutContext(
        detector_ids=detector_ids,
        measurement_call=MeasurementCall(method="projective", label="z"),
        qstate_result=qstate_result,
        signal=_signal(),
    )


def test_detector_exposure_validates_fields() -> None:
    exposure = DetectorExposure(
        detector_id="d0",
        signal_present=False,
        outcome_label="0",
        time_offset_ticks=2,
        meta=(("basis", "z"),),
    )

    assert exposure.detector_id == "d0"
    assert exposure.signal_present is False
    assert exposure.outcome_label == "0"
    assert exposure.time_offset_ticks == 2


def test_fixed_readout_uses_explicit_detector() -> None:
    exposure = FixedReadout("d1").resolve_exposures(_context())[0]

    assert exposure.detector_id == "d1"
    assert exposure.signal_present is True
    assert exposure.outcome_label == "0"


def test_fixed_readout_uses_only_detector_when_implicit() -> None:
    exposures = FixedReadout().resolve_exposures(
        _context(detector_ids=("only",), qstate_result=None)
    )

    assert exposures == (DetectorExposure(detector_id="only"),)


def test_fixed_readout_requires_detector_id_for_multiple_detectors() -> None:
    with pytest.raises(ValueError, match="detector_id is required"):
        FixedReadout().resolve_exposures(_context())


def test_fixed_readout_rejects_unknown_detector_id() -> None:
    with pytest.raises(ValueError, match="unknown readout detector_id"):
        FixedReadout("missing").resolve_exposures(_context())


def test_outcome_map_readout_maps_result_label_to_detector() -> None:
    exposures = OutcomeMapReadout({"0": "d0", "1": "d1"}).resolve_exposures(
        _context(qstate_result=Result("1"))
    )

    assert exposures == (
        DetectorExposure(
            detector_id="d0",
            signal_present=False,
            outcome_label="0",
        ),
        DetectorExposure(detector_id="d1", outcome_label="1"),
    )


def test_outcome_map_readout_maps_zero_to_d0() -> None:
    exposures = OutcomeMapReadout({"0": "D0", "1": "D1"}).resolve_exposures(
        _context(detector_ids=("D0", "D1"), qstate_result=Result("0"))
    )

    assert exposures == (
        DetectorExposure(detector_id="D0", outcome_label="0"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="1",
        ),
    )


def test_outcome_map_readout_maps_plus_to_d0() -> None:
    exposures = OutcomeMapReadout({"+": "D0", "-": "D1"}).resolve_exposures(
        _context(detector_ids=("D0", "D1"), qstate_result=Result("+"))
    )

    assert exposures == (
        DetectorExposure(detector_id="D0", outcome_label="+"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="-",
        ),
    )


def test_outcome_map_readout_maps_e2_to_d2() -> None:
    exposures = OutcomeMapReadout(
        {
            "E0": "D0",
            "E1": "D1",
            "E2": "D2",
        }
    ).resolve_exposures(
        _context(detector_ids=("D0", "D1", "D2"), qstate_result=Result("E2"))
    )

    assert exposures == (
        DetectorExposure(
            detector_id="D0",
            signal_present=False,
            outcome_label="E0",
        ),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="E1",
        ),
        DetectorExposure(detector_id="D2", outcome_label="E2"),
    )


def test_basis_outcome_map_readout_uses_measurement_label() -> None:
    readout = BasisOutcomeMapReadout(
        {
            "z": {
                "0": "D0",
                "1": "D1",
            },
            "x": {
                "+": "D0",
                "-": "D1",
            },
        }
    )

    z_exposures = readout.resolve_exposures(
        _context(
            detector_ids=("D0", "D1"),
            qstate_result=Result("0"),
        )
    )
    x_context = ReadoutContext(
        detector_ids=("D0", "D1"),
        measurement_call=MeasurementCall(method="projective", label="x"),
        qstate_result=Result("+"),
        signal=_signal(),
    )
    x_exposures = readout.resolve_exposures(x_context)

    assert z_exposures == (
        DetectorExposure(detector_id="D0", outcome_label="0"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="1",
        ),
    )
    assert x_exposures == (
        DetectorExposure(detector_id="D0", outcome_label="+"),
        DetectorExposure(
            detector_id="D1",
            signal_present=False,
            outcome_label="-",
        ),
    )


def test_basis_outcome_map_readout_rejects_unknown_measurement_label() -> None:
    readout = BasisOutcomeMapReadout(
        {
            "z": {
                "0": "D0",
                "1": "D1",
            },
        }
    )
    context = ReadoutContext(
        detector_ids=("D0", "D1"),
        measurement_call=MeasurementCall(method="projective", label="x"),
        qstate_result=Result("+"),
        signal=_signal(),
    )

    with pytest.raises(ValueError, match="unmapped readout measurement label"):
        readout.resolve_exposures(context)


def test_outcome_map_readout_supports_explicit_none_mapping() -> None:
    exposures = OutcomeMapReadout({None: "d0"}).resolve_exposures(
        _context(qstate_result=None)
    )

    assert exposures == (DetectorExposure(detector_id="d0"),)


def test_outcome_map_readout_rejects_none_without_mapping() -> None:
    with pytest.raises(ValueError, match="qstate result is None"):
        OutcomeMapReadout({"0": "d0"}).resolve_exposures(_context(qstate_result=None))


def test_outcome_map_readout_rejects_unmapped_label() -> None:
    with pytest.raises(ValueError, match="unmapped readout outcome label"):
        OutcomeMapReadout({"0": "d0"}).resolve_exposures(
            _context(qstate_result=Result("1"))
        )


def test_outcome_map_readout_rejects_mapped_detector_missing_from_context() -> None:
    with pytest.raises(ValueError, match="mapped detector_id does not exist"):
        OutcomeMapReadout({"0": "missing"}).resolve_exposures(_context())


def test_readout_from_spec_none_returns_fixed_readout_for_one_detector() -> None:
    readout = readout_from_spec(None, ("d0",))
    exposures = readout.resolve_exposures(_context(detector_ids=("d0",)))

    assert isinstance(readout, FixedReadout)
    assert exposures == (DetectorExposure(detector_id="d0", outcome_label="0"),)


def test_readout_from_spec_none_rejects_multiple_detectors() -> None:
    with pytest.raises(ValueError, match="readout is required"):
        readout_from_spec(None, ("d0", "d1"))


def test_readout_from_spec_mapping_returns_outcome_map_readout() -> None:
    readout = readout_from_spec({"0": "d0", "1": "d1"}, ("d0", "d1"))

    assert isinstance(readout, OutcomeMapReadout)


def test_readout_from_spec_nested_mapping_returns_basis_outcome_map_readout() -> None:
    readout = readout_from_spec(
        {
            "z": {
                "0": "d0",
                "1": "d1",
            },
        },
        ("d0", "d1"),
    )

    assert isinstance(readout, BasisOutcomeMapReadout)


def test_readout_from_spec_mapping_rejects_unknown_detector_id() -> None:
    with pytest.raises(ValueError, match="mapped detector_id does not exist"):
        readout_from_spec({"0": "missing"}, ("d0",))


def test_readout_from_spec_existing_readout_is_used_as_is() -> None:
    readout = FixedReadout("d0")

    assert readout_from_spec(readout, ("d0",)) is readout
    assert isinstance(readout, ReadoutLayout)


def test_readout_from_spec_custom_object_is_used_as_is() -> None:
    class CustomReadout:
        def resolve_exposures(
            self,
            context: ReadoutContext,
        ) -> tuple[DetectorExposure, ...]:
            return (DetectorExposure(detector_id=context.detector_ids[0]),)

    readout = CustomReadout()

    assert readout_from_spec(readout, ("d0",)) is readout
    assert isinstance(readout, ReadoutLayout)


def test_readout_from_spec_wraps_callable() -> None:
    def custom(context: ReadoutContext) -> tuple[DetectorExposure, ...]:
        return (DetectorExposure(detector_id=context.detector_ids[0]),)

    readout = readout_from_spec(custom, ("d0",))

    assert readout.resolve_exposures(_context(detector_ids=("d0",))) == (
        DetectorExposure(detector_id="d0"),
    )


def test_normalize_readout_exposures_fills_missing_detectors_in_order() -> None:
    exposures = normalize_readout_exposures(
        (
            DetectorExposure(detector_id="d2", outcome_label="2"),
            DetectorExposure(detector_id="d0", outcome_label="0"),
        ),
        detector_ids=("d0", "d1", "d2"),
    )

    assert exposures == (
        DetectorExposure(detector_id="d0", outcome_label="0"),
        DetectorExposure(detector_id="d1", signal_present=False),
        DetectorExposure(detector_id="d2", outcome_label="2"),
    )


def test_normalize_readout_exposures_returns_one_entry_per_detector() -> None:
    exposures = normalize_readout_exposures(
        (),
        detector_ids=("d0", "d1"),
    )

    assert exposures == (
        DetectorExposure(detector_id="d0", signal_present=False),
        DetectorExposure(detector_id="d1", signal_present=False),
    )


def test_normalize_readout_exposures_rejects_non_tuple() -> None:
    with pytest.raises(TypeError, match="readout must return tuple"):
        normalize_readout_exposures(
            [DetectorExposure(detector_id="d0")],
            detector_ids=("d0",),
        )


def test_normalize_readout_exposures_rejects_non_exposure() -> None:
    with pytest.raises(TypeError, match="readout exposures must contain"):
        normalize_readout_exposures(
            (object(),),
            detector_ids=("d0",),
        )


def test_normalize_readout_exposures_rejects_unknown_detector() -> None:
    with pytest.raises(ValueError, match="unknown readout detector_id"):
        normalize_readout_exposures(
            (DetectorExposure(detector_id="missing"),),
            detector_ids=("d0",),
        )


def test_normalize_readout_exposures_rejects_duplicate_detector() -> None:
    with pytest.raises(ValueError, match="duplicate detector_id"):
        normalize_readout_exposures(
            (
                DetectorExposure(detector_id="d0"),
                DetectorExposure(detector_id="d0"),
            ),
            detector_ids=("d0",),
        )


def test_run_qubit_readout_measures_zero_in_z() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    state_ref = qstate.prepare("|0>", subsystems=(q0,))

    report = run_qubit_readout(
        device_id="det",
        time=7,
        targets=(q0,),
        measurement="z",
        collapse=True,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=None,
        readout_rng=None,
        readout_model=None,
        report_id="det:report:0",
    )

    assert report.report_id == "det:report:0"
    assert report.device_id == "det"
    assert report.time == 7
    assert report.success is True
    assert report.outcome == "0"
    assert cast(MeasurementResult, report.qstate_result).state_ref == state_ref
    assert report.measurement_method == "projective"
    assert report.measurement_label == "z"
    assert ("targets", ("q0",)) in report.meta


def test_run_qubit_readout_measures_one_in_z() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    qstate.prepare("|1>", subsystems=(q0,))

    report = run_qubit_readout(
        device_id="det",
        time=0,
        targets=(q0,),
        measurement="z",
        collapse=True,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=None,
        readout_rng=None,
        readout_model=None,
    )

    assert report.outcome == "1"
    assert ("true_label", "1") in report.meta
    assert ("reported_label", "1") in report.meta


def test_run_qubit_readout_collapse_true_writes_collapsed_state() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    qstate.prepare("|+>", subsystems=(q0,))

    report = run_qubit_readout(
        device_id="det",
        time=0,
        targets=(q0,),
        measurement="z",
        collapse=True,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=random.Random(1),
        readout_rng=None,
        readout_model=None,
    )

    assert report.outcome == "0"
    assert qstate.measure(targets=(q0,), basis="z").outcome == (0,)


def test_run_qubit_readout_collapse_false_preserves_state() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    qstate.prepare("|+>", subsystems=(q0,))

    report = run_qubit_readout(
        device_id="det",
        time=0,
        targets=(q0,),
        measurement="z",
        collapse=False,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=random.Random(1),
        readout_rng=None,
        readout_model=None,
    )

    assert report.outcome == "0"
    assert qstate.measure(targets=(q0,), basis="x").label == "+"


def test_run_qubit_readout_applies_readout_model_after_true_label() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    qstate.prepare("|0>", subsystems=(q0,))

    report = run_qubit_readout(
        device_id="det",
        time=0,
        targets=(q0,),
        measurement="z",
        collapse=True,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=None,
        readout_rng=None,
        readout_model=FlipReadoutModel(),
    )

    assert report.success is True
    assert report.outcome == "1"
    assert ("true_label", "0") in report.meta
    assert ("reported_label", "1") in report.meta


def test_run_qubit_readout_report_includes_signal_and_detector_meta() -> None:
    qstate = QuantumStateManager()
    q0 = SubsystemId("q0")
    qstate.prepare("|0>", subsystems=(q0,))
    signal = _signal()

    report = run_qubit_readout(
        device_id="det",
        time=0,
        targets=(q0,),
        measurement="z",
        collapse=True,
        qstate=qstate,
        measurement_choice_rng=None,
        qstate_rng=None,
        readout_rng=None,
        readout_model=None,
        signal=signal,
        input_port_name="qin",
        detector_meta=(("basis_source", "test"),),
    )

    assert report.signal_id == signal.id
    assert ("input_signal_id", signal.id) in report.meta
    assert ("input_port_name", "qin") in report.meta
    assert ("basis_source", "test") in report.meta
