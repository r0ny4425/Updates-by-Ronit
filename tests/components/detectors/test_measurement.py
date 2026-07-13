from __future__ import annotations

from typing import Protocol, cast

import pytest

from simyuj.components.detectors.primitives.measurement import (
    Measure,
    MeasurementCall,
    MeasurementContext,
    execute_measurement_call,
    resolve_measurement_targets,
)
from simyuj.qstate import QuantumStateManager, SubsystemId
from simyuj.qstate.measure import POVM, MeasurementBasis, POVMElement


class _LabeledResult(Protocol):
    label: str


class FakeRNG:
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)


def _context(
    *,
    signal_targets: tuple[SubsystemId, ...] | None = None,
    named_targets: tuple[tuple[str, SubsystemId], ...] = (),
    detector_meta: tuple[tuple[str, object], ...] = (),
) -> MeasurementContext:
    return MeasurementContext(
        device_id="det",
        time=0,
        signal=None,
        signal_targets=signal_targets or (SubsystemId("q0"),),
        detector_meta=detector_meta,
        named_targets=named_targets,
    )


def _z_povm() -> POVM:
    return POVM(
        (
            POVMElement("zero", [[1, 0], [0, 0]]),
            POVMElement("one", [[0, 0], [0, 1]]),
        ),
        name="z-povm",
    )


def test_basis_measure_creates_projective_measurement_call() -> None:
    call = Measure.basis("z").choose(_context())

    assert isinstance(call, MeasurementCall)
    assert call.method == "projective"
    assert isinstance(call.operator, MeasurementBasis)
    assert call.operator.name == "z"
    assert call.targets == "signal"
    assert call.collapse is True
    assert call.label == "z"


def test_measure_from_spec_string_returns_measure() -> None:
    assert isinstance(Measure.from_spec("z"), Measure)


def test_measure_from_spec_existing_measure_returns_same_object() -> None:
    measure = Measure.none()

    assert Measure.from_spec(measure) is measure


def test_target_resolution_supports_signal_indexes_explicit_and_named_targets() -> None:
    q0 = SubsystemId("q0")
    q1 = SubsystemId("q1")
    context = _context(
        signal_targets=(q0, q1),
        named_targets=(("left", q0), ("right", q1)),
    )

    assert resolve_measurement_targets("signal", context) == (q0, q1)
    assert resolve_measurement_targets(0, context) == (q0,)
    assert resolve_measurement_targets(q0, context) == (q0,)
    assert resolve_measurement_targets((q0, q1), context) == (q0, q1)
    assert resolve_measurement_targets("left", context) == (q0,)
    assert resolve_measurement_targets(("left", "right"), context) == (q0, q1)


def test_target_resolution_rejects_invalid_specs() -> None:
    q0 = SubsystemId("q0")
    context = _context(signal_targets=(q0,))

    with pytest.raises(ValueError, match="non-empty"):
        resolve_measurement_targets((), context)

    with pytest.raises(ValueError, match="unique"):
        resolve_measurement_targets((q0, q0), context)

    with pytest.raises(IndexError, match="non-negative"):
        resolve_measurement_targets(-1, context)

    with pytest.raises(TypeError, match="unsupported"):
        resolve_measurement_targets(True, context)

    with pytest.raises(KeyError, match="unknown named"):
        resolve_measurement_targets("missing", context)

    with pytest.raises(TypeError, match="tuple must contain SubsystemId"):
        resolve_measurement_targets(lambda _context: ("q0",), context)


@pytest.mark.parametrize(
    ("choices", "error_type", "message"),
    [
        ({}, ValueError, "non-empty"),
        ({"z": 0.0}, ValueError, "positive"),
        ({"z": -0.1}, ValueError, "positive"),
        ({"z": 0.5, "x": 0.25}, ValueError, "sum to 1"),
        ({"z": "half"}, TypeError, "numeric"),
    ],
)
def test_random_measurement_validates_choices_once(
    choices: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        Measure.random(choices)  # type: ignore[arg-type]


def test_random_measurement_selects_by_rng() -> None:
    q0 = SubsystemId("q0")
    context = MeasurementContext(
        device_id="det",
        time=0,
        signal=None,
        signal_targets=(q0,),
    )

    measurement = Measure.random(
        {
            "z": 0.5,
            "x": 0.5,
        }
    )

    z_call = measurement.choose(context, rng=FakeRNG((0.1,)))
    x_call = measurement.choose(context, rng=FakeRNG((0.9,)))

    assert z_call.label == "z"
    assert z_call.selection_index == 0
    assert z_call.selection_probability == 0.5
    assert z_call.selection_label == "z"

    assert x_call.label == "x"
    assert x_call.selection_index == 1
    assert x_call.selection_probability == 0.5
    assert x_call.selection_label == "x"


def test_by_meta_selects_from_detector_meta() -> None:
    context = _context(detector_meta=(("basis", "x"),))
    measurement = Measure.by_meta("basis", {"z": "z", "x": "x"})

    call = measurement.choose(context)

    assert call.method == "projective"
    assert call.label == "x"
    assert isinstance(call.operator, MeasurementBasis)
    assert call.operator.name == "x"


def test_by_meta_raises_key_error_for_missing_metadata_without_default() -> None:
    measurement = Measure.by_meta("basis", {"z": "z"})

    with pytest.raises(KeyError, match="metadata key not found"):
        measurement.choose(_context())


def test_execute_projective_z_measurement() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    manager.prepare("|0>", subsystems=(q0,))

    context = MeasurementContext(
        device_id="det",
        time=0,
        signal=None,
        signal_targets=(q0,),
    )

    call = Measure.basis("z").choose(context)

    result = execute_measurement_call(
        call=call,
        context=context,
        qstate=manager,
        rng=None,
    )

    assert cast(_LabeledResult, result).label == "0"


def test_execute_projective_measurement_can_discard_after() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    manager.prepare("|0>", subsystems=(q0,))

    context = MeasurementContext(
        device_id="det",
        time=0,
        signal=None,
        signal_targets=(q0,),
    )
    call = Measure.basis("z").choose(context)

    result = execute_measurement_call(
        call=call,
        context=context,
        qstate=manager,
        rng=None,
        discard_after=True,
    )

    assert cast(_LabeledResult, result).label == "0"
    assert manager.size() == 0


def test_execute_povm_measurement() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    manager.prepare("|0>", subsystems=(q0,))

    context = _context(signal_targets=(q0,))
    call = Measure.povm(_z_povm()).choose(context)

    result = execute_measurement_call(
        call=call,
        context=context,
        qstate=manager,
        rng=None,
    )

    assert cast(_LabeledResult, result).label == "zero"


def test_execute_none_measurement_can_discard_targets() -> None:
    manager = QuantumStateManager()
    q0 = SubsystemId("q0")
    manager.prepare("|0>", subsystems=(q0,))

    context = _context(signal_targets=(q0,))
    call = Measure.none(discard=True).choose(context)

    result = execute_measurement_call(
        call=call,
        context=context,
        qstate=manager,
        rng=None,
    )

    assert result is None
    assert manager.size() == 0
