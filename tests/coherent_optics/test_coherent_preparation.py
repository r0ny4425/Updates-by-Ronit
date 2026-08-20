"""Tests for the per-pulse preparation selectors.

These need no ``Timeline``: the selectors are pure frozen strategy objects that
take a caller-supplied RNG, which is exactly the property that lets them be
tested against a two-line fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, pi

import pytest

from simyuj.components.sources.coherent_preparation import (
    DPS_PHASES,
    FixedCarrierPhase,
    FixedIntensity,
    FixedPhase,
    IntensitySelection,
    PerPulseRandomCarrierPhase,
    PhaseSelection,
    PhaseSequence,
    RandomPhaseChoice,
    validate_pulse_selectors,
)


class _TripwireRNG:
    """RNG that fails the test if anything draws from it.

    A selector documented as consuming no randomness must consume none; a
    discarded draw would still shift that stream for every later pulse.
    """

    def __getattr__(self, name: str):
        raise AssertionError(f"selector drew {name}() from an RNG it must not use")


@dataclass(slots=True)
class _ScriptedRNG:
    """RNG returning a fixed script of ``random()`` / ``uniform()`` values."""

    values: list[float]
    draws: int = field(default=0)

    def random(self) -> float:
        value = self.values[self.draws]
        self.draws += 1
        return value

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()


def test_deterministic_selectors_consume_no_randomness() -> None:
    rng = _TripwireRNG()

    assert FixedIntensity(0.2).select_intensity(0, rng) == IntensitySelection(0.2, 0)
    assert FixedCarrierPhase(0.5).select_carrier_phase(0, rng) == 0.5
    assert FixedPhase(pi).select_encoding_phase(0, rng) == PhaseSelection(pi, 0)
    assert PhaseSequence((0.0, pi)).select_encoding_phase(1, rng) == PhaseSelection(
        pi,
        1,
    )


def test_random_phase_choice_draws_once_per_pulse() -> None:
    rng = _ScriptedRNG(values=[0.1, 0.9])
    selector = RandomPhaseChoice(DPS_PHASES)

    assert selector.select_encoding_phase(0, rng) == PhaseSelection(0.0, 0)
    assert selector.select_encoding_phase(1, rng) == PhaseSelection(pi, 1)
    assert rng.draws == 2


def test_random_phase_choice_guards_the_random_equals_one_boundary() -> None:
    # numpy never returns exactly 1.0, but int(1.0 * 2) == 2 would index past
    # the end of a two-phase alphabet if the min() guard were dropped.
    selector = RandomPhaseChoice(DPS_PHASES)
    selection = selector.select_encoding_phase(0, _ScriptedRNG(values=[1.0]))

    assert selection.index == len(DPS_PHASES) - 1
    assert selection.phase_rad == pi


def test_random_phase_choice_default_alphabet_is_the_dps_pair() -> None:
    assert RandomPhaseChoice().phases == (0.0, pi)


def test_per_pulse_random_carrier_phase_draws_on_the_half_open_interval() -> None:
    # uniform(a, b) is [a, b): the low end is reachable, the high end is not.
    assert (
        PerPulseRandomCarrierPhase().select_carrier_phase(
            0,
            _ScriptedRNG(values=[0.0]),
        )
        == -pi
    )

    high = PerPulseRandomCarrierPhase().select_carrier_phase(
        0,
        _ScriptedRNG(values=[1.0 - 1e-16]),
    )
    assert -pi <= high < pi


def test_phase_sequence_is_zero_indexed_by_the_pulse_counter() -> None:
    selector = PhaseSequence((0.25, 0.5, 0.75))
    rng = _TripwireRNG()

    # The first emitted pulse selects with index 0, so it gets phases[0].
    assert selector.select_encoding_phase(0, rng).phase_rad == 0.25
    assert selector.select_encoding_phase(2, rng).phase_rad == 0.75


def test_phase_sequence_raises_on_exhaustion_rather_than_wrapping() -> None:
    selector = PhaseSequence((0.0, pi))

    with pytest.raises(RuntimeError, match="phase sequence exhausted after 2 phases"):
        selector.select_encoding_phase(2, _TripwireRNG())


def test_phase_sequence_repeat_wraps_and_reports_the_wrapped_index() -> None:
    selector = PhaseSequence((0.0, pi), repeat=True)
    selection = selector.select_encoding_phase(5, _TripwireRNG())

    assert selection == PhaseSelection(pi, 1)


def test_selectors_are_pure_so_one_instance_can_drive_two_sources() -> None:
    # No cursor: the same instance queried out of order gives the same answers,
    # which is what lets one selector be shared between components.
    selector = PhaseSequence((0.0, 0.25, 0.5, 0.75))
    rng = _TripwireRNG()

    forward = [selector.select_encoding_phase(i, rng) for i in range(4)]
    backward = [selector.select_encoding_phase(i, rng) for i in reversed(range(4))]

    assert forward == list(reversed(backward))


def test_fixed_intensity_accepts_vacuum_and_bright_pulses() -> None:
    # mu = 0 is coherent vacuum, a real state. There is no upper bound either:
    # "weak" names the source, it is not a validation constraint.
    assert FixedIntensity(0.0).mean_photon_number == 0.0
    assert FixedIntensity(4.0).mean_photon_number == 4.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.2", TypeError),
    ],
)
def test_fixed_intensity_rejects_invalid_mean_photon_number(value, expected) -> None:
    with pytest.raises(expected):
        FixedIntensity(value)


@pytest.mark.parametrize("selector_cls", [FixedPhase, FixedCarrierPhase])
@pytest.mark.parametrize(
    ("value", "expected"),
    [(float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError)],
)
def test_scalar_phase_selectors_reject_invalid_phases(
    selector_cls,
    value,
    expected,
) -> None:
    with pytest.raises(expected):
        selector_cls(value)


@pytest.mark.parametrize("selector_cls", [RandomPhaseChoice, PhaseSequence])
@pytest.mark.parametrize(
    ("phases", "expected"),
    [
        ((), ValueError),
        ((0.0, float("nan")), ValueError),
        ((0.0, True), TypeError),
        ("0,pi", TypeError),
        (0.0, TypeError),
    ],
)
def test_phase_alphabets_are_validated_at_construction(
    selector_cls,
    phases,
    expected,
) -> None:
    with pytest.raises(expected):
        selector_cls(phases)


def test_phase_sequence_rejects_non_bool_repeat() -> None:
    with pytest.raises(TypeError, match="repeat must be bool"):
        PhaseSequence((0.0,), repeat=1)


def test_phase_alphabets_are_normalized_to_float_tuples() -> None:
    selector = RandomPhaseChoice([0, 3])

    assert selector.phases == (0.0, 3.0)
    assert all(isinstance(phase, float) for phase in selector.phases)


def test_dps_phases_orders_zero_before_pi() -> None:
    # The decode convention rests on this ordering; see RandomPhaseChoice's
    # warning about what a reversed alphabet does silently.
    assert DPS_PHASES == (0.0, pi)
    assert isclose(DPS_PHASES[1], pi)


def _valid_selector_kwargs() -> dict[str, object]:
    return {
        "intensity": FixedIntensity(0.2),
        "carrier_phase": FixedCarrierPhase(),
        "encoding_phase": FixedPhase(),
    }


def test_validate_pulse_selectors_accepts_the_built_in_selectors() -> None:
    validate_pulse_selectors(**_valid_selector_kwargs())


@pytest.mark.parametrize(
    ("field_name", "method_name"),
    [
        ("intensity", "select_intensity"),
        ("carrier_phase", "select_carrier_phase"),
        ("encoding_phase", "select_encoding_phase"),
    ],
)
def test_validate_pulse_selectors_names_the_missing_method(
    field_name,
    method_name,
) -> None:
    kwargs = _valid_selector_kwargs()
    kwargs[field_name] = object()

    with pytest.raises(TypeError, match=f"{field_name} must implement {method_name}"):
        validate_pulse_selectors(**kwargs)


def test_validate_pulse_selectors_rejects_a_non_callable_attribute() -> None:
    @dataclass(frozen=True, slots=True)
    class _NotCallable:
        select_intensity: int = 1

    kwargs = _valid_selector_kwargs()
    kwargs["intensity"] = _NotCallable()

    with pytest.raises(TypeError, match="intensity must implement select_intensity"):
        validate_pulse_selectors(**kwargs)
