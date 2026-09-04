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
    PolarizationSelection,
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


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        # The non-negative branch. require_finite_real's own tests reach nan,
        # inf and bool (tests/primitives/test_validation.py); they do not reach
        # this one, and it is the only thing separating an intensity from a
        # phase.
        (lambda: FixedIntensity(-0.1), ValueError),
        # An empty alphabet would make select_* index out of range at the first
        # pulse, on both alphabet-taking selectors.
        (lambda: RandomPhaseChoice(()), ValueError),
        (lambda: PhaseSequence(()), ValueError),
        # A str is a Sequence, so without the explicit guard "0,pi" would be
        # accepted as a four-phase alphabet of single characters.
        (lambda: RandomPhaseChoice("0,pi"), TypeError),
    ],
)
def test_selector_construction_rejects_the_domain_violations(factory, expected) -> None:
    with pytest.raises(expected):
        factory()


def test_phase_sequence_rejects_non_bool_repeat() -> None:
    # repeat=1 is truthy: without the type check it would wrap where it must
    # raise, which is the silent-wrong-key failure PhaseSequence exists to
    # prevent.
    with pytest.raises(TypeError, match="repeat must be bool"):
        PhaseSequence((0.0,), repeat=1)


def test_dps_phases_orders_zero_before_pi() -> None:
    # The decode convention rests on this ordering; see RandomPhaseChoice's
    # warning about what a reversed alphabet does silently. The default
    # argument is pinned here too, since a changed default would reach every
    # caller that omits the alphabet.
    assert DPS_PHASES == (0.0, pi)
    assert isclose(DPS_PHASES[1], pi)
    assert RandomPhaseChoice().phases == DPS_PHASES


def _valid_selector_kwargs() -> dict[str, object]:
    return {
        "intensity": FixedIntensity(0.2),
        "carrier_phase": FixedCarrierPhase(),
        "encoding_phase": FixedPhase(),
    }


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


# --------------------------------------------------------------------------
# polarization: the selection record only. No selector ships in src/, so the
# ones used here and in test_weak_coherent_pulse_source.py are local fixtures,
# not previews of an alphabet.
# --------------------------------------------------------------------------


def test_polarization_selection_normalizes_its_jones_vector() -> None:
    # int/float components are converted, so an alphabet may be written the
    # readable way.
    selection = PolarizationSelection(jones=(1.0, 0.0), index=0)

    assert selection.jones == (1 + 0j, 0j)
    assert all(isinstance(component, complex) for component in selection.jones)


def test_polarization_selection_rejects_an_unnormalized_jones_vector() -> None:
    # This is the whole reason the check lives here. The source builds its
    # signals with validation_flag=False, so nothing on the emit path would
    # catch a bad vector -- it would be accepted in silence.
    with pytest.raises(ValueError, match="must be normalized"):
        PolarizationSelection(jones=(1.0 + 0j, 1.0 + 0j), index=0)


def test_validate_pulse_selectors_treats_polarization_as_optional() -> None:
    # None is not a missing selector: the other three name a quantity every
    # pulse has, while a pulse need not occupy a described mode at all.
    validate_pulse_selectors(**_valid_selector_kwargs(), polarization=None)

    with pytest.raises(
        TypeError,
        match="polarization must implement select_polarization",
    ):
        validate_pulse_selectors(**_valid_selector_kwargs(), polarization=object())
